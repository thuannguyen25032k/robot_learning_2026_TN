"""
Mock GRP Model that replays trajectories instead of learning a model.
This mock replays the first trajectory from the dataset, returning actions
sequentially until terminated == True.
"""

import numpy as np
import torch
from torch import nn
from dreamerV3 import GRPBase


class ReplayModel(GRPBase):
    """
    A mock GRP model that replays trajectories from the dataset.
    Instead of learning to predict actions, it stores and replays a trajectory
    sequentially, returning the next action each time it's called.
    """
    
    def __init__(self, cfg, dataset=None):
        """
        Initialize the ReplayModel.
        
        Args:
            cfg: Configuration object
            dataset: Dataset object containing trajectories
        """
        super(ReplayModel, self).__init__(cfg)
        self._cfg = cfg
        self.dataset = dataset
        
        # Trajectory storage
        self.trajectory = None
        self.current_step = 0
        self.trajectory_loaded = False
        
        # Load the first trajectory if dataset is provided
        if dataset is not None:
            self._load_first_trajectory()

    def set_dataset(self, dataset):
        """
        Set the dataset for the model.
        
        Args:
            dataset: Dataset object containing trajectories
        """
        self.dataset = dataset
        self._load_first_trajectory()   
    
    def _load_first_trajectory(self):
        """
        Load the first trajectory from the dataset.
        Extracts states, actions, and termination flags up to the first terminated=True.
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided to load trajectories")
        
        # Try to get the first trajectory from the dataset
        # This assumes the dataset has a method to access trajectories
        # Adjust based on your actual dataset structure
        # Get first trajectory using method __getitem__ of dataset
        trajectory = self.dataset.get_trajectory(0)  # Assuming dataset has a method to get trajectory by index
        
        self.trajectory = trajectory
        self.current_step = 0
        self.trajectory_loaded = True
        
        # Find the step where terminated == True
        self.terminal_step = 0
        for i, step_data in enumerate(trajectory):
            if step_data.get('terminated', False) or step_data.get('done', False):
                self.terminal_step = i
                break
            else:
                # If no terminal step found, use the entire trajectory
                self.terminal_step = len(trajectory) - 1
    
    def load_trajectory(self, trajectory_data):
        """
        Manually load a trajectory.
        
        Args:
            trajectory_data: List of step dictionaries containing 'observation', 'action', 'terminated', etc.
        """
        self.trajectory = trajectory_data
        self.current_step = 0
        self.trajectory_loaded = True
        
        # Find the terminal step
        self.terminal_step = len(trajectory_data) - 1
        for i, step_data in enumerate(trajectory_data):
            if step_data.get('terminated', False) or step_data.get('done', False):
                self.terminal_step = i
                break
    
    def reset(self):
        """Reset the current step to the beginning of the trajectory."""
        self.current_step = 0
        return self.trajectory[0]['init_state'] if self.trajectory_loaded else None
    
    def forward(self, observations, text_goal=None, goal_image=None, targets=None, pose=None, mask_=False, prev_actions=None):
        """Forward pass that returns the next action in the replay trajectory.

        This evaluation code (`sim_eval.py`) expects the model output to be a dict
        containing an `actions` tensor, so we mirror the interface of the learned
        models.
        """
        if self.current_step > self.terminal_step:
            step_data = self.trajectory[self.terminal_step]
            action = step_data.get('action', None)
            print(f"ReplayModel reached terminal step {self.terminal_step}. Returning last action: {action}")
            return {"actions": action.unsqueeze(0), "loss": None}
        step_data = self.trajectory[self.current_step]
        action = step_data.get('action', None)
        self.current_step += 1  # Move to the next step for the next call
        print(f"ReplayModel step {self.current_step}/{self.terminal_step}, action: {action}, reward: {step_data.get('reward', None)}")
        return {"actions": action.unsqueeze(0), "loss": None}

    def get_trajectory_info(self):
        """
        Get information about the currently loaded trajectory.
        
        Returns:
            dict: Information about the trajectory including length and terminal step
        """
        if not self.trajectory_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "trajectory_length": len(self.trajectory),
            "terminal_step": self.terminal_step,
            "current_step": self.current_step,
            "episode_complete": self.current_step > self.terminal_step,
        }
    
    def is_episode_complete(self):
        """Check if the trajectory replay has reached the terminal state."""
        return self.current_step > self.terminal_step
    
    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        return 0
    
    def preprocess_state(self, state):
        return state
    
    def preprocess_goal_image(self, goal_img): 
        return goal_img
    def decode_action(self, action):
        return action
