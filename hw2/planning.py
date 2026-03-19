try:
    # When imported as a package: `from hw2.planning import CEMPlanner`
    from .dreamerV3 import GRPBase
except ImportError:
    # When executed with cwd=hw2/
    from dreamerV3 import GRPBase
import numpy as np
import torch
import os
from pathlib import Path


class Planner(GRPBase):
    """
    Base class for planners. Defines the interface for planning algorithms.
    """
    def __init__(self, cfg=None):
        super(Planner, self).__init__(cfg)

    def update(self, states, actions):
        """
        Update the planner's internal model or policy based on collected states and actions.
        This method can be overridden by planners that learn from data (e.g., PolicyPlanner).
        
        Args:
            states: Tensor of shape (B, state_dim) containing collected states
            actions: Tensor of shape (B, action_dim) containing collected actions
        """
        pass  # Default implementation does nothing
    
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences given an initial state.
        
        Args:
            initial_state: Dictionary containing initial state information
            return_best_sequence: If True, returns the best action sequence; else returns action mean
            
        Returns:
            actions: Tensor of shape (horizon, action_dim) with the planned action sequence
            predicted_reward: Float value of the expected cumulative reward for the planned sequence
        """
        raise NotImplementedError("Plan method must be implemented by subclasses")

class CEMPlanner(Planner):
    """
    Cross-Entropy Method (CEM) planner for model-based planning.
    Samples action sequences and uses a world model to find high-reward plans.
    """
    def __init__(self, 
                 world_model,
                 action_dim,
                 cfg):
        """
        Initialize CEM planner.
        
        Args:
            world_model: World model (DreamerV3 or SimpleWorldModel) used for imagining future trajectories
            action_dim: Dimensionality of the action space
            cfg: Configuration object
        """
        # TODO: Part 1.3 - Initialize CEM planner
        ## Set up world model reference and determine if using DreamerV3 or SimpleWorldModel
        super(CEMPlanner, self).__init__(cfg)
        self.world_model = world_model
        self.action_dim = action_dim
        self.type = 'cem'
        self.cfg = cfg
        self.device = self._cfg.device
        self.to(self.device)

        # CEM hyperparameters (extract from cfg or set defaults)
        self.horizon = getattr(cfg.planner, 'horizon', 12)  # Planning horizon
        self.K = getattr(cfg.planner, 'num_samples', 100)  # Number of action sequences to sample
        self.M = getattr(cfg.planner, 'num_elites', 20)  # Number of elite sequences to select
        self.L = getattr(cfg.planner, 'num_iterations', 3)  # Number of CEM iterations
        self.alpha = getattr(cfg.planner, 'temperature', 0.1)  # Temperature for updating distribution

        self.mu = torch.zeros(self.horizon, self.action_dim, device=self.cfg.device)  # Initial mean of action distribution
        self.std = torch.ones(self.horizon, self.action_dim, device=self.cfg.device)  # Initial std of action distribution

    def load_world_model(self, path):
        """Load pretrained world model from checkpoint."""

        # Hydra may change the working directory (e.g., into `outputs/...`).
        # Resolve relative paths against the hw2 folder (where `checkpoints/` lives).
        p = os.path.expandvars(os.path.expanduser(str(path)))
        p_path = Path(p)
        if not p_path.is_absolute():
            hw2_root = Path(__file__).resolve().parent
            p_path = (hw2_root / p_path).resolve()

         # Safer loading: `weights_only=True` avoids unpickling arbitrary objects.
        self.world_model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences using CEM to maximize predicted rewards.
        
        Args:
            initial_state: Dictionary containing initial state 
                          - For DreamerV3: {'h', 'z', 'z_probs'}
                          - For SimpleWorldModel: {'pose'}
            return_best_sequence: If True, returns best action sequence; else returns action mean
            
        Returns:
            best_actions: Tensor of shape (horizon, action_dim) with the best action sequence
            best_reward: Float value of the sum of predicted rewards for the best sequence
        """
        # TODO: Part 1.3 - Implement CEM planning algorithm
        ## Sample action sequences, evaluate with world model, select elites, update distribution
        # Reset action distribution each planning call to avoid bias from previous plans
        mu = torch.zeros((self.horizon, self.action_dim), device=self.device)
        std = torch.ones((self.horizon, self.action_dim), device=self.device)*6
        # top_std = self.world_model.encode_action(torch.ones_like(mu))
        # bottom_std = self.world_model.encode_action(-torch.ones_like(mu))

        for iteration in range(self.L):
            # Sample action sequences from the current distribution
            action_sequences = mu + std * torch.randn(self.K, self.horizon, self.action_dim, device=self.device)  # (K, H, action_dim)
            # action_sequences = torch.clamp(action_sequences, bottom_std, top_std)  # Ensure actions are within valid range
            # Evaluate the sampled action sequences using the world model
            rewards = self._evaluate_sequences(initial_state, action_sequences)  # (K,)
            
            # Select the top M elite sequences based on rewards
            elite_indices = torch.topk(rewards, self.M).indices  # (M,)
            elite_action_sequences = action_sequences[elite_indices]  # (M, H, action_dim)
            
            # Update the mean and std of the action distribution using the elites
            new_mu = elite_action_sequences.mean(dim=0)  # (H, action_dim)
            # We could use unbiased=False to avoid NaN when M==1 (avoids division by N-1=0)
            new_std = elite_action_sequences.std(dim=0, unbiased=True).clamp(min=1e-3)  # (H, action_dim)
            
            # Smoothly update the distribution parameters with temperature alpha
            mu = self.alpha * new_mu + (1 - self.alpha) * mu
            std = self.alpha * new_std + (1 - self.alpha) * std
            # Track best action sequence and reward for return
            best_actions = elite_action_sequences[0] if return_best_sequence else mu  # (H, action_dim) - best sequence from elites
            best_reward = rewards[elite_indices[0]]  # Best reward from elites

        return best_actions, best_reward

    def _evaluate_sequences(self, initial_state, action_sequences):
        """
        Evaluate a batch of action sequences by rolling them out in the world model.
        
        Args:
            initial_state: Dictionary with initial state (RSSM state for DreamerV3 or pose for SimpleWorldModel)
            action_sequences: Tensor of shape (num_samples, horizon, action_dim)
            
        Returns:
            rewards: Tensor of shape (num_samples,) with sum of predicted rewards
        """
        # TODO: Part 1.3 - Route to appropriate evaluation method
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if self.cfg.model_type == 'dreamer':
            return self._evaluate_sequences_dreamer(initial_state, action_sequences)
        else:
            return self._evaluate_sequences_simple(initial_state, action_sequences)
    
    def _evaluate_sequences_dreamer(self, initial_state, action_sequences):
        """
        Evaluate sequences using DreamerV3 RSSM-based rollout.
        """
        # TODO: Part 3.3 - Implement CEM planning with DreamerV3
        ## Roll out action sequences in the DreamerV3 world model and compute total rewards
        if not isinstance(initial_state, dict) or 'h' not in initial_state or 'z' not in initial_state:
            raise ValueError(
                "DreamerV3 planning requires initial_state with keys {'h','z'} (and optionally 'z_probs')."
            )

        state = {
            'h': initial_state['h'].repeat(self.K, 1).to(self.device),  # (K, deter_dim)
            'z': initial_state['z'].repeat(self.K, 1).to(self.device),  # (K, stoch_dim)
            'z_probs': initial_state.get('z_probs', None)
            }

        self.world_model.eval()
        full_feat = []
        with torch.no_grad():
            for t in range(self.horizon):
                a_t = action_sequences[:, t, :]
                # RSSM imagination step: no embed -> sample from prior.
                step_out = self.world_model.rssm_step(state, a_t, embed=None)
                h = step_out['h']
                z = step_out['z']

                feat = torch.cat([h, z], dim=-1)
                full_feat.append(feat)
                state = {
                    'h': h,
                    'z': z,
                    'z_probs': step_out.get('z_probs', None),
                }
            full_feat = torch.stack(full_feat, dim=1)  # (K, H, feat_dim)
            r_t = self.world_model.reward_head(full_feat.view(self.K, self.horizon, -1)).view(self.K, self.horizon)  # (K, H) because reward_head outputs normal distribution parameters and we take the mean as the predicted reward at each step
            total_rewards = r_t.sum(dim=-1)  # Sum rewards over the horizon to get total reward for each sequence (K,)
        return total_rewards
    
    def _evaluate_sequences_simple(self, initial_state, action_sequences):
        """
        Evaluate sequences using SimpleWorldModel pose-based rollout.
        """
        # TODO: Part 1.3 - Implement CEM planning with SimpleWorldModel
        ## Roll out action sequences using SimpleWorldModel and compute total rewards
        current_state = initial_state['pose']  # (B, pose_dim)
        # 1. Broadcast current state to match the batch size of action sequences
        current_state = current_state.expand(action_sequences.shape[0], -1, -1).contiguous()  # (K, pose_dim)
        # 2. Initialize total rewards tensor on the same device as current_state
        total_rewards = torch.zeros(action_sequences.shape[0], device=self.cfg.device)  # (K,)
        # 3. Roll out each action sequence in the world model and accumulate rewards
        for step in range(self.horizon):  # Iterate over each step in the horizon
            action = action_sequences[:, step, :].unsqueeze(1).to(self.cfg.device)  # (K, 1, action_dim)
            self.world_model.eval()  # Set world model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for evaluation
                next_pose_pred, reward_pred = self.world_model.predict_next_pose(current_state, action)  # (K, 1, pose_dim), (K, 1)
            total_rewards += reward_pred.squeeze(-1)  # Accumulate rewards (K,)
            current_state = next_pose_pred  # Because next_pose_pred is in the same normalized space expected by the world model for the next step
        return total_rewards  # (K,)
    
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that works with both DreamerV3 and SimpleWorldModel.
        This wrapper obtains the current state and plans actions.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (for DreamerV3)
            prev_actions: Previous actions (optional, for state initialization)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Planned action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': Expected cumulative reward
                - 'final_state': Final state after processing inputs
        """
        # TODO: Part 1.3 - Route forward pass to appropriate model
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if self.cfg.model_type == 'dreamer':
            return self._forward_dreamer(observations, prev_actions, prev_state, return_full_sequence)
        else:
            return self._forward_simple(pose, return_full_sequence)
    
    def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
        """Forward pass for DreamerV3 model."""
        # TODO: Part 4.2 - Implement DreamerV3 forward pass for policy
        ## Encode observations, roll through RSSM, and plan with policy from current state
        if observations is None:
            raise ValueError("Dreamer planning requires observations (B,T,C,H,W)")
        if prev_actions is None:
            raise ValueError("Dreamer planning requires prev_actions (B,T,A)")

        if observations.dim() != 5:
            raise ValueError(f"observations must have shape (B,T,C,H,W); got {tuple(observations.shape)}")
        if prev_actions.dim() != 3:
            raise ValueError(f"prev_actions must have shape (B,T,A); got {tuple(prev_actions.shape)}")

        B, T, C, H, W = observations.shape

        # Build current RSSM state from the provided history.
        if prev_state is None:
            # print("No previous state provided; initializing new RSSM state.")
            state = self.world_model.get_initial_state(B, device=self.device)
        else:
            state = prev_state
            
        self.world_model.eval()
        with torch.no_grad():
            # print(f"Input observations shape: {observations.shape}")
            obs_flat = observations.view(B * T, C, H, W)
            embed_flat = self.world_model.encoder(obs_flat)
            embed = embed_flat.view(B, T, -1)

            for t in range(T):
                a_t = prev_actions[:, t, :]
                e_t = embed[:, t, :]
                step_out = self.world_model.rssm_step(state, a_t, embed=e_t)
                state = {
                    'h': step_out['h'],
                    'z': step_out['z'],
                    'z_probs': step_out.get('z_probs', None),
                }

        best_actions_seq, best_reward = self.plan(state)
        if return_full_sequence:
            planned = best_actions_seq
        else:
            planned = best_actions_seq[0]

        return {
            'actions': planned,
            'predicted_reward': best_reward,
            'final_state': state,
        }
    
    def _forward_simple(self, pose, return_full_sequence):
        """Forward pass for SimpleWorldModel."""
        # TODO: Part 1.3 - Implement SimpleWorldModel forward pass for policy
        ## Encode pose, roll through model, and plan with policy from current state
        #1. Encode pose to get initial state for planning
        initial_state = {'pose': pose}  # (B, 1, pose_dim)
        #2. Plan using CEM to get action sequence and predicted reward
        best_actions, best_reward = self.plan(initial_state)
        #3. Return the planned action sequence and predicted reward
        return {
            'actions': best_actions,  # (H, action_dim) or (action_dim,)
            'predicted_reward': best_reward,
            'final_state': initial_state  # Return the initial state used for planning (could also return the final state after rollout if desired)
        }


class PolicyPlanner(GRPBase):
    """
    Policy-based planner that uses a trained policy model to generate action sequences.
    Rolls out the policy over a horizon by predicting actions and states at each timestep.
    """
    def __init__(self, 
                 world_model,
                 policy_model,
                 action_dim,
                 cfg=None,
                 horizon=None):
        """
        Initialize Policy planner.
        
        Args:
            world_model: World model (DreamerV3 or SimpleWorldModel) used for predicting future states
            policy_model: Trained policy model that predicts actions given states
            action_dim: Dimensionality of the action space
            cfg: Configuration object
            horizon: Planning horizon (number of timesteps to plan ahead)
        """
        # TODO: Part 2.2 - Initialize Policy planner
        ## Set up world model, policy model, optimizer, and scheduler
        super(PolicyPlanner, self).__init__(cfg)
        self.world_model = world_model
        self.policy_model = policy_model
        self.action_dim = action_dim
        self.type = 'policy'
        self.cfg = cfg
        self.device = self.cfg.device
        self.to(self.device)

        self.horizon = horizon if horizon is not None else getattr(cfg.planner, 'horizon', 12) # Planning horizon
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=getattr(cfg.planner, 'learning_rate', 0.001))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)  # Example scheduler, can be configured as needed
        self.loss_fn = torch.nn.MSELoss()  # Example loss function for behavior cloning, can be changed based on the task
        self.K = getattr(cfg.planner, 'num_samples', 100)  # Number of action sequences to sample
        self.M = getattr(cfg.planner, 'num_elites', 20)  # Number of elite sequences to select
        self.L = getattr(cfg.planner, 'num_iterations', 3)  # Number of CEM iterations
        self.alpha = getattr(cfg.planner, 'temperature', 0.1)  # Temperature for updating distribution
        # self.training_epochs = getattr(cfg.training, 'num_epochs', 3)  # Small: outer loop already iterates epochs
        self.batch_size = getattr(cfg, 'batch_size', 64)  # Batch size for training the policy

    def load_world_model(self, path):
        """Load pretrained world model from checkpoint."""

        # Hydra may change the working directory (e.g., into `outputs/...`).
        # Resolve relative paths against the hw2 folder (where `checkpoints/` lives).
        p = os.path.expandvars(os.path.expanduser(str(path)))
        p_path = Path(p)
        if not p_path.is_absolute():
            hw2_root = Path(__file__).resolve().parent
            p_path = (hw2_root / p_path).resolve()

         # Safer loading: `weights_only=True` avoids unpickling arbitrary objects.
        self.world_model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def load_policy_model(self, path):
        """Load pretrained policy model from checkpoint."""
        # Hydra may change the working directory (e.g., into `outputs/...`).
        # Resolve relative paths against the hw2 folder (where `checkpoints/` lives).
        p = os.path.expandvars(os.path.expanduser(str(path)))
        p_path = Path(p)
        if not p_path.is_absolute():
            hw2_root = Path(__file__).resolve().parent
            p_path = (hw2_root / p_path).resolve()

        # Safer loading: `weights_only=True` avoids unpickling arbitrary objects.
        # This is recommended by PyTorch and will become the default in a future release.
        state_dict = torch.load(str(p_path), map_location=self.device, weights_only=True)
        self.policy_model.load_state_dict(state_dict)

    def update(self, states, actions):
        """
        Docstring for update
        Update the policy model using collected states and actions.
        
        :param self: Description
        :param states: Description
        :param actions: Description
        """
        # TODO: Part 2.2 - Implement policy training
        ## Train the policy using behavior cloning on collected state-action pairs
        # Check if any poses or actions contain NaNs and print a warning if so (debugging step)
        self.policy_model.train()
        B, T, action_dim = actions.shape
        # check whether self.action_dim == action_dim
        if self.action_dim != action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {action_dim}")
        else:
            actions = actions.view(B * T, action_dim)  # Flatten actions to (B*T, action_dim) for loss computation
        if self.cfg.model_type == 'simple':
            # Flatten sequence dimension so the policy sees individual (pose -> action) pairs:
            # (B, T, D) -> (B*T, D).  This matches eval-time where a single step is passed in.
            policy_input = states.view(B * T, -1)  # (B*T, pose_dim)
        elif self.cfg.model_type == 'dreamer':
            # --- NEW Part 4.1: Dreamer Image Logic ---
            B, T, C, H, W = states.shape
            obs_flat = states.view(B * T, C, H, W)  # Flatten time dimension

            # Use the World Model to encode the images into latent states for the policy.
            with torch.no_grad():
                embed_flat = self.world_model.encoder(obs_flat)  # (B*T, embed_dim)
                embed = embed_flat.view(B, T, -1)  # (B, T, hidden_dim)
                state = self.world_model.get_initial_state(B, device=self.device)  # Initialize RSSM state
                previous_actions = torch.zeros(B, self.action_dim, device=self.device)  # Start with zero actions

                hz_list = []
                for t in range(T):
                    e_t = embed[:, t, :]
                    step_out = self.world_model.rssm_step(state, previous_actions, embed=e_t)
                    state = {
                        'h': step_out['h'],
                        'z': step_out['z'],
                        'z_probs': step_out.get('z_probs', None),
                    }
                    hz_list.append(torch.cat([state['h'], state['z']], dim=-1))  # (B, deter_dim + stoch_dim*discrete_dim)
            policy_input = torch.stack(hz_list, dim=1).view(B * T, -1)  # (B*T, deter_dim + stoch_dim*discrete_dim)
        # Policy outputs (B, action_dim*2): first half = mean, second half = log_std
        policy_output = self.policy_model(policy_input)  # (B*T, action_dim*2)
        mean, log_std = torch.chunk(policy_output, 2, dim=-1)   # (B*T, action_dim)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        loss = -dist.log_prob(actions).mean()
        self.optimizer.zero_grad()
        # Use mean for deterministic behaviour cloning loss
        loss.backward()
        # Clip gradients to prevent large updates from destabilising training
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()  # Return average loss across epochs for monitoring

    
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences by rolling out the policy model over the horizon.
        
        Args:
            initial_state: Dictionary containing initial state 
                          - For DreamerV3: {'h', 'z', 'z_probs'}
                          - For SimpleWorldModel: {'pose'}
            return_best_sequence: If True, returns the planned sequence (unused here for consistency)
            
        Returns:
            actions: Tensor of shape (horizon, action_dim) with the planned action sequence
            total_reward: Float value of the sum of predicted rewards
        """
        # TODO: Part 2.2 - Implement policy rollout planning
        ## Roll out the policy over the horizon, predicting actions and accumulating rewards
        # Get the current conditioning feature for the policy.
        # - Simple: uses encoded pose.
        # - Dreamer: uses RSSM feature [h, z].
        mu = torch.zeros((self.horizon, self.action_dim), device=self.device)
        std = torch.zeros((self.horizon, self.action_dim), device=self.device)

        if self.cfg.model_type == 'simple':
            pose = initial_state['pose']
            if pose.dim() == 1: pose = pose.unsqueeze(0)
            current_state = pose  # (B, pose_dim)
            
            for t in range(self.horizon):
                with torch.no_grad():
                    policy_out = self.policy_model(current_state)
                    step_mean, step_log_std = torch.chunk(policy_out, 2, dim=-1)
                    
                    mu[t] = step_mean.squeeze(0)
                    std[t] = torch.exp(step_log_std).clamp(min=1e-3, max=1.0).squeeze(0)
                    current_state, _ = self.world_model.forward(current_state, step_mean)
        elif self.cfg.model_type == 'dreamer':
            current_state = initial_state
            for t in range(self.horizon):
                with torch.no_grad():
                    policy_out = self.policy_model(torch.cat([current_state['h'], current_state['z']], dim=-1))
                    step_mean, step_log_std = torch.chunk(policy_out, 2, dim=-1)
                    
                    mu[t] = step_mean.squeeze(0)
                    std[t] = torch.exp(step_log_std).clamp(min=1e-3, max=1.0).squeeze(0)
                    # RSSM imagination step: no embed -> sample from prior.
                    step_out = self.world_model.rssm_step(current_state, step_mean, embed=None)
                    current_state = {
                        'h': step_out['h'],
                        'z': step_out['z'],
                        'z_probs': step_out.get('z_probs', None),
                    }
        # ---- CEM Refinement Loop ----
        for iteration in range(self.L):
            # Sample action sequences from the current distribution
            action_sequences = mu + std * torch.randn(self.K, self.horizon, self.action_dim, device=self.device)  # (K, H, action_dim)

            # Evaluate the sampled action sequences using the world model
            rewards = self._evaluate_sequences(initial_state, action_sequences)  # (K,)
            
            # Select the top M elite sequences based on rewards
            elite_indices = torch.topk(rewards, self.M).indices  # (M,)
            elite_action_sequences = action_sequences[elite_indices]  # (M, H, action_dim)
            
            # Update the mean and std of the action distribution using the elites
            new_mu = elite_action_sequences.mean(dim=0)  # (H, action_dim)
            # We could use unbiased=False to avoid NaN when M==1 (avoids division by N-1=0)
            new_std = elite_action_sequences.std(dim=0, unbiased=True).clamp(min=1e-3)  # (H, action_dim)
            
            # Smoothly update the distribution parameters with temperature alpha
            mu = self.alpha * new_mu + (1 - self.alpha) * mu
            std = self.alpha * new_std + (1 - self.alpha) * std
            # Track best action sequence and reward for return
            best_actions = elite_action_sequences[0] if return_best_sequence else mu  # (H, action_dim) - best sequence from elites
            best_reward = rewards[elite_indices[0]]  # Best reward from elites

        return best_actions, best_reward

    def _evaluate_sequences(self, initial_state, action_sequences):
        """
        Evaluate a batch of action sequences by rolling them out in the world model.
        
        Args:
            initial_state: Dictionary with initial state (RSSM state for DreamerV3 or pose for SimpleWorldModel)
            action_sequences: Tensor of shape (num_samples, horizon, action_dim)
            
        Returns:
            rewards: Tensor of shape (num_samples,) with sum of predicted rewards
        """
        # TODO: Part 1.3 - Route to appropriate evaluation method
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if self.cfg.model_type == 'dreamer':
            return self._evaluate_sequences_dreamer(initial_state, action_sequences)
        else:
            return self._evaluate_sequences_simple(initial_state, action_sequences)

    def _evaluate_sequences_dreamer(self, initial_state, action_sequences):
        """
        Evaluate sequences using DreamerV3 RSSM-based rollout.
        """
        # TODO: Part 3.3 - Implement CEM planning with DreamerV3
        ## Roll out action sequences in the DreamerV3 world model and compute total rewards
        if not isinstance(initial_state, dict) or 'h' not in initial_state or 'z' not in initial_state:
            raise ValueError(
                "DreamerV3 planning requires initial_state with keys {'h','z'} (and optionally 'z_probs')."
            )

        # We evaluate K candidate action sequences in parallel.
        # Expand the RSSM state to match the action batch dimension (K).
        K = action_sequences.shape[0]
        state = {
            'h': initial_state['h'].expand(K, -1).contiguous(),
            'z': initial_state['z'].expand(K, -1).contiguous(),
        }
        if 'z_probs' in initial_state and initial_state['z_probs'] is not None:
            # z_probs may be (B, Z, C). Expand along batch.
            state['z_probs'] = initial_state['z_probs'].expand(K, -1, -1).contiguous()
        else:
            state['z_probs'] = None

        self.world_model.eval()
        full_feat = []
        with torch.no_grad():
            for t in range(self.horizon):
                a_t = action_sequences[:, t, :]
                # RSSM imagination step: no embed -> sample from prior.
                step_out = self.world_model.rssm_step(state, a_t, embed=None)
                h = step_out['h']
                z = step_out['z']

                feat = torch.cat([h, z], dim=-1)
                full_feat.append(feat)
                state = {
                    'h': h,
                    'z': z,
                    'z_probs': step_out.get('z_probs', None),
                }
            full_feat = torch.stack(full_feat, dim=1)  # (K, H, feat_dim)
            # RewardPredictor outputs a scalar per feature vector.
            # DreamerV3 trains reward in symlog-space (see DreamerV3.compute_loss),
            # so here we sum predicted symlog rewards for ranking action sequences.
            r_symlog = self.world_model.reward_head(full_feat.view(K * self.horizon, -1)).view(K, self.horizon)
            total_rewards = r_symlog.sum(dim=-1)  # (K,)
        return total_rewards

    def _evaluate_sequences_simple(self, initial_state, action_sequences):
        """
        Evaluate sequences using SimpleWorldModel pose-based rollout.
        """
        # TODO: Part 1.3 - Implement CEM planning with SimpleWorldModel
        ## Roll out action sequences using SimpleWorldModel and compute total rewards
        current_state = initial_state['pose']  # (B, pose_dim)
        # 1. Broadcast current state to match the batch size of action sequences
        current_state = current_state.expand(action_sequences.shape[0], -1, -1).contiguous()  # (K, pose_dim)
        # 2. Initialize total rewards tensor on the same device as current_state
        total_rewards = torch.zeros(action_sequences.shape[0], device=self.cfg.device)  # (K,)
        # 3. Roll out each action sequence in the world model and accumulate rewards
        for step in range(self.horizon):  # Iterate over each step in the horizon
            action = action_sequences[:, step, :].unsqueeze(1).to(self.cfg.device)  # (K, 1, action_dim)
            self.world_model.eval()  # Set world model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for evaluation
                next_pose_pred, reward_pred = self.world_model.predict_next_pose(current_state, action)  # (K, 1, pose_dim), (K, 1)
            total_rewards += reward_pred.squeeze(-1)  # Accumulate rewards (K,)
            current_state = next_pose_pred  # Because next_pose_pred is in the same normalized space expected by the world model for the next step
        return total_rewards  # (K,)
    
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that works with both DreamerV3 and SimpleWorldModel.
        This wrapper obtains the current state and plans actions using the policy.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (for DreamerV3)
            prev_actions: Previous actions (optional, for state initialization)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Planned action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': Expected cumulative reward
                - 'final_state': Final state after processing inputs
        """
        # TODO: Part 2.2 - Route forward pass to appropriate model
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if self.cfg.model_type == 'dreamer':
            return self._forward_dreamer(observations, prev_actions, prev_state, return_full_sequence)
        elif self.cfg.model_type == 'simple' and self.cfg.planner.type == 'policy':
            self.policy_model.eval()  # Set policy model to evaluation mode for planning
            with torch.no_grad():  # Ensure no gradients are computed during planning
                # print(f"Shape of input pose: {pose.shape}")  # Debugging print to check input shape
                actions = self.policy_model(pose.squeeze(1))  # Get action distribution from policy model
                # print(f"Shape of policy output: {actions.shape}")  # Debugging print to check output shape
            mu = actions[:, :self.action_dim]  # Extract mean (B, action_dim)
            std = actions[:, self.action_dim:]  # Extract std (B, action_dim)
            # print(f"Action mean shape: {mu.shape}, Action std shape: {std.shape}")  # Debugging print to check shapes
            sampled_actions = mu  # For deterministic policy; for stochastic, you would sample from the distribution using mu and std
            return {
                'actions': sampled_actions,  # (1, action_dim)
                'predicted_reward': None,  # Predicted reward is not computed here since we're directly using the policy's output; could be computed by rolling out in the world model if desired
                'final_state': None  # Final state is not computed here
            }
        elif self.cfg.model_type == 'simple' and self.cfg.planner.type == 'policy_guided_cem':
            # print("Policy-guided CEM planning with SimpleWorldModel is implemented.")
            return self._forward_simple(pose, return_full_sequence)
    
    def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
        """Forward pass for DreamerV3 model."""
        # TODO: Part 4.2 - Implement DreamerV3 forward pass for policy
        ## Encode observations, roll through RSSM, and plan with policy from current state
        if observations is None:
            raise ValueError("Dreamer planning requires observations (B,T,C,H,W)")
        if prev_actions is None:
            raise ValueError("Dreamer planning requires prev_actions (B,T,A)")

        if observations.dim() != 5:
            raise ValueError(f"observations must have shape (B,T,C,H,W); got {tuple(observations.shape)}")
        if prev_actions.dim() != 3:
            raise ValueError(f"prev_actions must have shape (B,T,A); got {tuple(prev_actions.shape)}")

        B, T, C, H, W = observations.shape

        # Build current RSSM state from the provided history.
        if prev_state is None:
            # print("No previous state provided; initializing new RSSM state.")
            state = self.world_model.get_initial_state(B, device=self.device)
        else:
            state = prev_state
            
        self.world_model.eval()
        with torch.no_grad():
            obs_flat = observations.view(B * T, C, H, W)
            embed_flat = self.world_model.encoder(obs_flat)
            embed = embed_flat.view(B, T, -1)

            for t in range(T):
                a_t = prev_actions[:, t, :]
                e_t = embed[:, t, :]
                step_out = self.world_model.rssm_step(state, a_t, embed=e_t)
                state = {
                    'h': step_out['h'],
                    'z': step_out['z'],
                    'z_probs': step_out.get('z_probs', None),
                }

        best_actions_seq, best_reward = self.plan(state)
        if return_full_sequence:
            planned = best_actions_seq
        else:
            planned = best_actions_seq[0]

        return {
            'actions': planned,
            'predicted_reward': best_reward,
            'final_state': state,
        }
    
    def _forward_simple(self, pose, return_full_sequence):
        """Forward pass for SimpleWorldModel."""
        # TODO: Part 1.3 - Implement SimpleWorldModel forward pass for policy
        ## Encode pose, roll through model, and plan with policy from current state
        #1. Encode pose to get initial state for planning
        initial_state = {'pose': pose}  # (B, 1, pose_dim)
        #2. Plan using CEM to get action sequence and predicted reward
        best_actions_seq, best_reward = self.plan(initial_state)
        #3. Return the planned action sequence and predicted reward
        return {
            'actions': best_actions_seq,  # (H, action_dim) or (action_dim,)
            'predicted_reward': best_reward,
            'final_state': initial_state  # Return the initial state used for planning (could also return the final state after rollout if desired)
        }


class RandomPlanner(GRPBase):
    """
    Random action planner that generates random actions uniformly distributed between -1 and 1.
    Useful as a baseline for comparing planning algorithms.
    """
    def __init__(self, 
                 action_dim,
                 cfg):
        """
        Initialize Random planner.
        
        Args:
            world_model: World model (optional, not used but kept for API compatibility)
            action_dim: Dimensionality of the action space (default: 7)
            cfg: Configuration object (optional)
            horizon: Planning horizon (number of timesteps to plan ahead)
        """
        super(RandomPlanner, self).__init__(cfg)
        
        self.action_dim = action_dim
            
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that generates random actions.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (optional)
            prev_actions: Previous actions (optional)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Random action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': 0.0 (no prediction for random actions)
                - 'final_state': None or dummy state
        """
        ## compute random actions
        actions = torch.rand((1, self.action_dim), device=pose.device) * 2 - 1  # (1, action_dim) in range [-1, 1]
        
        return {
            'actions': actions,
            'predicted_reward': 0.0,
            'final_state': prev_state if prev_state is not None else None
        }