try:
    # When imported as a package: `from hw2.planning import CEMPlanner`
    from .dreamerV3 import GRPBase
except ImportError:
    # When executed with cwd=hw2/
    from dreamerV3 import GRPBase
import numpy as np
import torch


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
        self.H = getattr(cfg.planner, 'horizon', 12)  # Planning horizon
        self.K = getattr(cfg.planner, 'num_samples', 100)  # Number of action sequences to sample
        self.M = getattr(cfg.planner, 'num_elites', 20)  # Number of elite sequences to select
        self.L = getattr(cfg.planner, 'num_iterations', 3)  # Number of CEM iterations
        self.alpha = getattr(cfg.planner, 'temperature', 0.1)  # Temperature for updating distribution

        self.action_mean = torch.zeros(self.H, self.action_dim, device=self.cfg.device)  # Initial mean of action distribution
        self.action_std = torch.ones(self.H, self.action_dim, device=self.cfg.device)  # Initial std of action distribution
        
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
        action_mean = torch.zeros(self.H, self.action_dim, device=self.device)
        action_std = torch.ones(self.H, self.action_dim, device=self.device)
        for iteration in range(self.L):
            # Sample action sequences from the current distribution
            action_sequences = action_mean + action_std * torch.randn(self.K, self.H, self.action_dim, device=self.device)  # (K, H, action_dim)
            
            # Evaluate the sampled action sequences using the world model
            rewards = self._evaluate_sequences(initial_state, action_sequences)  # (K,)
            
            # Select the top M elite sequences based on rewards
            elite_indices = torch.topk(rewards, self.M).indices  # (M,)
            elite_action_sequences = action_sequences[elite_indices]  # (M, H, action_dim)
            
            # Update the mean and std of the action distribution using the elites
            new_action_mean = elite_action_sequences.mean(dim=0)  # (H, action_dim)
            # We could use unbiased=False to avoid NaN when M==1 (avoids division by N-1=0)
            new_action_std = elite_action_sequences.std(dim=0, unbiased=True).clamp(min=1e-6)  # (H, action_dim)
            
            # Smoothly update the distribution parameters with temperature alpha
            action_mean = self.alpha * new_action_mean + (1 - self.alpha) * action_mean
            action_std = self.alpha * new_action_std + (1 - self.alpha) * action_std
            # Track best action sequence and reward for return
            best_actions = elite_action_sequences[0] if return_best_sequence else action_mean  # (H, action_dim) - best sequence from elites
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

        # action_sequences: (K, H, A)
        K, H, A = action_sequences.shape
        if H != self.H:
            raise ValueError(f"Expected action_sequences horizon {self.H}, got {H}")
        if A != self.action_dim:
            raise ValueError(f"Expected action_dim {self.action_dim}, got {A}")

        wm = self.world_model
        device = action_sequences.device

        # Expand initial RSSM state across K candidate sequences.
        h0 = initial_state['h']
        z0 = initial_state['z']
        if h0.dim() == 2 and h0.shape[0] == 1:
            h0 = h0.expand(K, -1)
        if z0.dim() == 2 and z0.shape[0] == 1:
            z0 = z0.expand(K, -1)
        if h0.shape[0] != K:
            raise ValueError(f"initial_state['h'] batch must be 1 or K={K}, got {h0.shape[0]}")
        if z0.shape[0] != K:
            raise ValueError(f"initial_state['z'] batch must be 1 or K={K}, got {z0.shape[0]}")

        state = {
            'h': h0.to(device),
            'z': z0.to(device),
            # Not strictly needed for imagination; kept for completeness.
            'z_probs': initial_state.get('z_probs', None),
        }

        total_rewards = torch.zeros(K, device=device)
        wm.eval()
        with torch.no_grad():
            for t in range(self.H):
                a_t = action_sequences[:, t, :]
                # RSSM imagination step: no embed -> sample from prior.
                step_out = wm.rssm_step(state, a_t, embed=None)
                h = step_out['h']
                z = step_out['z']

                feat = torch.cat([h, z], dim=-1)
                r_t = wm.reward_head(feat).mean  # (K,)
                total_rewards += r_t

                state = {
                    'h': h,
                    'z': z,
                    'z_probs': step_out.get('z_probs', None),
                }

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
        for step in range(self.H):  # Iterate over each step in the horizon
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
        device = observations.device

        # Build current RSSM state from the provided history.
        if prev_state is None:
            state = self.world_model.get_initial_state(B, device=device)
        else:
            state = prev_state

        wm = self.world_model
        wm.eval()
        with torch.no_grad():
            obs_flat = observations.reshape(B * T, C, H, W)
            embed_flat = wm.encoder(obs_flat)
            embed = embed_flat.view(B, T, -1)

            for t in range(T):
                a_t = prev_actions[:, t, :]
                e_t = embed[:, t, :]
                step_out = wm.rssm_step(state, a_t, embed=e_t)
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

        # [Imagine method remains mostly the same, ensuring valid input shapes for heads]
    def preprocess_state(self, image):
        """Preprocess observation image"""
        # TODO: Preprocess image for input
        ## Resize, normalize, and convert to channel-first format
        # Reuse GRPBase utilities: resize_image() + normalize_state().
        # GRPBase assumes channel-last (H,W,C), after which we convert to
        # channel-first (C,H,W) for the Dreamer encoder.

        # Accept torch tensors or numpy/PIL-style arrays.
        if torch.is_tensor(image):
            img = image.detach().cpu().numpy()
        else:
            img = np.array(image)

        # If a batch/time dimension is present, keep only the last frame.
        # Expected single frame shape is (H,W,C) or (C,H,W).
        if img.ndim == 4:
            img = img[-1]

        # Convert channel-first to channel-last if needed.
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # Use inherited helpers (no recursion!).
        img = self.resize_image(img)
        img = self.normalize_state(img)

        # (H,W,C) -> (C,H,W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return img


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

        self.H = horizon if horizon is not None else getattr(cfg.planner, 'horizon', 12) # Planning horizon
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=getattr(cfg.planner, 'learning_rate', 0.001))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)  # Example scheduler, can be configured as needed
        self.loss_fn = torch.nn.MSELoss()  # Example loss function for behavior cloning, can be changed based on the task
        self.K = getattr(cfg.planner, 'num_samples', 100)  # Number of action sequences to sample
        self.M = getattr(cfg.planner, 'num_elites', 20)  # Number of elite sequences to select
        self.L = getattr(cfg.planner, 'num_iterations', 3)  # Number of CEM iterations
        self.alpha = getattr(cfg.planner, 'temperature', 0.1)  # Temperature for updating distribution
        # self.training_epochs = getattr(cfg.training, 'num_epochs', 3)  # Small: outer loop already iterates epochs
        self.batch_size = getattr(cfg, 'batch_size', 64)  # Batch size for training the policy

    def load_policy_model(self, path):
        """Load pretrained policy model from checkpoint."""
        import os
        from pathlib import Path

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
        if torch.isnan(states).any():
            print("Warning: NaNs detected in encoded states. Check data preprocessing and encoding.")
            print(f"NaN count in states: {torch.isnan(states).sum().item()} out of {states.numel()}")
        if torch.isnan(actions).any():
            print("Warning: NaNs detected in encoded actions. Check data preprocessing and encoding.")
            print(f"NaN count in actions: {torch.isnan(actions).sum().item()} out of {actions.numel()}")
        # Flatten sequence dimension so the policy sees individual (pose -> action) pairs:
        # (B, T, D) -> (B*T, D).  This matches eval-time where a single step is passed in.
        if states.dim() == 3:
            B, T, D = states.shape
            states  = states.reshape(B * T, D)                    # (B*T, pose_dim)
            actions = actions.reshape(B * T, self.action_dim)     # (B*T, action_dim)

        # Policy outputs (B, action_dim*2): first half = mean, second half = log_std
        predicted_actions_distribution = self.policy_model(states)  # (B, action_dim*2)
        predicted_actions_mean = predicted_actions_distribution[:, :self.action_dim]   # (B, action_dim)
        predicted_actions_std = predicted_actions_distribution[:, self.action_dim:]  # (B, action_dim)
        self.optimizer.zero_grad()
        # Use mean for deterministic behaviour cloning loss
        loss = self.loss_fn(predicted_actions_mean, actions)  # MSE loss between predicted mean actions and true actions
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
        if self.cfg.model_type == 'dreamer':
            if not isinstance(initial_state, dict) or 'h' not in initial_state or 'z' not in initial_state:
                raise ValueError("Dreamer PolicyPlanner.plan requires RSSM state dict with keys {'h','z'}")
            feat = torch.cat([initial_state['h'], initial_state['z']], dim=-1)
            current_state = feat
        else:
            current_state = initial_state['pose']
            if current_state.dim() == 3:
                current_state = current_state.squeeze(1)

        self.policy_model.eval()  # Set policy to evaluation mode for planning
        with torch.no_grad():
            action_distribution = self.policy_model(current_state)  # (B, action_dim*2)

        action_mean = action_distribution[:, :self.action_dim].unsqueeze(1)  # (B,1,A)
        action_std = action_distribution[:, self.action_dim:].unsqueeze(1)   # (B,1,A)
        for iteration in range(self.L):
            # Sample action sequences from the current distribution
            action_sequences = action_mean + action_std * torch.randn(self.K, self.H, self.action_dim, device=self.device)  # (K, H, action_dim)
            
            # Evaluate the sampled action sequences using the world model
            rewards = self._evaluate_sequences(initial_state, action_sequences)  # (K,)
            
            # Select the top M elite sequences based on rewards
            elite_indices = torch.topk(rewards, self.M).indices  # (M,)
            elite_action_sequences = action_sequences[elite_indices]  # (M, H, action_dim)
            
            # Update the mean and std of the action distribution using the elites
            new_action_mean = elite_action_sequences.mean(dim=0)  # (H, action_dim)
            # We could use unbiased=False to avoid NaN when M==1 (avoids division by N-1=0)
            new_action_std = elite_action_sequences.std(dim=0, unbiased=True).clamp(min=1e-6)  # (H, action_dim)
            
            # Smoothly update the distribution parameters with temperature alpha
            action_mean = self.alpha * new_action_mean + (1 - self.alpha) * action_mean
            action_std = self.alpha * new_action_std + (1 - self.alpha) * action_std
            # Track best action sequence and reward for return
            best_actions = elite_action_sequences[0] if return_best_sequence else action_mean  # (H, action_dim) - best sequence from elites
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

        # action_sequences: (K, H, A)
        K, H, A = action_sequences.shape
        if H != self.H:
            raise ValueError(f"Expected action_sequences horizon {self.H}, got {H}")
        if A != self.action_dim:
            raise ValueError(f"Expected action_dim {self.action_dim}, got {A}")

        wm = self.world_model
        device = action_sequences.device

        # Expand initial RSSM state across K candidate sequences.
        h0 = initial_state['h']
        z0 = initial_state['z']
        if h0.dim() == 2 and h0.shape[0] == 1:
            h0 = h0.expand(K, -1)
        if z0.dim() == 2 and z0.shape[0] == 1:
            z0 = z0.expand(K, -1)
        if h0.shape[0] != K:
            raise ValueError(f"initial_state['h'] batch must be 1 or K={K}, got {h0.shape[0]}")
        if z0.shape[0] != K:
            raise ValueError(f"initial_state['z'] batch must be 1 or K={K}, got {z0.shape[0]}")

        state = {
            'h': h0.to(device),
            'z': z0.to(device),
            'z_probs': initial_state.get('z_probs', None),
        }

        total_rewards = torch.zeros(K, device=device)
        wm.eval()
        with torch.no_grad():
            for t in range(self.H):
                a_t = action_sequences[:, t, :]
                step_out = wm.rssm_step(state, a_t, embed=None)  # imagination step
                h = step_out['h']
                z = step_out['z']
                feat = torch.cat([h, z], dim=-1)
                r_t = wm.reward_head(feat).mean
                total_rewards += r_t
                state = {'h': h, 'z': z, 'z_probs': step_out.get('z_probs', None)}

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
        for step in range(self.H):  # Iterate over each step in the horizon
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
                actions = self.policy_model(self.world_model.encode_pose(pose.squeeze(1)))  # Get action distribution from policy model
                # print(f"Shape of policy output: {actions.shape}")  # Debugging print to check output shape
            action_mean = actions[:, :self.action_dim]  # Extract mean (B, action_dim)
            action_std = actions[:, self.action_dim:]  # Extract std (B, action_dim)
            # print(f"Action mean shape: {action_mean.shape}, Action std shape: {action_std.shape}")  # Debugging print to check shapes
            sampled_actions = action_mean  # For deterministic policy; for stochastic, you would sample from the distribution using action_mean and action_std
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
            raise ValueError("Dreamer policy planning requires observations (B,T,C,H,W)")
        if prev_actions is None:
            raise ValueError("Dreamer policy planning requires prev_actions (B,T,A)")
        if observations.dim() != 5:
            raise ValueError(f"observations must have shape (B,T,C,H,W); got {tuple(observations.shape)}")
        if prev_actions.dim() != 3:
            raise ValueError(f"prev_actions must have shape (B,T,A); got {tuple(prev_actions.shape)}")

        B, T, C, H, W = observations.shape
        device = observations.device

        wm = self.world_model

        # Build current RSSM state from history using posterior updates.
        if prev_state is None:
            state = wm.get_initial_state(B, device=device)
        else:
            state = prev_state

        wm.eval()
        with torch.no_grad():
            obs_flat = observations.reshape(B * T, C, H, W)
            embed_flat = wm.encoder(obs_flat)
            embed = embed_flat.view(B, T, -1)

            for t in range(T):
                a_t = prev_actions[:, t, :]
                e_t = embed[:, t, :]
                step_out = wm.rssm_step(state, a_t, embed=e_t)
                state = {
                    'h': step_out['h'],
                    'z': step_out['z'],
                    'z_probs': step_out.get('z_probs', None),
                }

        # Now plan from the latent state using the policy-guided CEM in `plan()`.
        best_actions_seq, best_reward = self.plan(state)

        planned = best_actions_seq if return_full_sequence else best_actions_seq[0]
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