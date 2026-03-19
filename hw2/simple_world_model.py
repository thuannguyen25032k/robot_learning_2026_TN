import torch
import torch.nn as nn
import numpy as np
try:
    from .dreamerV3 import GRPBase
except ImportError:
    from dreamerV3 import GRPBase

class ResMLPBlock(nn.Module):
    """MLP-ResNet block: x -> x + f(LN(x))."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hid = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hid, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))

class SimpleWorldModel(GRPBase):
    """
    Simple world model that predicts the next pose and reward given current pose and action.
    
    Architecture:
    - Takes current pose (7-d) + normalized action
    - Simple MLP to predict next pose (7-d) and reward (scalar)
    """
    
    def __init__(self, 
                 action_dim=7,
                 pose_dim=7,
                 hidden_dim=256,
                 cfg=None):
        # TODO: Part 1.1 - Initialize SimpleWorldModel architecture
        ## Define the feature network and output heads (pose and reward)
        super(SimpleWorldModel, self).__init__(cfg)
        input_dim = pose_dim + action_dim
        n_blocks = 2  # preferred 6-8 blocks; set to 4 by default

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(*[ResMLPBlock(hidden_dim, expansion=4, dropout=0) for _ in range(n_blocks)])
        self.output_norm = nn.LayerNorm(hidden_dim)
    # Output heads for next pose and reward
        self.pose_head = nn.Linear(hidden_dim, pose_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.type = 'simple'
        self.device = self._cfg.device
        self.to(self.device)
    
    def forward(self, pose, action):
        """
        Forward pass to predict next pose and reward.
        
        Args:
            pose: Pose tensor of shape (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor of shape (B, action_dim) or (B, T, action_dim), normalized
        
        Returns:
            next_pose_pred: Predicted normalized pose (B, pose_dim) or (B, T, pose_dim)
            reward_pred: Predicted reward (B, 1) or (B, T, 1)
        """
        # TODO: Part 1.1 - Implement forward pass
        ## Concatenate pose and action, pass through feature network and output heads
        # Handle both (B, D) and (B, T, D) shapes
        if pose.dim() == 2:  # (B, D)
            x = torch.cat([pose, action], dim=-1)  # (B, pose_dim + action_dim)
            features = self.output_norm(self.res_blocks(self.input_proj(x)))  # (B, hidden_dim)
            next_pose_pred = self.pose_head(features)  # (B, pose_dim)
            reward_pred = self.reward_head(features)  # (B, 1)
        else:  # (B, T, D)
            B, T, _ = pose.shape
            # (B, T, pose_dim + action_dim)
            x = torch.cat([pose, action], dim=-1)
            x = x.view(B * T, -1)  # (B*T, pose_dim + action_dim)
            features = self.output_norm(self.res_blocks(self.input_proj(x)))  # (B*T, hidden_dim)
            next_pose_pred = self.pose_head(features).view(
                B, T, -1)  # (B, T, pose_dim)
            reward_pred = self.reward_head(features).view(
                B, T)  # (B, T, 1) -> (B, T)
        return next_pose_pred, reward_pred
    
    def predict_next_pose(self, pose, action):
        """
        Predict the next pose and reward given current pose and action.

        Args:
            pose: Current pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor (B, action_dim) or (B, T, action_dim), normalized
        Returns:
            next_pose_pred: Predicted next pose in normalized space (B, pose_dim) or (B, T, pose_dim)
            reward_pred: Predicted reward (B, 1) or (B, T, 1)
        """
        # TODO: Part 1.1 - Implement prediction method
        ## Encode action, call forward, and decode pose to original space
        # I assume both pose and action are already normalized
        next_pose_pred, reward_pred = self.forward(pose, action) # (B, pose_dim), (B, 1) or (B, T, pose_dim), (B, T)
        # I also assume that the next_pose_pred will be decoded to original space later.
        return next_pose_pred, reward_pred
    
    def compute_loss(self, pred_pose, pred_reward, target_pose, target_reward=None):
        """
        Compute MSE loss between predicted and target pose and reward.
        
        Args:
            pose: Current pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor (B, action_dim) or (B, T, action_dim), normalized
            target_pose: Target pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            target_reward: Target reward tensor (B, 1) or (B, T, 1), optional
        
        Returns:
            loss: Total MSE loss (pose + reward if target_reward is provided)
        """
        # TODO: Part 1.2 - Implement SimpleWorldModel loss computation
        ## Compute MSE loss for pose and reward predictions
        pose_loss = nn.MSELoss()(pred_pose, target_pose)
        if target_reward is not None:
            reward_loss = nn.MSELoss()(pred_reward, target_reward)
            total_loss = pose_loss + reward_loss
        else:
            total_loss = pose_loss
        # Add optional weighting if desired (e.g., total_loss = pose_loss + reward_loss * reward_weight)
        # Add any regularization losses if needed (e.g., L2 on weights)
        loss_dict = {
            'pose_loss': pose_loss,
            'reward_loss': reward_loss if target_reward is not None else 0.0,
            'total_loss': total_loss,
        }
        return loss_dict
