import sys
import os
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"
# Ensure the vendored LIBERO package is importable even if it hasn't been pip-installed.
# Hydra may change the working directory, so we resolve relative to this file.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIBERO_ROOT = _REPO_ROOT / "LIBERO"
if _LIBERO_ROOT.exists():
    sys.path.insert(0, str(_LIBERO_ROOT))

import dill
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from torchvision import transforms
import h5py
import numpy as np

# Support both `python hw2/dreamer_model_trainer.py` (cwd=hw2) and
# `python -m hw2.dreamer_model_trainer` / importing as a package.
try:
    from .dreamerV3 import DreamerV3
    from .simple_world_model import SimpleWorldModel
    from .planning import CEMPlanner, PolicyPlanner, RandomPlanner
except ImportError:
    from dreamerV3 import DreamerV3
    from simple_world_model import SimpleWorldModel
    from planning import CEMPlanner, PolicyPlanner, RandomPlanner
import random
from collections import deque
from datasets import load_dataset
import datasets
from torch.nn.utils.rnn import pad_sequence



# Factory function to instantiate the correct model
def create_model(model_type, img_shape, action_dim, device, cfg):
    """
    Factory function to create a world model based on the specified type.

    Args:
        model_type: 'dreamer' or 'simple' 
        img_shape: Image shape [C, H, W]
        action_dim: Dimensionality of actions
        device: torch device
        cfg: Configuration object

    Returns:
        model: Instantiated model
    """
    if model_type.lower() == 'dreamer':
        model = DreamerV3(obs_shape=img_shape,
                          action_dim=action_dim, cfg=cfg).to(device)
    elif model_type.lower() == 'simple':
        model = SimpleWorldModel(
            action_dim=action_dim, pose_dim=7, hidden_dim=256, cfg=cfg).to(device)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose 'dreamer' or 'simple'.")

    return model

def batch_data(dataset, batch_size, cfg):
    """
    Utility function to batch data from the dataset with a fixed sequence length.
    Args:    
    dataset: Dataset object that returns (images, actions, rewards, dones, poses)
    batch_size: Number of sequences per batch
    sequence_length: Length of each sequence (T)

    Returns:
    A generator that yields batches of (images, actions, rewards, dones, poses) with shapes:
    - images: (B, T, C, H, W)
    - actions: (B, T, 7)
    - rewards: (B, T)
    - dones: (B, T)
    - poses: (B, T, 7)
    """
    # Collect sequences for the batch with fixed sequence length
    list_images, list_actions, list_rewards, list_dones, list_poses = [], [], [], [], []
    # padding short trajectories to max_seq_len with zeros
    for img, act, rew, don, pos in dataset:
        list_images += [img[i:i+cfg.policy.sequence_length] for i in range(0, len(img)-cfg.policy.sequence_length+1, cfg.policy.sequence_length)]
        list_actions += [act[i:i+cfg.policy.sequence_length] for i in range(0, len(act)-cfg.policy.sequence_length+1, cfg.policy.sequence_length)]
        list_rewards += [rew[i:i+cfg.policy.sequence_length] for i in range(0, len(rew)-cfg.policy.sequence_length+1, cfg.policy.sequence_length)]
        list_dones += [don[i:i+cfg.policy.sequence_length] for i in range(0, len(don)-cfg.policy.sequence_length+1, cfg.policy.sequence_length)]
        list_poses += [pos[i:i+cfg.policy.sequence_length] for i in range(0, len(pos)-cfg.policy.sequence_length+1, cfg.policy.sequence_length)]
    images = torch.stack(list_images)  # (B, T, H, W, C)
    actions = torch.stack(list_actions)  # (B, T, action_dim)
    rewards = torch.stack(list_rewards)  # (B, T)
    dones = torch.stack(list_dones)  # (B, T)
    poses = torch.stack(list_poses)  # (B, T, pose_dim)
    images = images.permute(0, 1, 4, 2, 3).to(cfg.device)  # (B, T, H, W, C) -> (B, T, C, H, W)
    actions = actions.float().to(cfg.device) # (B, T, action_dim)
    rewards = rewards.float().to(cfg.device) # (B, T)
    dones = dones.float().to(cfg.device) # (B, T)
    poses = poses.float().to(cfg.device) # (B, T, pose_dim)
    # for img, act, rew, don, pos in dataset:
    #     list_images.append(img)  # (T, H, W, C)
    #     list_actions.append(act)  # (T, action_dim)
    #     list_rewards.append(rew)  # (T,)
    #     list_dones.append(don)  # (T,)
    #     list_poses.append(pos)  # (T, pose_dim)
    # images = pad_sequence(list_images, batch_first=True, padding_value=0.0).permute(0, 1, 4, 2, 3).to(cfg.device)  # (B, T, H, W, C) -> (B, T, C, H, W)
    # actions = pad_sequence(list_actions, batch_first=True, padding_value=0.0).float().to(cfg.device)  # (B, T, action_dim)
    # rewards = pad_sequence(list_rewards, batch_first=True, padding_value=0.0).float().to(cfg.device)  # (B, T)
    # dones = pad_sequence(list_dones, batch_first=True, padding_value=0.0).float().to(cfg.device)  # (B, T)
    # poses = pad_sequence(list_poses, batch_first=True, padding_value=0.0).float().to(cfg.device)  # (B, T, pose_dim)
    print(f"[info] Batched data into tensors with shapes: images={images.shape}, actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}, poses={poses.shape}")
    out_dataset = torch.utils.data.TensorDataset(images, actions, rewards, dones, poses)
    print(f"[info] Created DataLoader with {len(out_dataset)} samples")
    return torch.utils.data.DataLoader(out_dataset, batch_size=batch_size, shuffle=True)

class ModelTrainingWrapper:
    """
    Wrapper to provide unified interface for training different world models.
    Handles differences in forward passes and loss computation between models.
    """

    def __init__(self, model, model_type, device):
        self.model = model
        self.model_type = model_type.lower()
        self.device = device

    def forward_pass(self, images, poses, actions):
        """
        Unified forward pass that works with both model types.

        Args:
            images: Image tensor (B, T, H, W, C) or None for simple model
            poses: Pose tensor (B, T, 7)
            actions: Action tensor (B, T, 7)

        Returns:
            output: Model output (format depends on model type)
        """
        if self.model_type == 'dreamer':
            # DreamerV3 returns a dict of rollout predictions.
            return self.model(images, actions)
        elif self.model_type == 'simple':
            # SimpleWorldModel expects normalized inputs
            pred_pose_seq, pred_reward_seq = self.model(poses, actions)
            return {
                'pred_poses': pred_pose_seq,
                'pred_rewards': pred_reward_seq
            }

    def compute_loss(self, model_out, normalized_images, rewards, dones, poses, actions):
        """
        Compute loss in a way that works for both model types.

        Args:
            output: Output from forward_pass
            normalized_images: Image tensor
            rewards: Reward tensor
            dones: Done tensor
            poses: Pose tensor (used for SimpleWorldModel)
            actions: Action tensor (used for SimpleWorldModel)
            pred_coeff, dyn_coeff, rep_coeff: Loss coefficients (used for DreamerV3)

        Returns:
            losses: Dictionary with loss information
        """
        if self.model_type == 'dreamer':
            # Use DreamerV3 loss computation
            if not isinstance(model_out, dict):
                raise ValueError(
                    f"DreamerV3 forward must return a dict, got {type(model_out)}"
                )
            return self.model.compute_loss(model_out, normalized_images, rewards, dones, self.device)
        elif self.model_type == 'simple':
            # TODO: Part 1.2 - Implement SimpleWorldModel training loss
            # Compute MSE loss between predicted and target poses/rewards
            # Ensure rewards are always (B, T)
            pred_poses = model_out['pred_poses']
            pred_rewards = model_out['pred_rewards']
            if pred_rewards is None:
                raise ValueError("SimpleWorldModel path expected pred_rewards, got None")
            if pred_rewards.dim() == 3 and pred_rewards.shape[-1] == 1:
                pred_rewards = pred_rewards.squeeze(-1)
            # Check shape of pred_poses and pred_rewards
            # print(f"Predicted poses shape: {pred_poses.shape}, Predicted rewards shape: {pred_rewards.shape}")
            if pred_poses.dim() == 2:
                print(
                    f"Warning: Predicted poses have shape {pred_poses.shape}, expected (B, T, 7). Check model output formatting.")
                raise ValueError("SimpleWorldModel output must be (B, T, 7); got 2D tensor")
            elif pred_poses.dim() == 3 and pred_poses.shape[2] != 7:
                print(
                    f"Warning: Predicted poses have last dimension {pred_poses.shape[2]}, expected 7. Check model output formatting.")
                raise ValueError("SimpleWorldModel pose dim must be 7")
            elif pred_poses.dim() == 3 and pred_poses.shape[2] == 7:
                B, T, _ = pred_poses.shape

                # Align shapes: predict at times [0..T-2] to match targets [1..T-1]
                pred_pose_seq = pred_poses[:, : T - 1, :]
                tgt_pose_seq = poses[:, 1:, :]

                # Rewards are (B, T). Use the same alignment.
                pred_rew_seq = pred_rewards
                tgt_rew_seq = rewards

                loss_dict = self.model.compute_loss(
                    pred_pose_seq,
                    pred_rew_seq,
                    target_pose=tgt_pose_seq,
                    target_reward=tgt_rew_seq,
                )
                return loss_dict

            raise ValueError(f"Unexpected pred_poses shape: {pred_poses.shape}")


class LIBERODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # crawl the data_dir and build the index map for h5py files
        self.index_map = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, 'r') as f:
                        for demo_key in f['data'].keys():
                            self.index_map.append((file_path, demo_key))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # Load your data here
        # data_path = os.path.join(self.data_dir, self.data_files[idx])
        file_path, demo_key = self.index_map[idx]
        # data_list = []
        with h5py.File(file_path, 'r') as f:
            # for demo in f['data'].keys():
            demo = f['data'][demo_key]
            image = torch.from_numpy(
                f['data'][demo_key]['obs']['agentview_rgb'][()])
            action = torch.from_numpy(f['data'][demo_key]['actions'][()])
            dones = torch.from_numpy(f['data'][demo_key]['dones'][()])
            rewards = torch.from_numpy(f['data'][demo_key]['rewards'][()])
            # poses = torch.from_numpy(f['data'][demo_key]['robot_states'][()])
            poses = torch.from_numpy(np.concatenate((f['data'][demo_key]['obs']["ee_pos"],
                                                     f['data'][demo_key]['obs']["ee_ori"][:, :3],
                                                     (f['data'][demo_key]['obs']["gripper_states"][:, :1])), axis=-1))
            # Note: Images are returned in channel-last format (T, H, W, C)
            # Conversion to channel-first (T, C, H, W) happens in the training loop
        # Return the image and label if needed
        return image, action, rewards, dones, poses


class CircularBufferDataset(torch.utils.data.Dataset):
    """Circular buffer dataset that holds up to max_trajectories.
    When full, oldest trajectories are overwritten.
    """

    def __init__(self, cfg=None, data_dir=None):
        self.trajectories = []
        self.write_idx = 0
        self._cfg = cfg

        if data_dir is None:
            data_dir = getattr(cfg, 'data_dir', None)
            if data_dir is None and cfg is not None:
                data_dir = getattr(
                    getattr(cfg, 'dataset', None), 'data_dir', None)
            if data_dir is None:
                data_dir = '/network/projects/real-g-grp/libero/targets_clean/'

        if cfg.dataset.load_dataset:
            dataset = LIBERODatasetLeRobot(
                repo_id=cfg.dataset.to_name,
                transform=transforms.ToTensor(),
                cfg=cfg
            )
        else:
            data_dir = getattr(
                cfg.dataset, 'data_dir', '/network/projects/real-g-grp/libero/targets_clean/')
            dataset = LIBERODataset(data_dir, transform=transforms.ToTensor())
        num_to_load = min(len(dataset), self._cfg.dataset.buffer_size)
        if num_to_load == 0:
            return

        indices = np.random.choice(
            len(dataset), size=num_to_load, replace=False)
        for idx in range(num_to_load):
            images, actions, rewards, dones, poses = dataset[idx]

            # dones = np.zeros_like(rewards)
            # dones[-1] = 1

            self.add_trajectory(
                np.array(images),
                np.array(actions),
                np.array(rewards),
                np.array(dones),
                np.array(poses)
            )

    def add_trajectory(self, images, actions, rewards, dones, poses):
        """Add a trajectory to the buffer. Overwrites oldest if full."""
        trajectory = {
            'images': torch.from_numpy(images),
            'actions': torch.from_numpy(actions),
            'rewards': torch.from_numpy(rewards),
            'dones': torch.from_numpy(dones),
            'poses': torch.from_numpy(poses)
        }

        if len(self.trajectories) < self._cfg.dataset.buffer_size:
            self.trajectories.append(trajectory)
        else:
            # Overwrite oldest trajectory
            self.trajectories[self.write_idx] = trajectory
            self.write_idx = (
                self.write_idx + 1) % self._cfg.dataset.buffer_size
            
    def get_trajectory(self, idx):
        trajectory = []
        traj = self.trajectories[idx]
        for i in range(len(traj['images'])):
            step_dict = {
                'observation': traj['images'][i],
                'action': traj['actions'][i],
                'reward': traj['rewards'][i],
                'done': traj['dones'][i],
                'pose': traj['poses'][i]
            }
            trajectory.append(step_dict)
        return trajectory

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return traj['images'], traj['actions'], traj['rewards'], traj['dones'], traj['poses']


class LIBERODatasetLeRobot(torch.utils.data.Dataset):

    """A dataset class for loading LIBERO data from the LeRobot repository."""

    def __init__(self, repo_id, transform=None, cfg=None):
        # super().__init__(repo_id, transform)
        self.repo_id = repo_id
        self.transform = transform
        self._dataset = datasets.load_dataset(repo_id, split='train[:{}]'.format(
            cfg.dataset.buffer_size), keep_in_memory=True)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # Load trajectory data from LeRobot dataset
        sample = self._dataset[idx]

        # Extract trajectory components
        images = torch.from_numpy(np.array(sample['img'])).float()
        actions = torch.from_numpy(np.array(sample['action'])).float()
        rewards = torch.from_numpy(np.array(sample['rewards'])).float(
        ) if 'rewards' in sample else torch.zeros(len(actions))
        dones = torch.from_numpy(np.array(sample['terminated'])).float(
        ) if 'terminated' in sample else torch.zeros(len(actions))
        poses = torch.from_numpy(np.array(sample['poses'])).float(
        ) if 'poses' in sample else torch.zeros(len(actions), 7)

        # Note: Images are returned in channel-last format (T, H, W, C)
        # Conversion to channel-first (T, C, H, W) happens in the training loop

        return images, actions, rewards, dones, poses


# ---------------------------------------------------------------------------
# Powerful stochastic policy network
# ---------------------------------------------------------------------------
class _ResLayer(torch.nn.Module):
    """Pre-norm residual MLP block: LayerNorm → Linear(d→2d) → SiLU → Linear(2d→d) + skip."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc1  = torch.nn.Linear(dim, dim * 4)
        self.act  = torch.nn.SiLU()
        self.fc2  = torch.nn.Linear(dim * 4, dim)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class PolicyNet(torch.nn.Module):
    """Expressive Gaussian policy for both SimpleWorldModel and DreamerV3.

    Architecture
    ────────────
    input_proj  : Linear(in_dim → hidden_dim) + LayerNorm + SiLU
    trunk       : N × _ResLayer(hidden_dim)   (pre-norm residual blocks)
    mean_head   : Linear → SiLU → Linear → Tanh   → action means in [-1, 1]
    logstd_head : Linear → SiLU → Linear → clamp  → log-std in [-5, 2]

    Forward returns torch.cat([mean, log_std], dim=-1)  shape (B, 2*action_dim)
    so it is a drop-in replacement for the old nn.Sequential policy.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, in_dim: int, action_dim: int,
                 hidden_dim: int = 512, n_layers: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.action_dim = action_dim

        # Input projection: lifts any input size into the hidden space
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
        )

        # Deep residual trunk
        self.trunk = torch.nn.Sequential(
            *[_ResLayer(hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )

        # Separate heads for mean and log-std → richer uncertainty estimates
        neck_dim = hidden_dim // 2
        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, neck_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(neck_dim, action_dim),
            torch.nn.Tanh(),           # bounded action means in [-1, 1]
        )
        self.logstd_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, neck_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(neck_dim, action_dim),
        )

    def forward(self, x):
        h = self.trunk(self.input_proj(x))
        mean    = self.mean_head(h)                                           # (B, A) in [-1,1]
        log_std = self.logstd_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)  # (B, A)
        return torch.cat([mean, log_std], dim=-1)                             # (B, 2A)


@hydra.main(version_base=None, config_path="./conf", config_name="64pix-pose")
def my_main(cfg: DictConfig):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb = None
    os.makedirs("checkpoints", exist_ok=True)
    subcheckpoint_dir = os.path.join("checkpoints", f"{cfg.experiment.name}")
    os.makedirs(subcheckpoint_dir, exist_ok=True)
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config=OmegaConf.to_container(cfg),
            name=cfg.experiment.name,
        )
        wandb.run.log_code(".")

    # Get model type from config or default to 'dreamer'
    model_type = getattr(cfg, 'model_type', 'dreamer')
    print(f"[info] Using model type: {model_type}")

    # Initialize the model using factory
    img_shape = [3, 64, 64]
    model = create_model(model_type, img_shape,
                         action_dim=7, device=device, cfg=cfg)

    # Wrap model for unified training interface
    model_wrapper = ModelTrainingWrapper(model, model_type, device)

    # Initialize planner (works with both model types through the model interface)
    if cfg.use_policy:
        print("[info] Using policy-based planner (CEMPlanner with policy)")
        import torch.nn as nn

        # PolicyPlanner expects the policy input to match the planner's state feature:
        # - SimpleWorldModel: encoded pose (dim=7)
        # - DreamerV3: concat([h, z]) with dim = deter_dim + stoch_dim * discrete_dim
        if model_type == 'dreamer':
            policy_in_dim = int(model.deter_dim + model.stoch_dim * model.discrete_dim)
        else:
            policy_in_dim = 7

        # Stochastic policy: outputs [mean (Tanh-bounded), log_std] concatenated → shape (B, 14).
        # _PolicyNet: deep residual MLP with pre-norm blocks and separate mean/log-std heads.
        policy = PolicyNet(in_dim=policy_in_dim, action_dim=7, hidden_dim=256, n_layers=2, dropout=cfg.policy.dropout)
        policy.to(device)
        planner = PolicyPlanner(
            model,
            policy_model=policy,
            action_dim=7,
            cfg=cfg
        )
        if cfg.planner.type == 'policy_guided_cem':
            # Load pretrained policy model for policy-guided CEM
            print(f"[info] Loading pretrained policy model from {cfg.load_policy}")
            planner.load_policy_model(cfg.load_policy)
    else:
        planner = CEMPlanner(
            model,
            action_dim=7,
            cfg=cfg
        )

    # Initialize circular buffer dataset
    if cfg.use_random_data:
        print("[info] Using CircularBufferDataset with random data collection")
        dataset = CircularBufferDataset(cfg=cfg)
        print(f"[info] Initialized buffer with {len(dataset)} trajectories")
    else:
        # Use Hugging Face dataset by default for portability; fall back to local HDF5 if requested.
        if cfg.dataset.load_dataset:
            dataset = LIBERODatasetLeRobot(
                repo_id=cfg.dataset.to_name,
                transform=transforms.ToTensor(),
                cfg=cfg
            )
        else:
            data_dir = getattr(
                cfg.dataset, 'data_dir', '/network/projects/real-g-grp/libero/targets_clean/')
            dataset = LIBERODataset(data_dir, transform=transforms.ToTensor())

    load_world_model = getattr(cfg, 'load_world_model', None)
    if load_world_model is not None:
        planner.load_world_model(load_world_model)
        print(f"[info] Loaded world model weights from {load_world_model}")

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Add linear learning rate scheduler that decays to 0 over training
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,  # Start at full learning rate
        end_factor=0.01,    # End at 0 learning rate
        total_iters=cfg.max_iters     # Decay over num_epochs
    )
    policy_loss = 0

    # Training loop
    for epoch in range(cfg.max_iters):
        num_idx = np.arange(len(dataset))
        np.random.shuffle(num_idx)
        loss = 0.0
        policy_loss = 0.0
        batch_counter = 0
        # Accumulate all encoded poses and actions for policy training at the end of the epoch
        if epoch == 0 or ((epoch-1) % cfg.eval_vid_iters == 0):
            print(f"[info] Starting epoch {epoch+1}/{cfg.max_iters} with {len(dataset)} trajectories in dataset")
            # Batch data using the batch_data utility function
            dataloader = batch_data(dataset, batch_size=cfg.batch_size, cfg=cfg)

        # Process data in batches
        for batch in dataloader:
            images, actions, rewards, dones, poses = batch
            # Normalize poses and actions for SimpleWorldModel
            normalized_poses = model.encode_pose(poses)
            normalized_actions = model.encode_action(actions)
            normalized_images = ((images.float() / 127.5) - 1.0).to(cfg.device) if model_type == 'dreamer' else None

            # Training world model on the batch
            model.train()  # Set model to training mode
            ## Call model_wrapper.forward_pass() with appropriate inputs based on model type
            if model_type == 'dreamer':
                if (cfg.use_policy and (cfg.planner.type == 'policy' or cfg.planner.type == 'policy_guided_cem')):  
                    # PolicyPlanner.update() for Dreamer expects image sequences (B,T,C,H,W)
                    # so it can encode them and build RSSM features [h,z] as policy inputs.
                    policy_loss = planner.update(normalized_images, normalized_actions)
                model_out = model_wrapper.forward_pass(normalized_images, None, normalized_actions)
                loss_dict = model_wrapper.compute_loss(
                    model_out,
                    normalized_images,
                    rewards,
                    dones,
                    None,
                    None,
                )
                batch_loss = loss_dict['total_loss']
            elif model_type == 'simple':
                if (cfg.use_policy and (cfg.planner.type == 'policy' or cfg.planner.type == 'policy_guided_cem')):  
                    policy_loss = planner.update(normalized_poses, normalized_actions)
                model_out = model_wrapper.forward_pass(
                    None,
                    normalized_poses,
                    normalized_actions,
                )
                loss_dict = model_wrapper.compute_loss(
                    model_out,
                    None,
                    rewards,
                    dones,
                    normalized_poses,
                    normalized_actions,
                )
                batch_loss = loss_dict['total_loss']
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            optimizer.zero_grad()
            batch_loss.backward()
            # Clip gradients — essential for DreamerV3: without this, prior/posterior logits
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss = batch_loss.item()
            batch_counter += 1
            # Implement data loading and training step for the batch
            if model_type == 'dreamer':
                # Dreamer: log the components for quick debugging.
                print(
                    f"Epoch [{epoch+1}/{cfg.max_iters }], Batch [{batch_counter}/{(len(dataset) + cfg.batch_size - 1) // cfg.batch_size}], "
                    f"Loss: {batch_loss.item():.4f}, recon: {loss_dict['recon_loss'].item():.4f}, "
                    f"reward: {loss_dict['reward_loss'].item():.4f}, cont: {loss_dict['continue_loss'].item():.4f}, "
                    f"dyn: {loss_dict['dyn_loss'].item():.4f}, rep: {loss_dict['rep_loss'].item():.4f}, policy_loss: {policy_loss:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{cfg.max_iters }], Batch [{batch_counter}/{(len(dataset) + cfg.batch_size - 1) // cfg.batch_size}], "
                    f"Loss: {batch_loss.item():.4f}, policy_loss: {policy_loss:.4f}"
                )

        # Log training loss to wandb
        if wandb is not None:
            if model_type == 'dreamer':
                log_payload = {
                    "train_loss": loss,
                    "policy_loss": policy_loss,
                    "loss/recon": float(loss_dict['recon_loss'].detach().cpu()),
                    "loss/reward": float(loss_dict['reward_loss'].detach().cpu()),
                    "loss/continue": float(loss_dict['continue_loss'].detach().cpu()),
                    "loss/dyn": float(loss_dict['dyn_loss'].detach().cpu()),
                    "loss/rep": float(loss_dict['rep_loss'].detach().cpu()),
                }
            else:
                log_payload = {
                    "train_loss": loss, 
                    "policy_loss": policy_loss,
                    "pose_loss": float(loss_dict['pose_loss'].detach().cpu()),
                    "reward_loss": float(loss_dict['reward_loss'].detach().cpu())
                }
            # log_payload = {"train_loss": loss, "policy_loss": policy_loss}
            # # If the last computed loss was Dreamer-style, add its components.
            # if 'loss_dict' in locals() and isinstance(locals().get('loss_dict', None), dict):
            #     ld = locals()['loss_dict']
            #     log_payload.update(
            #         {
            #             "loss/recon": float(ld['recon_loss'].detach().cpu()),
            #             "loss/reward": float(ld['reward_loss'].detach().cpu()),
            #             "loss/continue": float(ld['continue_loss'].detach().cpu()),
            #             "loss/dyn": float(ld['dyn_loss'].detach().cpu()),
            #             "loss/rep": float(ld['rep_loss'].detach().cpu()),
            #         }
            #     )
            wandb.log(log_payload)

        # save the model checkpoint
        if epoch % cfg.eval_vid_iters == 0:
            torch.save(model.state_dict(), os.path.join(subcheckpoint_dir, f'model_epoch_{epoch+1}_batch_{batch_counter}.pth'), pickle_module=dill)
            # Save policy model if using policy-based planner
            if cfg.use_policy:
                torch.save(planner.policy_model.state_dict(), os.path.join(subcheckpoint_dir, f'policy.pth'), pickle_module=dill)
            # Evaluate the model using eval_libero from sim_eval
            print("[info] Starting evaluation on LIBERO tasks...")
            # Import lazily so importing this module doesn't require robosuite/LIBERO deps.
            try:
                from .sim_eval import eval_libero
            except ImportError:
                from sim_eval import eval_libero
            data = eval_libero(planner, device, cfg, iter_=epoch, log_dir="./",
                               wandb=wandb)
            if cfg.use_random_data:
                # Add new random trajectories to the buffer
                for traj in data['traj']:
                    dones = np.zeros_like(traj['rewards'])
                    dones[-1] = 1
                    # observations need to be changed to channel first
                    # (T, 1, H, W, C) -> (T, H, W, C)
                    observations = np.array(traj['observations'])
                    # (T, H, W, C) -> (T, C, H, W)
                    # observations = np.transpose(observations, (0, 3, 1, 2))
                    dataset.add_trajectory(observations, np.array(traj['actions']),
                                           np.array(traj['rewards']), np.array(dones), np.array(traj['poses']))
                print(
                    f"[info] Added new random trajectories to buffer. Current buffer size: {len(dataset)}")

        # Step the learning rate scheduler after each epoch
        scheduler.step()
        print(
            f'Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}')
    torch.save(model.state_dict(), os.path.join(subcheckpoint_dir, f'world_model.pth'), pickle_module=dill)

if __name__ == '__main__':
    my_main()
