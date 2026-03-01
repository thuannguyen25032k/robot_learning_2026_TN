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
import wandb

# Support both `python hw2/dreamer_model_trainer.py` (cwd=hw2) and
# `python -m hw2.dreamer_model_trainer` / importing as a package.
try:
    from .dreamerV3 import DreamerV3
    from .simple_world_model import SimpleWorldModel
    from .planning import CEMPlanner, PolicyPlanner, RandomPlanner
    from .sim_eval import eval_libero
except ImportError:
    from dreamerV3 import DreamerV3
    from simple_world_model import SimpleWorldModel
    from planning import CEMPlanner, PolicyPlanner, RandomPlanner
    from sim_eval import eval_libero
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
            planner.load_policy_model(cfg.load_policy)
    else:
        planner = CEMPlanner(
            model,
            action_dim=7,
            cfg=cfg
        )
        
    planner.load_world_model(cfg.load_world_model)
    print(f"[info] Loaded world model weights from {cfg.load_world_model}")

    # Test the planner by running evaluation rollouts and logging results to wandb
    for epoch in range(cfg.sim.eval_episodes):
        data = eval_libero(planner, cfg.device, cfg, iter_=epoch, log_dir="./",
                                wandb=wandb)

if __name__ == '__main__':
    my_main()
