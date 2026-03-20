"""
HW3 Part 1: Train a dense MLP policy with PPO on privileged state.

Usage:
    python hw3/train_dense_rl.py \
        experiment.name=hw3_dense_ppo_seed0 \
        r_seed=0 \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9] \
        training.total_env_steps=200000 \
        training.rollout_length=128 \
        training.ppo_epochs=10 \
        training.minibatch_size=256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"
# Ensure the vendored LIBERO package is importable even if it hasn't been pip-installed.
# Hydra may change the working directory, so we resolve relative to this file.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIBERO_ROOT = _REPO_ROOT / "LIBERO"
if _LIBERO_ROOT.exists():
    sys.path.insert(0, str(_LIBERO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv


# ---------------------------------------------------------------------------
# Policy and value networks
# ---------------------------------------------------------------------------

class DensePolicy(nn.Module):
    """
    MLP policy that maps privileged state observations to action distributions.
    Outputs a Gaussian distribution (mean + log_std) over the 7-DoF action space.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            module = nn.Linear(in_dim, hidden_dim)
            # Orthogonal init for stability
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.constant_(module.bias, 0)
            
            layers.append(module)
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Mean head usually initialized with gain 0.01 for smaller initial actions.
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        # PPO-friendly default: state-independent log std (one per action dim).
        # This is widely used and tends to be more stable than predicting std from obs.
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.tanh = nn.Tanh() 

        # Numerical safety bounds for std; can be overridden later if needed.
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, obs_dim) float tensor
        Returns:
            dist: torch.distributions.Normal over actions
        """
        h = self.net(obs)
        mean = self.mean_head(h)

        # Expand (action_dim,) -> (B, action_dim) without allocating new storage.
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action and return (action, log_prob, entropy)."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy


class DenseValueFunction(nn.Module):
    """MLP value function V(s) for PPO critic."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            module = nn.Linear(in_dim, hidden_dim)
            # Orthogonal init for stability
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.constant_(module.bias, 0)
            
            layers.append(module)
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Value head usually initialized with gain 1.0
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate of shape (B,)."""
        return self.value_head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a fixed-length on-policy rollout for PPO updates."""

    def __init__(self, rollout_length: int, obs_dim: int, action_dim: int, device: torch.device):
        self.rollout_length = rollout_length
        self.device = device
        self.obs = torch.zeros(rollout_length, obs_dim, device=device)
        self.actions = torch.zeros(rollout_length, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_length, device=device)
        self.rewards = torch.zeros(rollout_length, device=device)
        self.values = torch.zeros(rollout_length, device=device)
        self.dones = torch.zeros(rollout_length, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def full(self):
        return self.ptr >= self.rollout_length

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages.

        Args:
            last_value: bootstrap value V(s_T) of shape ()
            gamma: discount factor
            gae_lambda: GAE lambda
        Returns:
            returns: (rollout_length,) tensor
            advantages: (rollout_length,) tensor
        """
        # Compute GAE advantages.
        # Returns are then R_t = A_t + V(s_t).
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)

        # Ensure scalar for bootstrap value.
        next_value = last_value.squeeze()
        next_adv = torch.zeros((), device=self.device)

        for t in reversed(range(self.rollout_length)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            next_adv = delta + gamma * gae_lambda * not_done * next_adv
            advantages[t] = next_adv
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return returns, advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy: DensePolicy,
               value_fn: DenseValueFunction,
               optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer,
               returns: torch.Tensor,
               advantages: torch.Tensor,
               cfg: DictConfig):
    """
    Perform `ppo_epochs` passes of minibatch PPO updates on the stored rollout.

    Returns a dict of mean losses for logging.
    """
    # Flatten rollout (T, ...) -> (N, ...)
    obs = buffer.obs.detach()
    actions = buffer.actions.detach()
    old_log_probs = buffer.log_probs.detach()
    returns = returns.detach()
    advantages = advantages.detach()

    # # PPO commonly normalizes advantages per update batch.
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    mb = int(cfg.training.minibatch_size)
    if mb <= 0:
        raise ValueError(f"training.minibatch_size must be > 0, got {mb}")
    if mb > n:
        mb = n

    # Support both naming conventions used across configs.
    clip_eps = float(getattr(cfg.training, "clip_epsilon", getattr(cfg.training, "clip_eps")))
    value_coef = float(getattr(cfg.training, "value_coef", getattr(cfg.training, "value_coeff")))
    entropy_coef = float(getattr(cfg.training, "entropy_coef", getattr(cfg.training, "entropy_coeff")))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 0.5))
    # Recommended target_kl: 0.01 to 0.015 for stability
    target_kl = float(getattr(cfg.training, "target_kl", 0.0))

    # For logging
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []
    clip_fracs = []

    for _epoch in range(int(cfg.training.ppo_epochs)):
        epoch_kls = [] # Track KLs for this specific epoch
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, mb):
            idx = perm[start:start + mb]

            # 1. Minibatch Advantage Normalization
            mb_adv = advantages[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            
            # 2. Re-calculate distribution and values
            dist = policy(obs[idx])
            new_logp = dist.log_prob(actions[idx]).sum(-1)
            values_pred = value_fn(obs[idx])
            mb_old_logp = old_log_probs[idx]
            
            # 3. Clip the value to reduce variability during Critic training and compute value loss.
            # 3.a. Get the value from the rollout (stored in buffer)
            mb_old_values = buffer.values[idx]
            # 3.b. Calculate the "unclipped" loss
            mb_returns = returns[idx]
            value_loss_unclipped = (values_pred - mb_returns)**2
            # 3.c. Calculate the "clipped" loss
            values_clipped = mb_old_values + torch.clamp(values_pred - mb_old_values,-clip_eps,clip_eps)
            value_loss_clipped = (values_clipped - mb_returns)**2
            # 3.d. Take the maximum of the two (pessimistic bound)
            v_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
            value_loss = 0.5 * v_loss_max.mean()
            
            # 4. Policy loss with clipped surrogate objective.
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 5. Entropy bonus for exploration
            entropy = dist.entropy().sum(-1).mean()
            
            # 6. Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), max_grad_norm)
            optimizer.step()

            # Logging stats
            with torch.no_grad():
                log_ratio = new_logp - mb_old_logp
                approx_kl = 0.5 * (log_ratio**2).mean() # The one you found
                
                # Your clipfrac is already perfect
                clipped = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
                epoch_kls.append(approx_kl.item())
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropies.append(entropy.detach())
            approx_kls.append(approx_kl.detach())
            clip_fracs.append(clipped.detach())
        # --- Check for Early Stopping after the epoch's minibatches ---
        if target_kl > 0.0:  # Only check if target_kl is set to a positive value
            avg_epoch_kl = np.mean(epoch_kls)
            if avg_epoch_kl > target_kl:
                print(f"Early stopping at epoch {_epoch} due to reaching target KL: {avg_epoch_kl:.4f} > {target_kl}")
                break

    def _mean(xs):
        if not xs:
            return 0.0
        return torch.stack(list(xs)).mean().item()

    return {
        "policy_loss": _mean(policy_losses),
        "value_loss": _mean(value_losses),
        "entropy": _mean(entropies),
        "approx_kl": _mean(approx_kls),
        "clip_frac": _mean(clip_fracs),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="dense_ppo", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logging ---
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- Environment ---
    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(
        task_id=task_id,
        max_episode_steps=cfg.sim.episode_length,
        cfg=cfg,
    )

    obs_dim = cfg.policy.obs_dim
    action_dim = cfg.policy.action_dim

    # --- Models ---
    policy = DensePolicy(obs_dim, action_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)
    value_fn = DenseValueFunction(obs_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=cfg.training.learning_rate,
    )

    buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)

    # --- Rollout state ---
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    episode_return = 0.0
    episode_steps = 0
    total_steps = 0
    episode_returns = []
    episode_successes = []

    # --- Main loop ---
    while total_steps < cfg.training.total_env_steps:
        buffer.reset()

        # Collect one rollout
        with torch.no_grad():
            for _ in range(cfg.training.rollout_length):
                action, log_prob, _ = policy.get_action(obs_tensor.unsqueeze(0))
                value = value_fn(obs_tensor.unsqueeze(0))
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, done, truncated, info = env.step(action_np)
                episode_return += reward
                episode_steps += 1
                total_steps += 1

                buffer.add(
                    obs_tensor,
                    action.squeeze(0),
                    log_prob.squeeze(0),
                    torch.tensor(reward, device=device),
                    value.squeeze(0),
                    torch.tensor(float(done or truncated), device=device),
                )

                if done or truncated:
                    episode_returns.append(episode_return)
                    episode_successes.append(float(info.get("success_placed", 0.0)))
                    episode_return = 0.0
                    episode_steps = 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

                if buffer.full():
                    break

            # Bootstrap last value
            last_value = value_fn(obs_tensor.unsqueeze(0)).squeeze(0)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.training.gamma, cfg.training.gae_lambda
        )
        # Annealing the rate if instructed to do so.
        if cfg.training.anneal_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.training.learning_rate * (1 - total_steps / cfg.training.total_env_steps)

        # PPO update
        update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

        # Logging
        if total_steps % cfg.log_interval < cfg.training.rollout_length:
            log_dict = {
                "train/total_steps": total_steps,
                **{f"train/{k}": v for k, v in update_info.items()},
            }
            if episode_returns:
                log_dict["train/episode_return"] = np.mean(episode_returns[-10:])
                log_dict["train/success_rate"] = np.mean(episode_successes[-10:])
            wandb.log(log_dict, step=total_steps)
            print(f"[{total_steps}/{cfg.training.total_env_steps}] "
                  f"return={log_dict.get('train/episode_return', float('nan')):.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

        # Checkpoint
        if total_steps % cfg.save_interval < cfg.training.rollout_length:
            ckpt = {
                "policy": policy.state_dict(),
                "value_fn": value_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_steps": total_steps,
                "cfg": OmegaConf.to_container(cfg),
            }
            # Make a directory for checkpoints if it doesn't exist.
            os.makedirs(os.path.join("checkpoints", cfg.experiment.name), exist_ok=True)
            torch.save(ckpt, os.path.join("checkpoints", cfg.experiment.name, f"dense_ppo_{total_steps}.pth"))

    # Final save
    torch.save({"policy": policy.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               os.path.join("checkpoints", cfg.experiment.name, "dense_ppo_final.pth"))
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
