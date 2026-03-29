"""
HW3 Part 2: Fine-tune a transformer policy from HW1 with PPO or GRPO.

Usage (PPO):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_ppo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=ppo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]

Usage (GRPO with ground-truth resets):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_grpo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=grpo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import dill
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.core.hydra_config import HydraConfig
from hw3.train_dense_rl import RolloutBuffer, ppo_update
from hw3.libero_env_fast import FastLIBEROEnv

# Pre-compute log(2) as a scalar constant so _tanh_log_prob does not
# allocate a new CPU tensor on every call.
_LOG2 = math.log(2.0)

# ---------------------------------------------------------------------------
# Value function built directly on a GRP checkpoint
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """Value network V(s) built directly from a GRP checkpoint.

    Loads the same GRP model from ``checkpoint_path``, replaces the action head
    (``model.mlp``) with a new ``value_head`` (Linear → scalar), and runs the
    full GRP forward pass up to the CLS token to produce V(s).

    - **Shared** (``shared_network=True``): the ``model`` attribute points to the
      *same* GRP model object as the policy so transformer weights are shared
      and updated jointly.
    - **Separate** (``shared_network=False``): loads an independent copy of the
      checkpoint so the critic can specialise without interfering with the actor.

    Goal conditioning is passed explicitly to ``forward()`` on every call —
    there is no cached state on this object, so stale-context bugs are impossible.
    """

    def __init__(
        self,
        policy: "TransformerPolicyWrapper",
        device: torch.device,
        cfg: DictConfig,
        shared_network: bool = True,
    ):
        super().__init__()
        self.device = device
        self.cfg    = cfg
        self.shared_network = shared_network

        if shared_network:
            # Point at the policy's own GRP model — no extra parameters.
            self.model = policy.model
        else:
            # Load an independent copy from the same checkpoint.
            self.model = torch.load(cfg.init_checkpoint, map_location=device, pickle_module=dill)
            self.model.to(device)

        n_embd = int(self.model._cfg.n_embd)
        self.value_head = nn.Sequential(
            nn.Linear(n_embd, 1),
        )
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.constant_(self.value_head[-1].bias, 0)
        self.to(device)

    def forward(
        self,
        obs: torch.Tensor,
        txt_goal: torch.Tensor,
        goal_state: torch.Tensor,
        pose: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute V(s).

        Runs the full GRP forward pass (identical to the policy) but feeds the
        CLS token into ``value_head`` instead of the action head.

        Args:
            obs:        (B, H, W, C) float image observation (raw [0,255] or [-1,1]).
            txt_goal:   (1, T) or (1, T, n_embd) text goal tensor from
                        ``TransformerPolicyWrapper.encode_goals()``.
            goal_state: (1, H, W, C) normalised [-1,1] goal image from
                        ``TransformerPolicyWrapper.encode_goals()``.
            pose:       (B, pose_emb_dim) encoded pose, or None.
        Returns:
            values: (B,) scalar estimates.
        """
        obs = obs.to(self.device)
        # Normalise raw [0,255] images to the [-1,1] range the GRP was trained on.
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(127.5).sub_(1.0)
        elif obs.max() > 1.5:
            obs = obs.div(127.5).sub(1.0)
        B = obs.shape[0]

        # Expand (1, …) goal tensors to match the batch dimension.
        txt_goal_b   = txt_goal.to(self.device).expand(B,   *txt_goal.shape[1:])
        goal_state_b = goal_state.to(self.device).expand(B, *goal_state.shape[1:])

        # Run the GRP forward pass up to the CLS token, bypassing model.mlp.
        # We monkey-patch model.mlp temporarily with nn.Identity so that GRP's
        # "logits = self.mlp(cls_token)" returns the raw (B, n_embd) CLS vector.
        # This is single-threaded safe (training loop never calls policy and
        # value_fn concurrently) and avoids duplicating the transformer code.
        original_mlp = self.model.mlp
        self.model.mlp = nn.Identity()
        try:
            cls_tok, _ = self.model(obs, txt_goal_b, goal_state_b, pose=pose)
        finally:
            self.model.mlp = original_mlp  # always restore, even if forward() throws

        return self.value_head(cls_tok).squeeze(-1)


# ---------------------------------------------------------------------------
# Transformer policy wrapper
# ---------------------------------------------------------------------------

class TransformerPolicyWrapper(nn.Module):
    """PPO-compatible wrapper around the HW1 GRP transformer.

    Key contract for PPO (``ppo_update`` in ``train_dense_rl.py``):
      - callable: ``dist = policy(obs_batch, txt_goal, goal_state)`` → Normal
      - ``get_action()`` → ``(action_np, log_prob, entropy, pre_tanh)``

    Goal conditioning is passed **explicitly** on every call — there is no
    hidden cached state, which makes the importance-weight ratio in PPO/GRPO
    exact even when a minibatch spans multiple episodes with different goals.
    Use ``encode_goals()`` once per episode to obtain the tensors.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, cfg: DictConfig):
        super().__init__()
        self.model  = torch.load(checkpoint_path, map_location=device, pickle_module=dill)
        self.action_head = self.model.mlp
        self.cfg         = cfg
        self.device      = device

        # action_dim: use model.mlp[0] (the Linear layer) regardless of whether
        # the model is discrete (mlp has 2 layers: Linear + LayerNorm) or
        # continuous (mlp has 1 layer: Linear).  mlp[0] is always the Linear.
        action_out_features = self.action_head[0].out_features

        # Learnable state-independent log std for Gaussian head.
        self._action_log_std = nn.Parameter(torch.zeros(
            action_out_features, device=device
        ))
        nn.init.constant_(self._action_log_std, -1)  # std ~ 0.45 at start of RL
        # Cache decode_action tensors on device so forward() never rebuilds them.
        self.register_buffer(
            "_action_mean",
            torch.tensor(self.model._cfg.dataset.action_mean, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_action_std",
            torch.tensor(self.model._cfg.dataset.action_std, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.to(device)

    def encode_goals(self, first_obs: np.ndarray, instruction: str):
        """Encode goal conditioning for one episode and return the tensors.

        Returns:
            txt_goal:   (1, T) int64 token ids  OR  (1, T, n_embd) T5 floats
            goal_state: (1, H, W, C) float tensor normalised to [-1, 1]

        The caller owns these tensors and passes them explicitly to
        ``forward()`` and ``get_action()`` on every step of the episode.
        This avoids any implicit cached state on the wrapper.
        """
        model = self.model
        txt_goal = model.encode_text_goal(instruction).to(self.device)
        if first_obs is not None:
            # preprocess_goal_image: resize + (img/255)*2-1 → (H, W, C) float [-1,1]
            goal_np    = model.preprocess_goal_image(first_obs)
            goal_state = torch.tensor(
                goal_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)   # (1, H, W, C)
        else:
            goal_state = torch.zeros(
                1, *model._cfg.image_shape, dtype=torch.float32, device=self.device
            )
        return txt_goal, goal_state

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise a raw [0,255] image tensor to [-1,1]."""
        if obs.dtype == torch.uint8:
            return obs.float().div_(127.5).sub_(1.0)
        if obs.max() > 1.5:          # float still in [0, 255]
            return obs.div(127.5).sub(1.0)
        return obs                   # already in [-1, 1]

    def forward(
        self,
        obs: torch.Tensor,
        txt_goal: torch.Tensor,
        goal_state: torch.Tensor,
        pose: torch.Tensor = None,
    ) -> Normal:
        """Return the action distribution for a batch of observations.

        Args:
            obs:        (B, H, W, C) raw or normalised image observation.
            txt_goal:   (1, T[, n_embd]) text goal from ``encode_goals()``.
            goal_state: (1, H, W, C) goal image from ``encode_goals()``.
            pose:       (B, pose_emb_dim) encoded pose, or None.
        Returns:
            Normal distribution in z-score (GRP normalised) action space.
        """
        obs = self._preprocess_obs(obs.to(self.device))
        B   = obs.shape[0]

        # Expand (1, …) goal tensors to the batch size — zero-copy views.
        txt_goal_b   = txt_goal.to(self.device).expand(B,   *txt_goal.shape[1:])
        goal_state_b = goal_state.to(self.device).expand(B, *goal_state.shape[1:])

        raw_logits, _ = self.model(obs, txt_goal_b, goal_state_b, pose=pose)
        # raw_logits is in the GRP normalised action space (z-scores).
        # Keep the distribution in that space — tanh squashing is done in
        # get_action(), so the pretrained mean is faithfully preserved at
        # the start of RL fine-tuning and the log-prob is self-consistent.
        log_std    = self._action_log_std.clamp(-5.0, 1.0)
        action_std = log_std.exp().expand_as(raw_logits)
        return Normal(raw_logits, action_std)

    def _decode_action(self, normalised_action: torch.Tensor) -> torch.Tensor:
        """Decode GRP normalised action (z-score) → raw action space."""
        return normalised_action * self._action_std + self._action_mean

    @staticmethod
    def _log_prob(dist: Normal, z_action: torch.Tensor) -> torch.Tensor:
        """Log-prob of a sampled z-score action under the Gaussian distribution.

        The policy distribution lives in z-score space and the env action is
        obtained via linear decode (z * std + mean), which is a volume-preserving
        affine map up to a constant Jacobian that cancels in importance ratios.
        No Jacobian correction is needed here.
        """
        return dist.log_prob(z_action).sum(-1)

    # Keep old name as alias so grpo_update and ppo_update call sites don't break.
    _tanh_log_prob = _log_prob

    def get_action(
        self,
        obs_t: torch.Tensor,
        txt_goal: torch.Tensor,
        goal_state: torch.Tensor,
        pose: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """Sample an action and return rollout data.

        Args:
            obs_t:      (1, H, W, C) or (H, W, C) raw image tensor.
            txt_goal:   (1, T[, n_embd]) from ``encode_goals()``.
            goal_state: (1, H, W, C) from ``encode_goals()``.
            pose:       (1, pose_emb_dim) encoded pose, or None.
            deterministic: if True, use dist.mean (no sampling).
        Returns:
            action_np:  (action_dim,) numpy array in the env's action space.
            log_prob:   scalar tensor (detached).
            entropy:    scalar tensor (detached).
            z_sample:   (action_dim,) tensor in z-score space (for PPO buffer).
        """
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        dist     = self.forward(obs_t, txt_goal, goal_state, pose)
        z_sample = dist.mean if deterministic else dist.rsample()  # z-score space
        log_prob = self._log_prob(dist, z_sample)
        entropy  = dist.entropy().sum(-1)
        # Decode z-score → raw action space (same transform used in BC training
        # and sim_eval.py: action = z * action_std + action_mean).
        action_t = self._decode_action(z_sample)
        return (
            action_t.squeeze(0).detach().cpu().numpy(),
            log_prob.squeeze(0),
            entropy.squeeze(0),
            z_sample.squeeze(0),
        )


# ---------------------------------------------------------------------------
# GRPO helpers
# ---------------------------------------------------------------------------

def _extract_pose_from_info(info: dict, policy: "TransformerPolicyWrapper", device: torch.device) -> torch.Tensor:
    """Extract and encode a (1, pose_emb_dim) pose tensor from an env info dict.

    ``FastLIBEROEnv`` stores the full state vector under ``info["state_obs"]``
    with layout ``[eef_pos(3), eef_quat_xyz(3), gripper(1), ...]``.
    Uses torch.from_numpy for a zero-copy transfer from the numpy buffer.
    """
    state_obs = info.get("state_obs", None)
    if state_obs is not None:
        # state_obs is already a numpy array; slice and copy minimally.
        pose_np = np.ascontiguousarray(state_obs[:7], dtype=np.float32)
    else:
        pose_np = np.concatenate([
            np.asarray(info["robot0_eef_pos"],           dtype=np.float32),
            np.asarray(info["robot0_eef_quat"][:3],      dtype=np.float32),
            np.asarray([info["robot0_gripper_qpos"][0]], dtype=np.float32),
        ], axis=-1)
    # from_numpy avoids a data copy; unsqueeze adds the batch dim.
    pose_t = torch.from_numpy(pose_np).unsqueeze(0)   # (1, 7)
    return policy.model.encode_pose(pose_t).to(device)  # (1, pose_emb_dim)


def collect_grpo_group(env: FastLIBEROEnv,
                       policy: TransformerPolicyWrapper,
                       init_state,
                       group_size: int,
                       max_steps: int,
                       device: torch.device):
    """
    Reset to the same initial state and collect `group_size` trajectories.

    Returns a list of trajectory dicts, each containing:
        obs, actions, log_probs, poses, txt_goal, goal_state, rewards, dones, total_return
    """
    trajectories = []
    instruction = env.instruction
    use_pose = policy.model._cfg.policy.use_pose_data

    if group_size <= 0:
        return trajectories

    # Disable dropout during collection so old_log_probs are deterministic
    # and the importance-weight ratio in grpo_update is meaningful.
    was_training = policy.training
    policy.eval()

    for _ in range(group_size):
        obs, info = env.reset(options={"init_state": init_state})
        obs = np.ascontiguousarray(obs)
        if obs.ndim != 3:
            raise ValueError(f"Expected image observations for transformer GRPO, got shape={obs.shape}")

        # Encode goal conditioning for this episode — returned as plain tensors,
        # no hidden state on the policy object.
        txt_goal, goal_state = policy.encode_goals(obs, instruction)

        traj_obs = []
        traj_actions = []
        traj_log_probs = []
        traj_poses = []
        traj_rewards = []
        traj_dones = []

        total_return = 0.0

        # Encode initial pose from reset info
        pose = _extract_pose_from_info(info, policy, device) if use_pose else None

        for _step in range(max_steps):
            # Convert obs to a GPU tensor for the network forward pass.
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                action_np, log_prob, _entropy, pre_tanh = policy.get_action(
                    obs_t, txt_goal, goal_state, pose
                )

            next_obs, reward, done, truncated, _info = env.step(action_np)
            terminal = bool(done or truncated)

            # Store on CPU immediately to keep GPU memory free during collection.
            traj_obs.append(obs_t.cpu())
            traj_actions.append(pre_tanh.cpu())
            traj_log_probs.append(log_prob.cpu())
            if use_pose and pose is not None:
                traj_poses.append(pose.squeeze(0).cpu())
            traj_rewards.append(float(reward))
            traj_dones.append(float(terminal))

            total_return += float(reward)
            obs = np.ascontiguousarray(next_obs)

            if use_pose:
                pose = _extract_pose_from_info(_info, policy, device)

            if terminal:
                break

        if traj_obs:
            trajectory = {
                "obs":        torch.stack(traj_obs,    dim=0),
                "actions":    torch.stack(traj_actions, dim=0),
                "log_probs":  torch.stack(traj_log_probs, dim=0),
                "poses":      torch.stack(traj_poses, dim=0) if traj_poses else None,
                "txt_goal":   txt_goal.cpu(),    # (1, T[, n_embd]) — same for whole traj
                "goal_state": goal_state.cpu(),  # (1, H, W, C) — same for whole traj
                "rewards":    torch.tensor(traj_rewards, dtype=torch.float32),
                "dones":      torch.tensor(traj_dones,   dtype=torch.float32),
                "total_return": float(total_return),
            }
        else:
            # Extremely defensive fallback: empty trajectory if env terminates instantly.
            trajectory = {
                "obs":        torch.empty((0, *obs.shape), dtype=torch.float32),
                "actions":    torch.empty((0, env._action_dim), dtype=torch.float32),
                "log_probs":  torch.empty((0,), dtype=torch.float32),
                "poses":      None,
                "txt_goal":   txt_goal.cpu(),
                "goal_state": goal_state.cpu(),
                "rewards":    torch.empty((0,), dtype=torch.float32),
                "dones":      torch.empty((0,), dtype=torch.float32),
                "total_return": 0.0,
            }
        trajectories.append(trajectory)

    policy.train(was_training)
    return trajectories


def grpo_update(policy: TransformerPolicyWrapper,
                value_fn: ValueFunction,
                policy_optimizer: torch.optim.Optimizer,
                trajectories_per_group: list,
                cfg: DictConfig,
                device: torch.device):
    """
    GRPO update: compute group-relative advantages and update policy.

    All valid trajectories are concatenated into a single batch so the policy
    runs one forward pass per group instead of one per trajectory, which is
    significantly faster for transformer models.

    Args:
        trajectories_per_group: list of lists; each inner list is a group of
            trajectory dicts collected from the same initial state.
    Returns:
        dict with "policy_loss", "mean_return"
    """
    clip_eps     = float(getattr(cfg.training, "clip_epsilon", getattr(cfg.training, "clip_eps")))
    entropy_coef = float(getattr(cfg.training, "entropy_coef", getattr(cfg.training, "entropy_coeff", 0.0)))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 0.5))

    # ---- 1. Build a flat list per timestep ----
    # Each trajectory carries its own txt_goal/goal_state, so we process each
    # trajectory separately (one forward pass per trajectory) to keep goal
    # conditioning exact.  This is slightly slower than one mega-batch but
    # correct — GRPO groups are small (≤8 trajs) so the cost is negligible.
    all_new_lp, all_old_lp, all_adv, all_entropy = [], [], [], []
    mean_returns, group_adv_stats = [], []

    policy_optimizer.zero_grad(set_to_none=True)
    was_training = policy.training
    policy.eval()  # eval mode for importance-weight recomputation (no dropout)

    for group in trajectories_per_group:
        if not group:
            continue
        group_returns = torch.tensor(
            [float(t.get("total_return", 0.0)) for t in group],
            dtype=torch.float32, device=device,
        )
        if group_returns.numel() == 0:
            continue
        g_mean = group_returns.mean()
        g_std  = group_returns.std(unbiased=False)
        group_adv = (group_returns - g_mean) / (g_std + 1e-8)  # (G,)

        mean_returns.append(g_mean.detach())
        group_adv_stats.append(group_adv.abs().mean().detach())

        for traj, traj_adv in zip(group, group_adv):
            obs     = traj["obs"]
            actions = traj["actions"]
            lp_old  = traj["log_probs"]
            poses   = traj["poses"]
            txt_goal   = traj["txt_goal"]
            goal_state = traj["goal_state"]
            if obs.numel() == 0:
                continue
            T = obs.shape[0]

            obs_dev    = obs.to(device)
            act_dev    = actions.to(device)
            lp_old_dev = lp_old.to(device)
            pose_dev   = poses.to(device) if poses is not None else None
            txt_dev    = txt_goal.to(device)
            gs_dev     = goal_state.to(device)

            # Recompute log-probs with the current policy using this trajectory's
            # exact goal conditioning — no stale-context risk.
            # actions stores z-score samples (output of dist.rsample()).
            dist     = policy(obs_dev, txt_dev, gs_dev, pose_dev)
            new_lp   = TransformerPolicyWrapper._log_prob(dist, act_dev)
            ent      = dist.entropy().sum(-1)

            adv_vec = torch.full((T,), float(traj_adv.item()), dtype=torch.float32, device=device)
            all_new_lp.append(new_lp)
            all_old_lp.append(lp_old_dev)
            all_adv.append(adv_vec)
            all_entropy.append(ent)

    policy.train(was_training)

    if not all_new_lp:
        return {"policy_loss": 0.0, "entropy": 0.0, "mean_return": 0.0, "group_adv_mean_abs": 0.0}

    new_lp_cat  = torch.cat(all_new_lp,  dim=0)
    old_lp_cat  = torch.cat(all_old_lp,  dim=0)
    adv_cat     = torch.cat(all_adv,     dim=0)
    entropy_cat = torch.cat(all_entropy, dim=0)

    ratio  = torch.exp(new_lp_cat - old_lp_cat)
    surr1  = ratio * adv_cat
    surr2  = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_cat
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy     = entropy_cat.mean()
    loss = policy_loss - entropy_coef * entropy
    loss.backward()

    nn.utils.clip_grad_norm_(list(policy.parameters()), max_grad_norm)
    policy_optimizer.step()

    def _mean(xs):
        return torch.stack(list(xs)).mean().item() if xs else 0.0

    return {
        "policy_loss":      policy_loss.detach().item(),
        "entropy":          entropy.detach().item(),
        "mean_return":      _mean(mean_returns),
        "group_adv_mean_abs": _mean(group_adv_stats),
    }


# ---------------------------------------------------------------------------
# GRPO with world model (Part 2d)
# ---------------------------------------------------------------------------

def grpo_worldmodel_update(policy: TransformerPolicyWrapper,
                            world_model,
                            current_obs: np.ndarray,
                            instruction: str,
                            group_size: int,
                            horizon: int,
                            cfg: DictConfig,
                            device: torch.device,
                            policy_optimizer: torch.optim.Optimizer | None = None):
    """
    GRPO using the HW2 world model to generate imagined trajectories.

    Algorithm
    ---------
    For each of ``group_size`` imagined rollouts starting from ``current_obs``:
      1. Sample an action from the policy distribution (so gradients flow).
      2. Ask the (frozen) world model for (next_obs_recon, reward, continue).
      3. Repeat for ``horizon`` steps or until the world model predicts episode end.
      4. Compute per-trajectory return = sum of predicted rewards.
    Then normalise returns within the group to get group-relative advantages and
    update the policy with the REINFORCE / clipped-surrogate objective.

    Args:
        world_model:       Trained HW2 world model.  Must be a DreamerV3 instance
                           (has ``encoder``, ``rssm_step``, ``reward_head``,
                           ``continue_head``, ``decoder`` attributes).
        current_obs:       (H, W, C) uint8 or float image — real env observation
                           used as the imagination starting point.
        group_size:        Number of imagined trajectories per update.
        horizon:           Max imagination steps per trajectory.
        policy_optimizer:  When provided the policy is updated in-place; otherwise
                           only metrics are returned (useful for debugging).
    Returns:
        dict with "policy_loss", "entropy", "mean_imagined_return",
                  "group_adv_mean_abs", "imagined_steps".
    """
    _empty = {
        "policy_loss": 0.0,
        "entropy": 0.0,
        "mean_imagined_return": 0.0,
        "group_adv_mean_abs": 0.0,
        "imagined_steps": 0,
    }
    if group_size <= 0 or horizon <= 0:
        return _empty

    clip_eps     = float(getattr(cfg.training, "clip_epsilon",
                                 getattr(cfg.training, "clip_eps", 0.2)))
    entropy_coef = float(getattr(cfg.training, "entropy_coef",
                                 getattr(cfg.training, "entropy_coeff", 0.0)))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 0.5))

    # ------------------------------------------------------------------ #
    # Validate world-model type — only DreamerV3 supports image-space     #
    # imagination (obs_shape, encoder, rssm_step, decoder, reward_head).  #
    # ------------------------------------------------------------------ #
    model_type = getattr(world_model, "type", "").lower()
    if "dreamer" not in model_type:
        raise NotImplementedError(
            "grpo_worldmodel_update requires a DreamerV3 world model. "
            f"Got type={world_model.__class__.__name__!r}."
        )

    if not isinstance(current_obs, np.ndarray):
        current_obs = np.asarray(current_obs, dtype=np.float32)
    current_obs = np.ascontiguousarray(current_obs)
    if current_obs.ndim != 3:
        raise ValueError(
            f"current_obs must be (H, W, C), got shape {current_obs.shape}"
        )

    # Encode goal conditioning once for all imagined trajectories in this group.
    # encode_goals is a pure function (no side effects on policy state).
    txt_goal, goal_state = policy.encode_goals(current_obs, instruction)

    # ------------------------------------------------------------------ #
    # Helper: (H,W,C) uint8/float numpy → (1,C,H,W) float tensor [-1,1] #
    # ------------------------------------------------------------------ #
    def _obs_to_dreamer(obs_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(obs_np)).float().to(device)
        if t.shape[-1] in (1, 3, 6, 9, 12):      # (H,W,C) → (C,H,W)
            t = t.permute(2, 0, 1)
        if t.max() > 1.5:                          # [0,255] → [-1,1]
            t = t / 127.5 - 1.0
        return t.unsqueeze(0).clamp(-1.0, 1.0)    # (1,C,H,W)

    # ------------------------------------------------------------------ #
    # Helper: dreamer recon (1,C,H,W) [-1,1] → (H,W,C) float [0,255]   #
    # for feeding back through the transformer policy.                   #
    # ------------------------------------------------------------------ #
    def _dreamer_to_policy_obs(recon: torch.Tensor) -> torch.Tensor:
        r = recon.squeeze(0)                       # (C,H,W)
        r = (r + 1.0) * 127.5                      # [-1,1] → [0,255]
        r = r.clamp(0.0, 255.0).permute(1, 2, 0)  # (H,W,C)
        return r.detach()                          # sever WM graph before policy fwd

    # ------------------------------------------------------------------ #
    # Freeze world model for imagination — we only update the policy.    #
    # ------------------------------------------------------------------ #
    was_training_wm = world_model.training
    world_model.eval()
    # Disable policy dropout during imagination — we want the score-function
    # gradient (REINFORCE) to estimate the *expected* advantage, not the
    # dropout-corrupted one.  We re-enable before the backward pass.
    was_training_policy = policy.training
    policy.eval()
    if policy_optimizer is not None:
        policy_optimizer.zero_grad(set_to_none=True)

    action_dim   = policy.action_head[0].out_features
    start_chw    = _obs_to_dreamer(current_obs)   # (1,C,H,W)

    # Per-trajectory accumulation for GRPO
    group_traj_log_probs : list[torch.Tensor] = []
    group_traj_entropies : list[torch.Tensor] = []
    group_total_returns  : list[float]        = []
    imagined_steps = 0

    with torch.no_grad():
        # Pre-encode the starting frame once to build initial RSSM state.
        embed0 = world_model.encoder(start_chw)   # (1, hidden_dim)

    for _traj in range(group_size):
        # Each trajectory starts from the SAME real observation but samples
        # different stochastic actions → different imagined futures.
        rssm_state = world_model.get_initial_state(1, device)

        # Step the RSSM once on the real observation (posterior) so the
        # deterministic state h₀ is grounded in real data.
        with torch.no_grad():
            step_out = world_model.rssm_step(
                rssm_state,
                action=torch.zeros(1, action_dim, device=device),
                embed=embed0,
            )
        rssm_state = {
            "h": step_out["h"].detach(),
            "z": step_out["z"].detach(),
            "z_probs": step_out.get("z_probs", None),
        }

        # Current "imagined" obs for the policy (H,W,C float, on device)
        cur_policy_obs = _dreamer_to_policy_obs(start_chw)   # no grad needed

        traj_log_probs : list[torch.Tensor] = []
        traj_entropies : list[torch.Tensor] = []
        traj_reward_sum = 0.0

        for _step in range(horizon):
            # ---- Policy samples action (grad flows through log_prob) ----
            dist     = policy(cur_policy_obs.unsqueeze(0), txt_goal, goal_state)  # → Normal
            z_sample = dist.rsample()                         # (1, action_dim), z-score
            action   = policy._decode_action(z_sample)        # decode to raw action space
            lp       = TransformerPolicyWrapper._log_prob(dist, z_sample)  # (1,)
            ent      = dist.entropy().sum(-1)                  # (1,)
            traj_log_probs.append(lp.squeeze(0))
            traj_entropies.append(ent.squeeze(0))

            # ---- World model step (no grad — WM is frozen) ----
            with torch.no_grad():
                step_out = world_model.rssm_step(
                    rssm_state,
                    action=action.detach(),
                    embed=None,              # imagination: use prior, not posterior
                )
                h_t = step_out["h"]          # (1, deter_dim)
                z_t = step_out["z"]          # (1, stoch_dim * discrete_dim)
                feat_t = torch.cat([h_t, z_t], dim=-1)   # (1, feat_dim)

                reward_pred  = world_model.reward_head(feat_t).squeeze()    # scalar
                cont_logit   = world_model.continue_head(feat_t).squeeze()  # scalar
                cont_prob    = torch.sigmoid(cont_logit).item()

                # Reconstruct next observation for the policy
                recon_t = world_model.decoder(feat_t)     # (1, C, H, W)

            traj_reward_sum += float(reward_pred.item())
            imagined_steps  += 1

            rssm_state = {
                "h": h_t,
                "z": z_t,
                "z_probs": step_out.get("z_probs", None),
            }
            cur_policy_obs = _dreamer_to_policy_obs(recon_t)   # (H,W,C)

            if cont_prob < 0.5:   # world model predicts episode end
                break

        if not traj_log_probs:
            continue

        group_traj_log_probs.append(torch.stack(traj_log_probs))
        group_traj_entropies.append(torch.stack(traj_entropies))
        group_total_returns.append(traj_reward_sum)

    # Restore world model training mode
    if was_training_wm:
        world_model.train()
    policy.train(was_training_policy)

    if not group_total_returns:
        return _empty

    # ------------------------------------------------------------------ #
    # Group-relative advantage normalisation (GRPO core)                 #
    # ------------------------------------------------------------------ #
    returns_t   = torch.tensor(group_total_returns, dtype=torch.float32, device=device)
    group_mean  = returns_t.mean()
    group_std   = returns_t.std(unbiased=False)
    group_advs  = (returns_t - group_mean) / (group_std + 1e-8)   # (G,)

    # ------------------------------------------------------------------ #
    # Compute policy objective and backpropagate                          #
    # ------------------------------------------------------------------ #
    # Accumulate as a Python list instead of pre-allocating a zero tensor,
    # so total_loss is always a real computation graph node (requires_grad=True)
    # when there are trajectories — avoiding the "tensor(0.0).requires_grad=False"
    # pitfall that would silently skip the optimizer step.
    total_loss_terms : list[torch.Tensor] = []
    policy_losses : list[torch.Tensor] = []
    entropies_acc : list[torch.Tensor] = []

    for log_probs_t, entropies_t, adv_scalar in zip(
        group_traj_log_probs, group_traj_entropies, group_advs
    ):
        adv      = adv_scalar.detach()
        adv_vec  = adv.expand_as(log_probs_t)

        # Pure REINFORCE score-function estimator.
        # (No old-policy ratio is available because we generate fresh on-policy
        # actions every call — ratio = 1 identically — so the clipped-surrogate
        # collapses to a plain REINFORCE term.  Using log_probs directly keeps the
        # gradient flowing correctly through the sampled actions.)
        policy_loss = -(adv_vec.detach() * log_probs_t).mean()
        entropy     = entropies_t.mean()
        traj_loss   = policy_loss - entropy_coef * entropy

        total_loss_terms.append(traj_loss)
        policy_losses.append(policy_loss.detach())
        entropies_acc.append(entropy.detach())

    if policy_optimizer is not None and total_loss_terms:
        total_loss = torch.stack(total_loss_terms).mean()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters()), max_grad_norm)
        policy_optimizer.step()

    def _mean_scalar(xs: list) -> float:
        if not xs:
            return 0.0
        return torch.stack(xs).mean().item()

    return {
        "policy_loss":          _mean_scalar(policy_losses),
        "entropy":              _mean_scalar(entropies_acc),
        "mean_imagined_return": float(np.mean(group_total_returns)),
        "group_adv_mean_abs":   group_advs.abs().mean().item(),
        "imagined_steps":       imagined_steps,
    }


def evaluate_policy(
    policy: TransformerPolicyWrapper,
    eval_env: "FastLIBEROEnv",
    cfg: DictConfig,
    device: torch.device,
    total_steps: int,
    log_dir: str,
) -> dict:
    """Run deterministic evaluation episodes and log metrics + a video to W&B.

    One video is captured from the episode with the **highest success rate**
    (ties broken by lowest episode index, i.e. the first episode wins).
    Metrics are averaged over all ``cfg.sim.eval_episodes`` episodes and
    returned as a plain dict (keys: ``eval/success_rate``,
    ``eval/avg_reward``, ``eval/avg_episode_length``).

    Args:
        policy:      The policy network to evaluate (set to eval mode internally).
        eval_env:    A ``FastLIBEROEnv`` instantiated with ``render_mode='rgb_array'``
                     so that ``render()`` returns frames.
        cfg:         Hydra config (uses ``cfg.sim.eval_episodes``,
                     ``cfg.sim.episode_length``).
        device:      Torch device.
        total_steps: Current training step count (used as W&B x-axis).
        log_dir:     Hydra output directory; videos are saved there before upload.

    Returns:
        dict with scalar metrics (all under the ``eval/`` prefix).
    """
    n_episodes    = int(cfg.sim.eval_episodes)
    max_ep_steps  = int(cfg.sim.episode_length)
    video_fps     = int(getattr(cfg.sim, "video_fps", 20))
    cam_name      = str(getattr(cfg.sim, "fast_env_image_camera", "agentview"))
    render_size   = int(getattr(cfg.sim, "fast_env_image_size", 256))

    ep_returns    : list[float]             = []
    ep_lengths    : list[int]               = []
    ep_successes  : list[float]             = []
    all_ep_frames : list[list[np.ndarray]]  = []   # one list of frames per episode

    was_training = policy.training
    policy.eval()

    use_pose = policy.model._cfg.policy.use_pose_data

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            obs = np.ascontiguousarray(obs)
            # Encode goal conditioning for this episode — explicit, no hidden state.
            txt_goal, goal_state = policy.encode_goals(obs, eval_env.instruction)
            pose = _extract_pose_from_info(info, policy, device) if use_pose else None
            ep_return = 0.0
            ep_length = 0
            success   = 0.0
            ep_frames : list[np.ndarray] = []

            for _ in range(max_ep_steps):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action_np, _, _, _ = policy.get_action(
                    obs_t, txt_goal, goal_state, pose, deterministic=True
                )

                obs, reward, done, truncated, info = eval_env.step(action_np)
                obs = np.ascontiguousarray(obs)
                pose = _extract_pose_from_info(info, policy, device) if use_pose else None

                ep_return += float(reward)
                ep_length += 1

                frame = eval_env.render(camera_name=cam_name, width=render_size, height=render_size)
                if frame is not None:
                    ep_frames.append(frame)

                if done or truncated:
                    success = float(info.get("success_placed", 0.0))
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            ep_successes.append(success)
            all_ep_frames.append(ep_frames)

    policy.train(was_training)

    # ---- Pick the best episode for video (highest success; first episode on tie) ----
    best_idx = int(np.argmax(ep_successes))   # argmax returns the *first* max index
    video_frames = all_ep_frames[best_idx]

    # ---- Scalar metrics ----
    metrics = {
        "eval/success_rate":        float(np.mean(ep_successes)),
        "eval/avg_reward":          float(np.mean(ep_returns)),
        "eval/avg_episode_length":  float(np.mean(ep_lengths)),
    }

    # ---- Video logging (best episode) ----
    if video_frames:
        # video_frames: list of (H, W, 3) uint8 arrays
        # wandb.Video expects (T, C, H, W) when passing a numpy array
        video_array = np.stack(video_frames, axis=0)          # (T, H, W, 3)
        video_array = video_array.transpose(0, 3, 1, 2)       # (T, 3, H, W)

        # Also save a local mp4 for reference
        video_dir = os.path.join(log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"eval_{total_steps:07d}.mp4")
        try:
            import imageio
            imageio.mimwrite(
                video_path,
                [f.transpose(1, 2, 0) for f in video_array],  # back to (H, W, C) per frame
                fps=video_fps,
                codec="libx264",
                quality=7,
            )
            metrics["eval/video"] = wandb.Video(video_path, fps=video_fps, format="mp4")
        except Exception as e:
            # imageio not available or codec missing — fall back to raw wandb.Video from array
            print(f"[eval] mp4 save failed ({e}); uploading raw frames to W&B.")
            metrics["eval/video"] = wandb.Video(video_array, fps=video_fps, format="gif")

    print(
        f"[eval @ {total_steps}] "
        f"success={metrics['eval/success_rate']:.2f}  "
        f"avg_reward={metrics['eval/avg_reward']:.3f}  "
        f"avg_ep_len={metrics['eval/avg_episode_length']:.1f}  "
        f"video=ep{best_idx} (success={ep_successes[best_idx]:.1f})"
    )
    return metrics

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="transformer_rl", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    log_dir = HydraConfig.get().runtime.output_dir

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    task_id = int(cfg.sim.eval_tasks[0])
    # Lazy import so this file can be imported without LIBERO installed (useful for unit tests).
    from hw3.libero_env_fast import FastLIBEROEnv
    env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg)
    eval_env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg)
    instruction = env.instruction
    print(f"Loaded environment with instruction: {instruction}")
    action_dim  = env._action_dim

    # Load transformer policy from HW1 checkpoint
    policy   = TransformerPolicyWrapper(cfg.init_checkpoint, device, cfg)

    # Value function: shared or separate model, same head architecture.
    shared_network = bool(cfg.value.get("shared_network", True))
    value_fn = ValueFunction(policy, device, cfg, shared_network=shared_network)
    print(f"Value network mode: {'shared model' if shared_network else 'separate model'}")

    if shared_network:
        # Shared model — deduplicate parameters so the GRP model is only
        # optimized once even though both policy and value_fn reference it.
        _seen, _params = set(), []
        for p in list(policy.parameters()) + list(value_fn.parameters()):
            if id(p) not in _seen:
                _seen.add(id(p))
                _params.append(p)
        optimizer = torch.optim.Adam(_params, lr=cfg.training.learning_rate)
    else:
        # Separate models — policy and value_fn own entirely distinct parameters;
        # no deduplication needed, but value critic may benefit from a higher LR.
        value_lr = float(cfg.value.get("learning_rate", cfg.training.learning_rate))
        optimizer = torch.optim.Adam([
            {"params": list(policy.parameters()),   "lr": cfg.training.learning_rate},
            {"params": list(value_fn.parameters()), "lr": value_lr},
        ])

    algorithm = cfg.rl.algorithm.lower()

    if algorithm == "ppo":
        # ------------------------------------------------------------------
        # PPO loop (reuses RolloutBuffer + ppo_update from Part 1)
        # ------------------------------------------------------------------
        obs, info = env.reset()
        obs    = np.ascontiguousarray(obs)
        if obs.ndim != 3:
            raise ValueError(f"Expected image obs (H,W,C), got shape={obs.shape}")

        use_pose = policy.model._cfg.policy.use_pose_data
        pose = _extract_pose_from_info(info, policy, device) if use_pose else None
        pose_dim = pose.shape[-1] if pose is not None else 7

        buffer = RolloutBuffer(cfg.training.rollout_length, obs.shape, action_dim, device, pose_dim=pose_dim)
        txt_goal, goal_state = policy.encode_goals(obs, instruction)

        obs_t = torch.from_numpy(obs).float().to(device)
        print(f"Observation shape: {obs.shape}")
        print(f"Pose shape: {pose.shape if pose is not None else 'N/A (pose disabled)'}")

        total_steps = 0
        episode_returns, episode_successes = [], []
        ep_ret = 0.0

        while total_steps < cfg.training.total_env_steps:
            buffer.reset()

            # --- Rollout collection ---
            policy.eval()
            value_fn.eval()
            with torch.no_grad():
                for _ in range(cfg.training.rollout_length):
                    action_np, log_prob, _, pre_tanh = policy.get_action(
                        obs_t.unsqueeze(0), txt_goal, goal_state, pose
                    )
                    value = value_fn(obs_t.unsqueeze(0), txt_goal, goal_state, pose)

                    next_obs, reward, done, truncated, info = env.step(action_np)
                    ep_ret      += reward
                    total_steps += 1
                    pose = _extract_pose_from_info(info, policy, device) if use_pose else None
                    buffer.add(
                        obs_t,
                        pre_tanh.to(device),
                        log_prob,
                        torch.tensor(reward,               device=device),
                        value.squeeze(0),
                        torch.tensor(float(done or truncated), device=device),
                        pose,
                        goal_state=goal_state,
                        txt_goal=txt_goal,
                    )

                    if done or truncated:
                        episode_returns.append(ep_ret)
                        episode_successes.append(float(info.get("success_placed", 0.0)))
                        ep_ret  = 0.0
                        obs, info = env.reset()
                        obs = np.ascontiguousarray(obs)   # always safe after reset
                        pose = _extract_pose_from_info(info, policy, device) if use_pose else None
                        txt_goal, goal_state = policy.encode_goals(obs, instruction)
                    else:
                        obs = next_obs

                    # Ensure numpy array is C-contiguous (some envs return views with
                    # negative strides) before zero-copy torch.from_numpy.
                    if not obs.flags["C_CONTIGUOUS"]:
                        obs = np.ascontiguousarray(obs)
                    obs_t = torch.from_numpy(obs).float().to(device)
                    if buffer.full():
                        break

                last_value = value_fn(obs_t.unsqueeze(0), txt_goal, goal_state, pose).squeeze(0)

            policy.train()
            value_fn.train()
            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, cfg.training.gamma, cfg.training.gae_lambda
            )

            # --- LR annealing ---
            if getattr(cfg.training, "anneal_lr", False):
                frac = 1.0 - total_steps / cfg.training.total_env_steps
                if shared_network:
                    for pg in optimizer.param_groups:
                        pg["lr"] = cfg.training.learning_rate * frac
                else:
                    base_lrs = [cfg.training.learning_rate,
                                float(cfg.value.get("learning_rate", cfg.training.learning_rate))]
                    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                        pg["lr"] = base_lr * frac

            update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

            # --- Logging ---
            if total_steps % cfg.log_interval < cfg.training.rollout_length:
                log = {"train/total_steps": total_steps,
                       **{f"train/{k}": v for k, v in update_info.items()}}
                if episode_returns:
                    log["train/episode_return"] = np.mean(episode_returns[-10:])
                    log["train/success_rate"]   = np.mean(episode_successes[-10:])
                wandb.log(log, step=total_steps)
                print(f"[PPO {total_steps}] return={log.get('train/episode_return', float('nan')):.3f} "
                      f"policy_loss={update_info['policy_loss']:.4f}")

            if total_steps % cfg.eval_interval < cfg.training.rollout_length:
                eval_metrics = evaluate_policy(
                    policy, eval_env, cfg, device, total_steps, log_dir
                )
                wandb.log(eval_metrics, step=total_steps)

            # --- Periodic checkpoint ---
            if total_steps % cfg.save_interval < cfg.training.rollout_length:
                ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "policy":      policy.model.state_dict(),
                    "value_fn":    value_fn.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "total_steps": total_steps,
                    "cfg":         OmegaConf.to_container(cfg),
                }, os.path.join(ckpt_dir, f"transformer_ppo_{total_steps}.pth"))

    elif algorithm == "grpo":
        # ------------------------------------------------------------------
        # GRPO loop with ground-truth resets (Part 2c)
        # ------------------------------------------------------------------
        from libero.libero import benchmark  # lazy import — only needed for GRPO
        total_steps  = 0
        update_count = 0
        all_returns  = []

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.sim.task_set]()
        init_states = task_suite.get_task_init_states(task_id)
        if len(init_states) == 0:
            raise RuntimeError(f"No init states found for task_id={task_id} in task_set={cfg.sim.task_set}")

        while total_steps < cfg.training.total_env_steps:
            trajectories_per_group = []
            steps_this_update = 0

            for _group_idx in range(int(cfg.grpo.num_groups)):
                init_state_idx = np.random.randint(len(init_states))
                init_state = init_states[init_state_idx]
                group = collect_grpo_group(
                    env=env,
                    policy=policy,
                    init_state=init_state,
                    group_size=int(cfg.grpo.group_size),
                    max_steps=int(cfg.sim.episode_length),
                    device=device,
                )
                trajectories_per_group.append(group)
                steps_this_update += sum(int(traj["rewards"].shape[0]) for traj in group)

            total_steps += steps_this_update

            update_info = grpo_update(policy, value_fn, optimizer,
                                      trajectories_per_group, cfg, device)
            update_count += 1
            all_returns.extend([t["total_return"] for g in trajectories_per_group for t in g])

            log = {
                "train/total_steps":    total_steps,
                "train/steps_this_update": steps_this_update,
                "train/update":         update_count,
                **{f"train/{k}": v for k, v in update_info.items()},
                "train/episode_return": np.mean(all_returns[-50:]) if all_returns else 0.0,
            }
            wandb.log(log, step=total_steps)
            print(f"[GRPO {total_steps}] return={log['train/episode_return']:.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

            if total_steps % cfg.eval_interval < max(1, steps_this_update):
                eval_metrics = evaluate_policy(
                    policy, eval_env, cfg, device, total_steps, log_dir
                )
                wandb.log(eval_metrics, step=total_steps)

            if total_steps % cfg.save_interval < max(1, steps_this_update):
                ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "policy":      policy.model.state_dict(),
                    "value_fn":    value_fn.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "total_steps": total_steps,
                    "cfg":         OmegaConf.to_container(cfg),
                }, os.path.join(ckpt_dir, f"transformer_grpo_{total_steps}.pth"))

    else:
        raise ValueError(f"Unknown rl.algorithm: {algorithm!r}. Choose 'ppo' or 'grpo'.")

    # --- Final checkpoint ---
    ckpt_dir = os.path.join("checkpoints", cfg.experiment.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "policy":   policy.model.state_dict(),
        "value_fn": value_fn.state_dict(),
        "cfg":      OmegaConf.to_container(cfg),
    }, os.path.join(ckpt_dir, "transformer_rl_final.pth"))

    env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
