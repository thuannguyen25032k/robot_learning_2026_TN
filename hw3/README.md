# HW3 — Making a Generalist Robotics Policy with Reinforcement Learning

## 1. Code Organisation

| File | Role |
|------|------|
| `train_dense_rl.py` | **Part 1 — Dense PPO.** Defines `DensePolicy` (MLP actor-critic), `DenseValueFunction`, `RolloutBuffer`, and the full `ppo_update` (rollout collection → GAE → clipped surrogate + value loss). Also contains `evaluate_policy` for the dense setting and the Hydra `main` entry-point. |
| `train_transformer_rl.py` | **Part 2 — Transformer RL.** Defines `TransformerPolicyWrapper` (wraps the HW1 GRP model), `ValueFunction` (separate MLP head), `collect_grpo_group`, `grpo_update` (GRPO with ground-truth resets, Part 2c), `grpo_worldmodel_update` (GRPO via DreamerV3 world model, Part 2d), and PPO fine-tuning logic. Contains `evaluate_policy` for the transformer and the Hydra `main` that dispatches on `rl.algorithm ∈ {ppo, grpo}`. |
| `train_dagger.py` | **Part 3 (Optional) — DAgger.** Defines `DensePolicyTeacher` (loads the Part 1 checkpoint), `DAggerDataset` (aggregated rollout storage), `collect_dagger_rollout` (beta-mixed teacher/student rollouts), and `bc_update` (NLL behavioural cloning on aggregated data). |
| `train_dense_rl.py` → `evaluate_policy` | **Evaluation — dense policy.** Runs `cfg.sim.eval_episodes` rollouts on `FastLIBEROEnv`, logs success rate and mean return to WandB, and optionally saves MP4 videos. |
| `train_transformer_rl.py` → `evaluate_policy` | **Evaluation — transformer policy.** Same interface; used by both `train_transformer_rl.py` and `train_dagger.py`. |
| `sim_eval.py` | **Standalone evaluation / HW1-compatible loader.** Loads a raw `miniGRP.pth` pickle (as produced by HW1 training), and can run `libero`, `libero_fast`, or `simple_env` evaluation loops. Use this for any checkpoint that uses the original HW1 file format. |
| `libero_env_fast.py` | Vectorised `FastLIBEROEnv` wrapper around LIBERO used by all three trainers. |
| `grp_model.py` | HW3-local copy of the GRP transformer architecture (mirrors `mini-grp/grp_model.py`). |
| `networks.py` | Shared network utilities (encoders, MLPs). |
| `conf/dense_ppo.yaml` | Hydra config for Part 1. |
| `conf/transformer_rl.yaml` | Hydra config for Part 2 (PPO and GRPO). |
| `conf/dagger.yaml` | Hydra config for Part 3. |

---

## 2. Part 1 — Dense PPO

### Main experiment commands

All runs target **LIBERO-Spatial task 9** and are launched from the `hw3/` directory.

**Seed 0** (original template command from `hw3.md`):
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python train_dense_rl.py \
    experiment.name=hw3_dense_ppo_seed0 \
    r_seed=0 \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    training.total_env_steps=200000 \
    training.rollout_length=128 \
    training.ppo_epochs=10 \
    training.minibatch_size=256
```

**Seed 2** (main run used in the report — 2 M steps, larger rollout buffer):
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p logs
nohup python train_dense_rl.py \
    experiment.name=hw3_dense_ppo_seed2 \
    r_seed=2 \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.reward_scale=0.1 \
    training.total_env_steps=2000000 \
    training.rollout_length=4096 \
    training.ppo_epochs=10 \
    training.minibatch_size=256 \
    training.learning_rate=3e-4 \
    > logs/hw3_dense_ppo_seed2.log 2>&1 &
```

### Seed table

| Experiment name | `r_seed` |
|-----------------|----------|
| `hw3_dense_ppo_seed0` | 0 |
| `hw3_dense_ppo_seed1` | 1 |
| `hw3_dense_ppo_seed2` | 2 |

### Checkpoint loading

Checkpoints are saved by Hydra to
`outputs/<date>/<time>/checkpoints/dense_ppo_step<N>.pth`.
Each file contains the keys `policy`, `value`, `optimizer`, `step`, and `cfg`.

To evaluate a saved checkpoint with the standalone evaluator:
```bash
# from the repo root
python hw3/sim_eval.py \
    checkpoint=/path/to/dense_ppo_step200000.pth \
    simEval=[libero_fast] \
    sim.eval_tasks=[9]
```

---

## 3. Part 2 — Transformer RL Fine-tuning

All runs initialise from the HW1 GRP checkpoint and target **LIBERO-Spatial task 0**.

### (a) PPO fine-tuning

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p logs
nohup python train_transformer_rl.py \
    experiment.name=hw3_transformer_ppo_seed0 \
    r_seed=0 \
    init_checkpoint="checkpoints/grp-ver2/miniGRP.pth" \
    rl.algorithm=ppo \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[0] \
    sim.reward_scale=0.1 \
    training.total_env_steps=2000000 \
    training.rollout_length=4096 \
    training.ppo_epochs=10 \
    training.minibatch_size=256 \
    > logs/hw3_transformer_ppo_seed0.log 2>&1 &
```

### (c) GRPO with ground-truth resets

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p logs
nohup python hw3/train_transformer_rl.py \
    experiment.name=hw3_transformer_grpo_seed0 \
    r_seed=0 \
    init_checkpoint="checkpoints/grp-ver2/miniGRP.pth" \
    rl.algorithm=grpo \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[0] \
    sim.reward_scale=0.1 \
    training.total_env_steps=2000000 \
    training.rollout_length=4096 \
    training.ppo_epochs=10 \
    training.minibatch_size=256 \
    > logs/hw3_transformer_grpo_seed0.log 2>&1 &
```

### Seed table

| Experiment name | `r_seed` | Algorithm |
|-----------------|----------|-----------|
| `hw3_transformer_ppo_seed0` | 0 | PPO |
| `hw3_transformer_grpo_seed0` | 0 | GRPO (ground-truth reset) |

### Key config defaults (`conf/transformer_rl.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sim.reward_scale` | `1.0` | Set to `0.1` in launch scripts above (override as needed) |
| `training.entropy_coeff` | `0.01` | Prevents immediate entropy collapse |
| `training.clip_eps` | `0.2` | PPO clipping |
| `grpo.group_size` | `8` | Trajectories per initial state |
| `grpo.num_groups` | `16` | Initial states per update |
| `value.shared_network` | `false` | Separate value MLP, not a shared head |

### Checkpoint loading

Transformer checkpoints are saved to
`outputs/<date>/<time>/checkpoints/transformer_rl_step<N>.pth`
and contain the keys `policy`, `value`, `optimizer`, `value_optimizer`, `step`, and `cfg`.

To evaluate:
```bash
python hw3/sim_eval.py \
    checkpoint=/path/to/transformer_rl_step2000000.pth \
    simEval=[libero_fast] \
    sim.eval_tasks=[0]
```

---

## 4. Part 3 (Optional) — DAgger

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python hw3/train_dagger.py \
    experiment.name=hw3_dagger_seed0 \
    r_seed=0 \
    teacher_checkpoint=/path/to/dense_policy.pth \
    student_init_checkpoint=/path/to/hw1/miniGRP.pth \
    dagger.num_rounds=10 \
    dagger.rollouts_per_round=20
```

### Seed table

| Experiment name | `r_seed` |
|-----------------|----------|
| `hw3_dagger_seed0` | 0 |

### How DAgger works in this codebase

1. `collect_dagger_rollout` runs one episode: with probability `beta` the teacher drives; with probability `1 − beta` the student drives. Every visited state is labelled by the teacher.
2. `DAggerDataset.add_rollout` appends `(image_obs, teacher_action)` pairs to a growing on-disk dataset.
3. `bc_update` runs `dagger.bc_epochs_per_round` supervised epochs over the full aggregated dataset using negative log-likelihood in the student's z-score action space.
4. `beta` decays linearly from `beta_init=1.0` (pure teacher) to `0.0` (pure student) over `num_rounds` rounds.
5. Intermediate datasets are saved to `dagger.dataset_save_dir/dagger_round_NNN.pth`.
6. After all rounds the final student checkpoint is saved to `<hydra_output_dir>/checkpoints/dagger_student_final.pth`.

To continue training the DAgger-initialised student with RL (Part 3 → Part 2 pipeline):
```bash
python hw3/train_transformer_rl.py \
    experiment.name=hw3_dagger_then_ppo_seed0 \
    r_seed=0 \
    init_checkpoint=<hydra_output_dir>/checkpoints/dagger_student_final.pth \
    rl.algorithm=ppo \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[0]
```

---

## 5. Standalone Evaluation (`sim_eval.py`)

`hw3/sim_eval.py` accepts any checkpoint whose format is a raw `torch.save` of the GRP model object (HW1 style) **or** a dict with a `policy` key (HW3 PPO/DAgger style, handled via the `TransformerPolicyWrapper.load` path). The default config is `conf/64pix-pose.yaml`.

```bash
# HW1-style raw pickle
python hw3/sim_eval.py \
    simEval=[libero_fast] \
    sim.eval_tasks=[9]   # loads mini-grp/miniGRP.pth by default

# HW3-style checkpoint (adapt sim_eval.py checkpoint= override)
python hw3/sim_eval.py \
    checkpoint=/path/to/checkpoint.pth \
    simEval=[libero_fast] \
    sim.eval_tasks=[9]
```
