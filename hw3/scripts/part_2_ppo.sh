export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p logs
nohup python train_transformer_rl.py \
	experiment.name=hw3_transformer_ppo_seed0_reward_ver2 \
	r_seed=0 \
	init_checkpoint=../hw1/outputs/2026-01-27/14-55-19/miniGRP.pth \
	rl.algorithm=ppo \
	sim.task_set=libero_spatial \
	sim.eval_tasks=[9] \
	sim.reward_version='ver2' \
	training.total_env_steps=2000000 \
	training.rollout_length=2048 \
	training.ppo_epochs=10 \
	training.minibatch_size=256 \
	> logs/hw3_transformer_ppo_seed0_reward_ver2.log 2>&1 &

# mkdir -p logs
# nohup python train_transformer_rl.py \
# 	experiment.name=hw3_transformer_ppo_seed0 \
# 	r_seed=0 \
# 	init_checkpoint=../hw1/outputs/2026-01-27/14-55-19/miniGRP.pth \
# 	rl.algorithm=ppo \
# 	sim.task_set=libero_spatial \
# 	sim.eval_tasks=[9] \
# 	> logs/hw3_transformer_ppo_seed0.log 2>&1 &