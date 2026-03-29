export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p logs
nohup python train_transformer_rl.py \
	experiment.name=hw3_transformer_ppo_seed0\
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

