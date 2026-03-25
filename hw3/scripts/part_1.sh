export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# python train_dense_rl.py \
# 	experiment.name=hw3_dense_ppo_seed0 \
# 	r_seed=0 \
# 	sim.task_set=libero_spatial \
# 	sim.eval_tasks=[9] \
# 	training.total_env_steps=200000 \
# 	training.rollout_length=128 \
# 	training.ppo_epochs=10 \
# 	training.minibatch_size=1024

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
