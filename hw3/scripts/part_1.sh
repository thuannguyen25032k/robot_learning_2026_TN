export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p logs
nohup python train_dense_rl.py \
	experiment.name=hw3_dense_ppo_seed0 \
	r_seed=0 \
	sim.task_set=libero_spatial \
	sim.eval_tasks=[9] \
	training.total_env_steps=200000 \
	training.rollout_length=128 \
	training.ppo_epochs=10 \
	training.minibatch_size=1024 \
    > logs/hw3_dense_ppo_seed0.log 2>&1 &
