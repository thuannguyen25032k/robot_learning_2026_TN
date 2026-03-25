export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p logs
nohup python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_grpo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=grpo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9] \
    > logs/hw3_transformer_grpo_seed0.log 2>&1 &