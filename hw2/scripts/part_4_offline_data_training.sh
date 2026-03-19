# python dreamer_model_trainer.py \
#     model_type=dreamer \
#     planner.type=policy \
#     planner.horizon=10 \
#     training.num_epochs=100 \
#     exp_name=q4_dreamer_policy \
#     experiment.name=q4_dreamer_policy \
#     use_policy=true
# To run hiddenly, we can use nohup and redirect output to a log file:
mkdir -p logs
nohup python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=policy \
    max_iters=500 \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=20 \
    policy.sequence_length=20 \
    planner.temperature=0.4 \
    sim.eval_tasks="[0]" \
    +load_world_model=./checkpoints/q3_dreamer_cem_H_15/world_model.pth \
    exp_name=q4_dreamer_policy \
    experiment.name=q4_dreamer_policy \
    use_policy=true \
    > logs/q4_dreamer_policy.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be
