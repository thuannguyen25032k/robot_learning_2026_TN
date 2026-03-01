# python dreamer_model_trainer.py \
#     model_type=simple \
#     planner.type=policy \
#     planner.horizon=10 \
#     training.num_epochs=50 \
#     exp_name=q2_policy_training \
#     experiment.name=q2_policy_training \
#     use_policy=true

# To run hiddenly, we can use nohup and redirect output to a log file:
mkdir -p logs
nohup python dreamer_model_trainer.py \
    model_type=simple \
    max_iters=50 \
    planner.type=policy \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=20 \
    policy.sequence_length=20 \
    planner.temperature=0.4 \
    sim.eval_tasks="[0]" \
    +load_world_model=./checkpoints/q1_simple_cem_H_15/world_model.pth \
    exp_name=q2_policy_training_default \
    experiment.name=q2_policy_training_default \
    use_policy=true \
    > logs/q2_policy_training_default.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be detached and reattached later.