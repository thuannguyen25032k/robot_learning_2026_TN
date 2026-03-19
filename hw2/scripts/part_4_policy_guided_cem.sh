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
    planner.type=policy_guided_cem \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=10 \
    policy.sequence_length=20 \
    sim.eval_tasks="[0, 0]" \
    sim.episode_length=200 \
    +load_world_model=./checkpoints/q4_dreamer_policy/world_model.pth \
    +load_policy=./checkpoints/q4_dreamer_policy/policy.pth \
    exp_name=q4_dreamer_policy_cem \
    experiment.name=q4_dreamer_policy_cem \
    use_policy=true \
    > logs/q4_dreamer_policy_cem.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be