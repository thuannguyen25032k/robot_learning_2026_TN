# Then, use policy-guided CEM planning
# python dreamer_model_trainer.py \
#     model_type=simple \
#     planner.type=policy_guided_cem \
#     planner.horizon=10 \
#     planner.num_samples=50 \
#     planner.num_elites=5 \
#     +load_policy=checkpoints/q2_policy_training/policy.pth \
#     +load_world_model=checkpoints/q1_world_model_training/world_model.pth \
#     exp_name=q2_policy_cem \
#     experiment.name=q2_policy_cem \
#     use_policy=true

# To run hiddenly, we can use nohup and redirect output to a log file:
mkdir -p logs
nohup python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=policy_guided_cem \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=10 \
    policy.sequence_length=20 \
    sim.eval_tasks="[0, 0]" \
    sim.episode_length=200 \
    +load_world_model=./checkpoints/q1_simple_cem_H_15/world_model.pth \
    +load_policy=./checkpoints/q2_policy_training_default/policy.pth \
    exp_name=q2_policy_cem \
    experiment.name=q2_policy_cem \
    use_policy=true \
    > logs/q2_policy_cem.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be detached and reattached later.