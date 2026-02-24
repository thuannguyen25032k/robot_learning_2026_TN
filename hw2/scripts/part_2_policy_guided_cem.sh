# Then, use policy-guided CEM planning
python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=policy_guided_cem \
    planner.horizon=10 \
    planner.num_samples=50 \
    planner.num_elites=5 \
    +load_policy=/checkpoints/q2_policy_training/policy.pth \
    exp_name=q2_policy_cem \
    experiment.name=q2_policy_cem \
    use_policy=true

# To run hiddenly, we can use nohup and redirect output to a log file:
# nohup python dreamer_model_trainer.py \
#     model_type=simple \
#     planner.type=policy_guided_cem \
#     planner.horizon=10 \
#     planner.num_samples=50 \
#     planner.num_elites=5 \
#     load_policy=/checkpoints/q2_policy_training/policy.pth \
#     exp_name=q2_policy_cem \
#     experiment.name=q2_policy_cem \
#     use_policy=true \
#     > /logs/q2_policy_cem.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be detached and reattached later.