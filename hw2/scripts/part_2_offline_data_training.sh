python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=policy \
    planner.horizon=10 \
    training.num_epochs=50 \
    exp_name=q2_policy_training \
    experiment.name=q2_policy_training \
    use_policy=true

# To run hiddenly, we can use nohup and redirect output to a log file:
# mkdir -p logs
# nohup python dreamer_model_trainer.py \
#     model_type=simple \
#     planner.type=policy \
#     planner.horizon=10 \
#     training.num_epochs=50 \
#     exp_name=q2_policy_training \
#     experiment.name=q2_policy_training \
#     use_policy=true \
#     > logs/q2_policy_training.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be detached and reattached later.