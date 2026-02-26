python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=policy \
    planner.horizon=10 \
    training.num_epochs=100 \
    exp_name=q4_dreamer_policy \
    experiment.name=q4_dreamer_policy \
    use_policy=true
# To run hiddenly, we can use nohup and redirect output to a log file:
# mkdir -p logs
# nohup python dreamer_model_trainer.py \
#     model_type=dreamer \
#     planner.type=policy \
#     planner.horizon=10 \
#     training.num_epochs=100 \
#     exp_name=q4_dreamer_policy \
#     experiment.name=q4_dreamer_policy \
#     use_policy=true \
#     > logs/q4_dreamer_policy.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be