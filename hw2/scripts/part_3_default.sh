# Train DreamerV3 world model
# python dreamer_model_trainer.py \
#     model_type=dreamer \
#     planner.type=cem \
#     planner.horizon=10 \
#     planner.num_samples=100 \
#     training.num_epochs=100 \
#     exp_name=q3_dreamer_cem \
#     experiment.name=q3_dreamer_cem \
#     use_policy=false

# To run hiddenly, we can use nohup and redirect output to a log file:
mkdir -p logs
nohup python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    training.num_epochs=100 \
    exp_name=q3_dreamer_cem \
    experiment.name=q3_dreamer_cem \
    use_policy=false \
    > logs/q3_dreamer_cem.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be detached and reattached later.