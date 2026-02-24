# Train SimpleWorldModel and evaluate with CEM planning
python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=5 \
    training.num_epochs=50 \
    exp_name=q1_simple_cem \
    experiment.name=q1_simple_cem \
    use_policy=false 

# To run hiddenly, we can use nohup and redirect output to a log file:
# nohup python dreamer_model_trainer.py \
#     model_type=simple \
#     planner.type=cem \
#     planner.horizon=10 \
#     planner.num_samples=100 \
#     planner.num_elites=10 \
#     planner.num_iterations=5 \
#     training.num_epochs=50 \
#     exp_name=q1_simple_cem \
#     experiment.name=q1_simple_cem \
#     use_policy=false \
#     > q1_simple_cem.log 2>&1 &
# or use tmux/screen to run the command in a separate terminal session that can be detached and reattached later.
