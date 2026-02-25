# Train DreamerV3 world model
python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    training.num_epochs=100 \
    exp_name=q3_dreamer_cem \
    experiment.name=q3_dreamer_cem \
    use_policy=false