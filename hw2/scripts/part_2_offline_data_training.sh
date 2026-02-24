python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=policy \
    planner.horizon=10 \
    training.num_epochs=50 \
    exp_name=q2_policy_training \
    experiment.name=q2_policy_training \
    use_policy=true