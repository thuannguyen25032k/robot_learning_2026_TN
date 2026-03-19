mkdir -p logs
nohup python dreamer_model_tester.py \
    model_type=dreamer \
    planner.type=policy_guided_cem \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=25 \
    planner.temperature=0.3 \
    +load_world_model=./checkpoints/q4_dreamer_policy_cem/world_model.pth \
    +load_policy=./checkpoints/q4_dreamer_policy_cem/policy.pth \
    exp_name=q4_dreamerv3_cem_test \
    experiment.name=q4_dreamerv3_cem_test \
    use_policy=true \
    > logs/q4_dreamerv3_cem_test.log 2>&1 &