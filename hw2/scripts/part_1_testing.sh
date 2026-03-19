mkdir -p logs
nohup python dreamer_model_tester.py \
    model_type=simple \
    planner.type=cem \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=25 \
    planner.temperature=0.3 \
    +load_world_model=./checkpoints/q1_simple_cem_H_10/world_model.pth \
    exp_name=q1_simple_cem_test \
    experiment.name=q1_simple_cem_test \
    use_policy=false \
    > logs/q1_simple_cem_test.log 2>&1 &