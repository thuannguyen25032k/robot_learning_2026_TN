mkdir -p logs
nohup python dreamer_model_tester.py \
    model_type=simple \
    planner.type=cem \
    planner.horizon=15 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    planner.num_iterations=25 \
    planner.temperature=0.3 \
    sim.eval_tasks="[0,0,0]" \
    +load_world_model=./checkpoints/q1_simple_cem_H_15/model_epoch_3251_batch_432.pth \
    exp_name=q1_simple_cem_H_15_test \
    experiment.name=q1_simple_cem_H_15_test \
    use_policy=false \
    > logs/q1_simple_cem_H_15_test.log 2>&1 &