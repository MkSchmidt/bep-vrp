import optuna
from ana_gen_multi import run_ga, period_breaks, demands_dict, depot_node_id, route_start_t, num_vehicles, vehicle_capacity
from functools import partial

def objective(trial, problem_data):
    TOTAL_EVALUATIONS_BUDGET = 20000
    pop_size = trial.suggest_categorical('pop_size', [50, 100, 200, 400])

    params = {
        "pop_size": pop_size,
        "tournament_size": trial.suggest_int("tournament_size", 2, 10, step=1),
        "crossover_rate":  trial.suggest_float("crossover_rate", 0.5, 1.0, step=0.1),
        "mutation_rate":   trial.suggest_float("mutation_rate", 0.01, 0.2, step=0.1),
        "elite_count": trial.suggest_int("elite_count", 2, max(2, int(pop_size * 0.15)))}
    
    max_gens = TOTAL_EVALUATIONS_BUDGET // params['pop_size']
    params['max_gens'] = max_gens

    num_runs_for_averaging = 3
    best_costs = []
    for i in range(num_runs_for_averaging):
        return_value_from_ga = run_ga(
            **params,
            **problem_data)

        if isinstance(return_value_from_ga, tuple):
            best_cost = return_value_from_ga[1] 
        else:
            best_cost = return_value_from_ga 
        
        best_costs.append(best_cost)
    
    return sum(best_costs) / len(best_costs)

def main():
    problem_data = {
        'route_start_t': route_start_t,
        'num_vehicles': num_vehicles,
        'vehicle_capacity': vehicle_capacity,
        'period_breaks': period_breaks,
        'demands_dict': demands_dict,
        'depot_node_id': depot_node_id}

    objective_with_data = partial(objective, problem_data=problem_data)

    # Use the TPE sampler with a seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        sampler=sampler,
        direction="minimize",
        study_name="ga_traveltime_tuning"
    )

    study.optimize(objective_with_data, n_trials=100, n_jobs=1)

    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Average Travel Time): {trial.value}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save all results to a CSV file
    results_df = study.trials_dataframe()
    results_df.to_csv("ga_tuning_full_results.csv", index=False)
    print("\n✅ Full tuning results saved to 'ga_tuning_full_results.csv'")

if __name__ == "__main__":
    main()
