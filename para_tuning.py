import optuna
from ana_gen_multi import run_ga, period_breaks, demands_dict, depot_node_id, route_start_t, num_vehicles, vehicle_capacity

def objective(trial):
    params = {
        "pop_size":     trial.suggest_int("pop_size", 100, 110, step=10),
        "max_gens":     trial.suggest_int("max_gens", 100,110, step=10),
        "tournament_size": trial.suggest_int("tournament_size", 2, 7, step=1),
        "crossover_rate":  trial.suggest_float("crossover_rate", 0.7, 1.0, step=0.1),
        "mutation_rate":   trial.suggest_float("mutation_rate", 0.2, 0.5, step=0.1),
        "elite_count":     trial.suggest_int("elite_count", 2, 7, step=1),}

    # Call your GA directly, passing params
    best_solution, best_cost, runtime = run_ga(
        **params,
        route_start_t=route_start_t,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        period_breaks=period_breaks,
        demands_dict=demands_dict,
        depot_node_id=depot_node_id
    )
    return best_cost


def main():
    # Use the TPE sampler (Tree-structured Parzen Estimator)
    sampler = optuna.samplers.TPESampler(seed=42)
    # A pruner to stop unpromising trials early
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=15)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="ga_runtime_tuning"
    )

    # You can limit by number of trials or wall-clock time
    study.optimize(objective, n_trials=100, timeout=None, n_jobs=4)

    print("Best parameters:", study.best_params)
    print("Best runtime   :", study.best_value, "seconds")

if __name__ == "__main__":
    main()
