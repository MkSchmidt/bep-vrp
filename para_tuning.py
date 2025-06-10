import optuna
from ana_gen_multi import run_ga, period_breaks, demands_dict, depot_node_id, route_start_t, num_vehicles, vehicle_capacity

def objective(trial):
    params = {
        "pop_size":     trial.suggest_int("pop_size",     10, 100, step=5),
        "max_gens":     trial.suggest_int("max_gens",    50, 250, step=10),
        "tournament_size": trial.suggest_int("tournament_size", 2, 10, step=1),
        "crossover_rate":  trial.suggest_float("crossover_rate", 0.5, 1.0, step=0.1),
        "mutation_rate":   trial.suggest_float("mutation_rate", 0.0, 0.5, step=0.1),
        "elite_count":     trial.suggest_int("elite_count", 0, 10, step=1),
    }

    # Call your GA directly, passing params
    best_solution, best_cost, runtime = run_ga(
        
        route_start_t=route_start_t,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        period_breaks=period_breaks,
        demands_dict=demands_dict,
        depot_node_id=depot_node_id
    )
    return best_cost


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=4)

    print("Best params:", study.best_params)
    print("Best runtime (s):", study.best_value)

if __name__ == "__main__":
    main()
