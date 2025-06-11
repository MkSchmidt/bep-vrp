import optuna
from functools import partial
import pandas as pd

# Import the necessary functions and data from your main BSO script
from ana_bso_multi import run_bso, customer_demands, vehicle_capacity, start_time, num_vehicles

def objective(trial, problem_data):
    """
    The objective function for Optuna to minimize.
    It takes a trial object and a dictionary of fixed problem data.
    """
    # 1. Set a budget for total evaluations to ensure fair comparison
    TOTAL_EVALUATIONS_BUDGET = 20000  
    # 2. Define the search space for hyperparameters
    pop_size = trial.suggest_categorical('pop_size', [50, 100, 150, 200])
    # Calculate max_iter based on the budget
    max_iter = TOTAL_EVALUATIONS_BUDGET // pop_size
    params = {
        "pop_size": pop_size,
        "max_iter": max_iter,
        "n_clusters": trial.suggest_int("n_clusters", 2, 8),
        "ideas_per_cluster": trial.suggest_int("ideas_per_cluster", 1, 10),
        "remove_rate": trial.suggest_float("remove_rate", 0.1, 0.6, step=0.05),
    }

    # 3. Run the algorithm multiple times and average the results for robustness
    num_runs_for_averaging = 3
    all_costs = []
    for _ in range(num_runs_for_averaging):
        _, best_cost, _ = run_bso(
            **params,
            **problem_data
        )
        all_costs.append(best_cost)
    
    # Return the average cost for this set of hyperparameters
    return sum(all_costs) / len(all_costs)

def main():
    # 4. Package all fixed problem data into a dictionary
    problem_data = {
        'start_time': start_time,
        'vehicle_capacity': vehicle_capacity,
        'demands': customer_demands,
    }

    # Use functools.partial to create a new function that has problem_data pre-filled
    objective_with_data = partial(objective, problem_data=problem_data)

    # Use the TPE sampler for efficient searching
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # A pruner can stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="bso_lns_tuning"
    )

    # Run the optimization
    study.optimize(objective_with_data, n_trials=100, n_jobs=1)

    # 5. Provide a detailed report at the end
    print("\n" + "="*30)
    print("--- OPTIMIZATION COMPLETE ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Average Cost): {trial.value:.2f}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save all results to a CSV file for analysis
    results_df = study.trials_dataframe()
    results_df.to_csv("bso_tuning_full_results.csv", index=False)
    print("\n✅ Full tuning results saved to 'bso_tuning_full_results.csv'")


if __name__ == "__main__":
    main()