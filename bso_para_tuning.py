import optuna
from ana_bso_multi import run_bso,customer_demands, depot_node_id, route_start_t, num_vehicles, vehicle_capacity 

def objective(trial):
    params = {
        "pop_size":  100,
        "max_iter":  100,
        "n_clusters": trial.suggest_int("n_clusters", 2, 10, step=1),
        "ideas_per_cluster":  trial.suggest_float("crossover_rate", 0.5, 1.0, step=0.1),
        "remove_rate":   trial.suggest_float("mutation_rate", 0.2, 1.0, step=0.1),}

    # Call BSO-LNS directly, passing params
    best_solution, best_cost, runtime = run_bso(
        **params,
        route_start_t=route_start_t,
        vehicle_capacity=vehicle_capacity,
        demands=customer_demands,
        depot_node_id=depot_node_id)
    
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
    study.optimize(objective, n_trials=300, timeout=None, n_jobs=4)

    print("Best parameters:", study.best_params)
    print("Best runtime   :", study.best_value, "seconds")
'''
    # --- Save results to Excel in the specified format ---
import os
import pandas as pd

def save_parameters(, runtime_seconds, , name="result"):
    """Save results to Excel in the specified format"""
    results_path = os.path.join(os.getcwd(), "output", f"{name}.xlsx")

    try:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if os.path.exists(results_path):
            df = pd.read_excel(results_path)
        else:
            df = pd.DataFrame()
        
        # Create new row
        new_data = {
            'route_start_t': route_start_t_label,
            'num_vehicles':num_vehicles,
            'test': len(df) + 1,
            'traveltime': cost,
            'runtime': runtime_seconds
        }
        
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_excel(results_path, index=False)
        print(f"✅ Results saved: {route_start_t_label}, Test {len(df)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
'''
if __name__ == "__main__":
    main()
