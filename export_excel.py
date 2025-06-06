# --- Save results to Excel in the specified format ---
import os
import pandas as pd

def save_results(cost, runtime_seconds, route_start_t, num_vehicles, name="result"):
    """Save results to Excel in the specified format"""
    results_path = os.path.join(os.getcwd(), "output", f"{name}.xlsx")
    
    # Convert sim_start to hours:minutes format
    hours = route_start_t // 3600
    minutes = (route_start_t % 3600) // 60
    route_start_t_label = f"route_start_t = {hours}:{minutes:02d}"

    
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