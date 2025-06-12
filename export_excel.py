import os
import pandas as pd

def save_results(cost, runtime_seconds, num_vehicles, name="result"):
    """
    Saves results to an Excel file, correctly appending a new row for each run.
    """
    # Create the 'output' directory if it doesn't already exist
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the full path to the results file
    results_path = os.path.join(output_dir, f"{name}.xlsx")
    
    try:
        # 1. Read the existing file if it's there.
        if os.path.exists(results_path):
            df = pd.read_excel(results_path)
        # 2. If not, start with an empty DataFrame.
        else:
            df = pd.DataFrame()
        
        # 3. Create the new row of data.
        new_data = {
            'run_number': len(df) + 1,
            'traveltime_seconds': cost,
            'runtime_seconds': runtime_seconds,
            'num_vehicles': num_vehicles
        }

        new_row_df = pd.DataFrame([new_data])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        df.to_excel(results_path, index=False, engine='openpyxl')
        
        print(f"✅ Results for run {len(df)} saved to {results_path}")
        
    except Exception as e:
        print(f"❌ Error saving results to Excel: {e}")