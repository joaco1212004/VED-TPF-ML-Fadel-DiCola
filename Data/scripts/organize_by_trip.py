import os
import pandas as pd
from pathlib import Path

def organize_trips_by_vehid():
    """
    Separates each week's dynamic data by trip and organizes into VehId folders.
    Structure: VED_DynamicData/VehId_{vehid}/Trip_{trip_id}_{week}.csv
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_folder = os.path.join(base_path, "VED_DynamicData")
    
    if not os.path.exists(source_folder):
        print(f"Source folder not found: {source_folder}")
        return
    
    total_trips_created = 0
    
    # Process each week CSV file
    for filename in sorted(os.listdir(source_folder)):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(source_folder, filename)
        
        try:
            print(f"\nProcessing {filename}...")
            df = pd.read_csv(file_path)
            
            if 'VehId' not in df.columns or 'Trip' not in df.columns:
                print(f"  Missing VehId or Trip column, skipping...")
                continue
            
            # Group by VehId and Trip
            for vehid, veh_group in df.groupby('VehId'):
                # Create VehId folder
                vehid_folder = os.path.join(source_folder, f"VehId_{int(vehid)}")
                os.makedirs(vehid_folder, exist_ok=True)
                
                # Separate by trip
                for trip_id, trip_data in veh_group.groupby('Trip'):
                    # Create filename: Trip_{trip_id}_{week}.csv
                    week_name = filename.replace('.csv', '')
                    trip_filename = f"Trip_{int(trip_id)}_{week_name}.csv"
                    trip_path = os.path.join(vehid_folder, trip_filename)
                    
                    # Save trip data
                    trip_data.to_csv(trip_path, index=False)
                    total_trips_created += 1
            
            print(f"  ✓ Extracted trips from {filename}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
    
    print(f"\n✓ Complete! Created {total_trips_created} trip CSV files")
    print(f"  Location: VED_DynamicData/VehId_*/Trip_*.csv")

if __name__ == "__main__":
    organize_trips_by_vehid()
