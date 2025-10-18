"""
Generate realistic demo CSV files for demonstration
Use this if you can't download from OpenAQ
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_pm25(base_value, hours, volatility=0.2):
    """Generate realistic PM2.5 values with temporal patterns"""
    values = []
    current = base_value
    
    for hour in range(hours):
        # Add daily pattern (higher in morning/evening)
        hour_of_day = hour % 24
        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
            daily_factor = 1.3  # Rush hour increase
        elif 22 <= hour_of_day or hour_of_day <= 6:
            daily_factor = 0.8  # Night decrease
        else:
            daily_factor = 1.0
        
        # Add random walk
        change = np.random.normal(0, volatility * base_value)
        current = max(5, current + change)  # Minimum 5 μg/m³
        
        # Apply daily pattern
        value = current * daily_factor
        
        # Add some noise
        value = max(0, value + np.random.normal(0, 2))
        
        values.append(value)
    
    return values

def create_demo_csv(city_name, base_pm25, num_records=100):
    """Create a demo CSV file for a city"""
    
    # Generate timestamps (last 100 hours)
    end_time = datetime.now()
    timestamps = [end_time - timedelta(hours=i) for i in range(num_records, 0, -1)]
    
    # Generate PM2.5 values
    pm25_values = generate_realistic_pm25(base_pm25, num_records)
    
    # Create DataFrame
    data = {
        'location_name': [f'{city_name} Monitoring Station' for _ in range(num_records)],
        'location_id': [np.random.randint(10000, 99999) for _ in range(num_records)],
        'latitude': [44.8125 if city_name == 'Beograd' else 
                     45.2671 if city_name == 'Novi Sad' else 
                     43.3209 for _ in range(num_records)],
        'longitude': [20.4612 if city_name == 'Beograd' else 
                      19.8335 if city_name == 'Novi Sad' else 
                      21.8954 for _ in range(num_records)],
        'parameter': ['pm25' for _ in range(num_records)],
        'value': pm25_values,
        'unit': ['µg/m³' for _ in range(num_records)],
        'datetimeUtc': [ts.strftime('%Y-%m-%dT%H:%M:%SZ') for ts in timestamps],
        'timezone': ['+01:00' for _ in range(num_records)],
        'isMobile': [False for _ in range(num_records)],
        'isMonitor': [True for _ in range(num_records)],
        'owner_name': [f'{city_name} Environmental Agency' for _ in range(num_records)],
        'provider': ['OpenAQ' for _ in range(num_records)]
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    """Generate demo CSVs for all cities"""
    
    print("\n" + "="*70)
    print("📝 GENERATING DEMO CSV FILES")
    print("="*70)
    
    # Create demo folder
    demo_folder = "air_quality_project/data/demo_csv"
    os.makedirs(demo_folder, exist_ok=True)
    
    # City configurations (city_name, typical_pm25_value)
    cities = [
        ('Beograd', 30),   # Typically moderate pollution
        ('Novi Sad', 25),   # Slightly less
        ('Nis', 35)         # Slightly more
    ]
    
    for city_name, base_pm25 in cities:
        print(f"\n📊 Generating data for {city_name}...")
        
        # Create CSV
        df = create_demo_csv(city_name, base_pm25, num_records=100)
        
        # Save
        filename = f"demo_{city_name.lower().replace(' ', '_')}.csv"
        filepath = os.path.join(demo_folder, filename)
        df.to_csv(filepath, index=False)
        
        print(f"   ✅ Created: {filepath}")
        print(f"   • Records: {len(df)}")
        print(f"   • Date range: {df['datetimeUtc'].iloc[0]} to {df['datetimeUtc'].iloc[-1]}")
        print(f"   • PM2.5 range: {df['value'].min():.1f} - {df['value'].max():.1f} μg/m³")
        print(f"   • Average: {df['value'].mean():.1f} μg/m³")
    
    print("\n" + "="*70)
    print("✅ DEMO CSV GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Files created in: {demo_folder}")
    print(f"\n💡 To use these files:")
    print(f"   1. Run: python demo_for_professor.py")
    print(f"   2. Follow the interactive demo steps")
    print(f"   3. The demo will automatically import these CSVs")
    
    print(f"\n⚠️  Note: These are SIMULATED data for demonstration purposes")
    print(f"   For real data, download from: https://explore.openaq.org/")

if __name__ == "__main__":
    main()