import pandas as pd
import os

def analyze_csv(filepath):
    """Analyze what parameters are in a CSV file"""
    print(f"\n{'='*70}")
    print(f"📊 ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    try:
        # Read the CSV
        df = pd.read_csv(filepath)
        
        print(f"\n📈 Total rows: {len(df):,}")
        
        # Check parameters
        if 'parameter' in df.columns:
            params = df['parameter'].value_counts()
            print(f"\n✅ Parameters found ({len(params)}):")
            for param, count in params.items():
                print(f"   - {param}: {count:,} measurements")
            
            # Date range
            if 'datetimeUtc' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetimeUtc'])
                print(f"\n📅 Date range:")
                print(f"   From: {df['datetime'].min()}")
                print(f"   To: {df['datetime'].max()}")
            
            # Locations
            if 'location_name' in df.columns:
                locations = df['location_name'].unique()
                print(f"\n📍 Locations ({len(locations)}):")
                for loc in locations[:10]:  # Show first 10
                    count = len(df[df['location_name'] == loc])
                    print(f"   - {loc}: {count:,} measurements")
                if len(locations) > 10:
                    print(f"   ... and {len(locations)-10} more")
            
            # Check for missing values
            print(f"\n⚠️  Data Quality:")
            print(f"   Rows with missing values: {df['value'].isna().sum():,}")
            print(f"   Rows with valid values: {df['value'].notna().sum():,}")
            
        else:
            print("❌ No 'parameter' column found!")
            print(f"Available columns: {', '.join(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) > 1:
        # Analyze specific file
        analyze_csv(sys.argv[1])
    else:
        # Analyze all CSV files in data folder
        csv_files = glob.glob("air_quality_project/data/csv_files/*.csv")
        
        if not csv_files:
            print("⚠️  No CSV files found in data/csv_files/")
            print("\n💡 Usage:")
            print("   python csv_analyzer.py path/to/file.csv")
            print("   OR place CSV files in data/csv_files/ folder")
        else:
            for csv_file in csv_files:
                analyze_csv(csv_file)
                print()  # Extra line between files