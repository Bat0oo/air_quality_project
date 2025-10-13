import pandas as pd
import sys
sys.path.append('..')
from air_quality_project.database.db_manager import DatabaseManager
from air_quality_project import config
from datetime import datetime
import os
import glob

class CSVDataCollector:
    def __init__(self):
        self.db = DatabaseManager()
        self.data_folder = "air_quality_project/data/csv_files"  # Folder where CSV files are stored
        
        # Create data folder if it doesn't exist
        os.makedirs(self.data_folder, exist_ok=True)
    
    def load_csv_file(self, filepath, city_name):
        """Load and process a single CSV file"""
        print(f"\n📄 Processing: {os.path.basename(filepath)}")
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            print(f"   Loaded {len(df)} rows")
            print(f"   Columns: {', '.join(df.columns.tolist())}")
            
            # Convert DataFrame to documents
            documents = []
            
            for idx, row in df.iterrows():
                try:
                    doc = self.process_row(row, city_name, filepath)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    print(f"   ⚠️  Error processing row {idx}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            print(f"   ❌ Error loading CSV: {e}")
            return []
    
    def process_row(self, row, city_name, filepath):
        """Convert a CSV row to a database document"""
        try:
            # Parse timestamp from datetimeUtc column
            timestamp = None
            date_utc = None
            
            # Handle datetimeUtc column
            if 'datetimeUtc' in row and pd.notna(row['datetimeUtc']):
                date_utc = str(row['datetimeUtc'])
                try:
                    timestamp = pd.to_datetime(date_utc.replace('Z', '+00:00'))
                except Exception as e:
                    print(f"      ⚠️  Date parse error for {date_utc}: {e}")
            
            # Extract value
            value = None
            if 'value' in row and pd.notna(row['value']):
                value = float(row['value'])
            
            # Extract parameter
            parameter = None
            if 'parameter' in row and pd.notna(row['parameter']):
                parameter = str(row['parameter'])
            
            # Skip if missing essential data
            if value is None or parameter is None:
                return None
            
            # Build document matching your CSV structure
            doc = {
                "location": str(row.get('location_name', 'Unknown')),
                "location_id": int(row['location_id']) if 'location_id' in row and pd.notna(row['location_id']) else None,
                "city": city_name,
                "country": "Serbia",  # Since your CSV doesn't have country_iso populated
                "latitude": float(row['latitude']) if 'latitude' in row and pd.notna(row['latitude']) else None,
                "longitude": float(row['longitude']) if 'longitude' in row and pd.notna(row['longitude']) else None,
                "parameter": parameter,
                "value": value,
                "unit": str(row.get('unit', '')),
                "date_utc": date_utc,
                "timestamp": timestamp,
                "timezone": str(row.get('timezone', '')),
                "is_mobile": bool(row.get('isMobile', False)) if pd.notna(row.get('isMobile')) else False,
                "is_monitor": bool(row.get('isMonitor', False)) if pd.notna(row.get('isMonitor')) else False,
                "owner": str(row.get('owner_name', '')),
                "provider": str(row.get('provider', '')),
                "source_file": os.path.basename(filepath),
                "fetched_at": datetime.utcnow()
            }
            
            return doc
            
        except Exception as e:
            print(f"      ⚠️  Row processing error: {e}")
            return None
    
    def collect_from_csvs(self):
        """Main collection method - load all CSV files"""
        print("\n" + "="*70)
        print("📊 CSV DATA COLLECTION")
        print("="*70)
        print(f"\n📁 Looking for CSV files in: {self.data_folder}")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        
        if not csv_files:
            print(f"\n⚠️  No CSV files found in {self.data_folder}")
            print("\n💡 Instructions:")
            print("   1. Download CSV files from OpenAQ Explorer: https://explore.openaq.org")
            print("   2. Place CSV files in the 'data/csv_files' folder")
            print("   3. Name files like: belgrade.csv, novi_sad.csv, etc.")
            print("   4. Run this script again")
            return 0
        
        print(f"✅ Found {len(csv_files)} CSV file(s)")
        
        total_records = 0
        
        for csv_file in csv_files:
            # Extract city name from filename
            filename = os.path.basename(csv_file)
            city_name = os.path.splitext(filename)[0].replace('_', ' ').title()
            
            print(f"\n{'='*70}")
            print(f"📍 Processing: {city_name}")
            print(f"{'='*70}")
            
            documents = self.load_csv_file(csv_file, city_name)
            
            if documents:
                try:
                    # Insert into database
                    count = self.db.insert_many(config.COLLECTION_RAW, documents)
                    total_records += count
                    print(f"   ✅ Stored {count} measurements")
                except Exception as e:
                    print(f"   ❌ Database error: {e}")
            else:
                print(f"   ⚠️  No valid data to store")
        
        print("\n" + "="*70)
        print(f"✅ CSV COLLECTION COMPLETE!")
        print(f"📊 Total records collected: {total_records}")
        print("="*70)
        
        if total_records > 0:
            self.print_summary()
        
        return total_records
    
    def print_summary(self):
        """Print data collection summary"""
        print("\n📊 Data Summary:")
        print("-" * 70)
        
        total = self.db.count(config.COLLECTION_RAW)
        print(f"Total measurements in database: {total}")
        
        print("\nBy City:")
        pipeline = [
            {"$group": {"_id": "$city", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(self.db.aggregate(config.COLLECTION_RAW, pipeline))
        for r in results:
            print(f"  - {r['_id']}: {r['count']}")
        
        print("\nBy Parameter:")
        pipeline = [
            {"$match": {"parameter": {"$ne": None}}},
            {"$group": {"_id": "$parameter", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(self.db.aggregate(config.COLLECTION_RAW, pipeline))
        if results:
            for r in results:
                print(f"  - {r['_id']}: {r['count']}")
        
        # Date range
        print("\nDate Range:")
        pipeline = [
            {"$match": {"timestamp": {"$ne": None}}},
            {"$group": {
                "_id": None,
                "min_date": {"$min": "$timestamp"},
                "max_date": {"$max": "$timestamp"}
            }}
        ]
        results = list(self.db.aggregate(config.COLLECTION_RAW, pipeline))
        if results and len(results) > 0:
            print(f"  From: {results[0]['min_date']}")
            print(f"  To: {results[0]['max_date']}")
        
        # Sample data
        print("\nSample Data (first 3 records):")
        try:
            # Use find without limit parameter, then slice the list
            sample = list(self.db.find(config.COLLECTION_RAW))[:3]
            for i, record in enumerate(sample, 1):
                print(f"  {i}. {record.get('location')} - {record.get('parameter')}: "
                      f"{record.get('value')} {record.get('unit')} "
                      f"at {record.get('timestamp')}")
        except Exception as e:
            print(f"  Could not fetch sample data: {e}")
    
    def show_csv_structure(self, filepath):
        """Helper to show the structure of a CSV file"""
        print(f"\n📋 CSV Structure for: {os.path.basename(filepath)}")
        print("="*70)
        
        try:
            df = pd.read_csv(filepath, nrows=5)
            
            print(f"\nColumns ({len(df.columns)}):")
            for col in df.columns:
                print(f"  - {col}")
            
            print(f"\nFirst few rows:")
            print(df.to_string())
            
            print(f"\nData types:")
            print(df.dtypes)
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
    
    def close(self):
        self.db.close()

if __name__ == "__main__":
    import sys
    collector = CSVDataCollector()
    
    try:
        # Check if user wants to inspect a CSV file
        if len(sys.argv) > 1:
            if sys.argv[1] == "--inspect":
                # Show structure of CSV files
                csv_files = glob.glob(os.path.join(collector.data_folder, "*.csv"))
                if csv_files:
                    for csv_file in csv_files:
                        collector.show_csv_structure(csv_file)
                else:
                    print(f"No CSV files found in {collector.data_folder}")
            elif os.path.isfile(sys.argv[1]):
                # Inspect specific file
                collector.show_csv_structure(sys.argv[1])
        else:
            # Normal collection
            collector.collect_from_csvs()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.close()
        print("\n✅ CSV collection script completed")