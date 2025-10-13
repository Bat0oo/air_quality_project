import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from air_quality_project.database.db_manager import DatabaseManager
from air_quality_project import config
import datetime as dt

class DataProcessor:
    def __init__(self):
        self.db = DatabaseManager()
    
    def load_raw_data(self):
        """Load raw data from MongoDB into DataFrame"""
        print("\n📥 Loading raw data from MongoDB...")
        data = self.db.find(config.COLLECTION_RAW)
        
        if not data:
            print("❌ No raw data found!")
            return None
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} records")
        return df
    
    def clean_data(self, df):
        """Clean and prepare data"""
        print("\n🔧 Cleaning data...")
        
        initial_count = len(df)
        
        # Remove nulls
        df = df.dropna(subset=['value', 'timestamp', 'city'])
        
        # Remove negative values
        df = df[df['value'] >= 0]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['date'] = df['timestamp'].dt.date
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        
        removed = initial_count - len(df)
        print(f"✅ Cleaned: {len(df)} records ({removed} removed)")
        
        return df
    
    def calculate_aqi(self, df):
        """Calculate Air Quality Index for PM2.5"""
        print("\n📊 Calculating AQI...")
        
        def pm25_to_aqi(pm25):
            """Convert PM2.5 to AQI (simplified)"""
            if pd.isna(pm25):
                return None
            if pm25 <= 12:
                return pm25 * 50 / 12
            elif pm25 <= 35.4:
                return 50 + (pm25 - 12) * 50 / 23.4
            elif pm25 <= 55.4:
                return 100 + (pm25 - 35.4) * 50 / 20
            elif pm25 <= 150.4:
                return 150 + (pm25 - 55.4) * 100 / 95
            elif pm25 <= 250.4:
                return 200 + (pm25 - 150.4) * 100 / 100
            else:
                return 300 + (pm25 - 250.4) * 200 / 149.6
        
        def aqi_category(aqi):
            """Get AQI category"""
            if pd.isna(aqi):
                return None
            if aqi <= 50:
                return "Good"
            elif aqi <= 100:
                return "Moderate"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                return "Unhealthy"
            elif aqi <= 300:
                return "Very Unhealthy"
            else:
                return "Hazardous"
        
        # Calculate AQI only for PM2.5
        df['aqi_score'] = df.apply(
            lambda row: pm25_to_aqi(row['value']) if row['parameter'] == 'pm25' else None,
            axis=1
        )
        
        df['aqi_category'] = df['aqi_score'].apply(aqi_category)
        
        print("✅ AQI calculated")
        return df
    
    def aggregate_daily(self, df):
        """Aggregate data by day"""
        print("\n📅 Aggregating daily statistics...")
        
        daily = df.groupby(['city', 'date', 'parameter']).agg({
            'value': ['mean', 'min', 'max', 'std', 'count']
        }).reset_index()
        
        daily.columns = ['city', 'date', 'parameter', 'avg_value', 'min_value', 
                        'max_value', 'std_value', 'count']
        
        print(f"✅ Generated {len(daily)} daily aggregations")
        return daily
    
    def aggregate_hourly(self, df):
        """Aggregate data by hour"""
        print("\n⏰ Aggregating hourly statistics...")
        
        hourly = df.groupby(['city', 'date', 'hour', 'parameter']).agg({
            'value': ['mean', 'min', 'max', 'count']
        }).reset_index()
        
        hourly.columns = ['city', 'date', 'hour', 'parameter', 
                         'avg_value', 'min_value', 'max_value', 'count']
        
        print(f"✅ Generated {len(hourly)} hourly aggregations")
        return hourly
    
    def city_comparison(self, df):
        """Compare air quality across cities"""
        print("\n🏙️  Generating city comparison...")
        
        comparison = df.groupby(['city', 'parameter']).agg({
            'value': ['mean', 'min', 'max', 'std']
        }).reset_index()
        
        comparison.columns = ['city', 'parameter', 'avg_value', 
                             'min_value', 'max_value', 'std_value']
        
        print("\n📊 City Comparison:")
        print(comparison.to_string())
        
        return comparison
    
    def temporal_patterns(self, df):
        """Analyze temporal patterns"""
        print("\n📈 Analyzing temporal patterns...")
        
        # Hourly patterns
        hourly_pattern = df.groupby(['hour', 'parameter'])['value'].mean().reset_index()
        hourly_pattern.columns = ['hour', 'parameter', 'avg_value']
        
        # Weekly patterns
        weekly_pattern = df.groupby(['day_of_week', 'day_name', 'parameter'])['value'].mean().reset_index()
        weekly_pattern.columns = ['day_of_week', 'day_name', 'parameter', 'avg_value']
        
        print("✅ Temporal patterns analyzed")
        return hourly_pattern, weekly_pattern
    
    def save_to_mongodb(self, df, collection_name):
        """Save DataFrame to MongoDB"""
        print(f"\n💾 Saving to {collection_name}...")
        
        # Convert DataFrame to dict records
        records = df.to_dict('records')
        
        # Convert numpy types to Python types
        for record in records:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    if pd.isna(value):
                        record[key] = None
                    else:
                        record[key] = float(value)
                elif pd.isna(value):
                    record[key] = None
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.to_pydatetime()
                elif isinstance(value, dt.date) and not isinstance(value, dt.datetime):
                    record[key] = dt.datetime.combine(value, dt.datetime.min.time())
            
            record['processed_at'] = dt.datetime.utcnow()
        
        # Clear existing data
        self.db.clear_collection(collection_name)
        
        # Insert new data
        if records:
            count = self.db.insert_many(collection_name, records)
            print(f"✅ Saved {count} records")
    
    def process_pipeline(self):
        """Execute complete processing pipeline"""
        print("\n" + "="*70)
        print("🔄 DATA PROCESSING PIPELINE")
        print("="*70)
        
        try:
            # Load data
            df = self.load_raw_data()
            if df is None:
                return
            
            # Clean data
            df = self.clean_data(df)
            
            # Calculate AQI
            df = self.calculate_aqi(df)
            
            # Save processed data
            self.save_to_mongodb(df, config.COLLECTION_PROCESSED)
            
            # Daily aggregations
            daily = self.aggregate_daily(df)
            self.save_to_mongodb(daily, config.COLLECTION_DAILY)
            
            # Hourly aggregations
            hourly = self.aggregate_hourly(df)
            self.save_to_mongodb(hourly, config.COLLECTION_HOURLY)
            
            # City comparison
            comparison = self.city_comparison(df)
            
            # Temporal patterns
            hourly_pattern, weekly_pattern = self.temporal_patterns(df)
            
            print("\n" + "="*70)
            print("✅ PROCESSING PIPELINE COMPLETED")
            print("="*70)
            
            # Show sample
            print("\n📊 Sample Processed Data:")
            print(df[['city', 'parameter', 'value', 'aqi_category', 'date', 'hour']].head(20))
            
        except Exception as e:
            print(f"\n❌ Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        self.db.close()

if __name__ == "__main__":
    processor = DataProcessor()
    try:
        processor.process_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.close()