import requests
import sys
sys.path.append('..')
from database.db_manager import DatabaseManager
import config
from datetime import datetime, timedelta
import time

class DataCollector:
    def __init__(self):
        self.base_url = config.OPENAQ_BASE_URL
        self.headers = {}
        if config.OPENAQ_API_KEY:
            self.headers['X-API-Key'] = config.OPENAQ_API_KEY
        self.db = DatabaseManager()
    
    def fetch_locations(self, city):
        """Get monitoring locations for a city"""
        url = f"{self.base_url}/locations"
        params = {"city": city, "limit": 100}
        
        try:
            response = requests.get(url, params=params, headers=self.headers, 
                                   timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            print(f"❌ Error fetching locations for {city}: {e}")
            return []
    
    def fetch_measurements(self, location_id, date_from, date_to):
        """Fetch measurements for a location"""
        url = f"{self.base_url}/measurements"
        params = {
            "location_id": location_id,
            "date_from": date_from,
            "date_to": date_to,
            "limit": 10000
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers,
                                   timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            print(f"❌ Error fetching measurements: {e}")
            return []
    
    def process_measurement(self, measurement, city):
        """Convert API measurement to database document"""
        try:
            return {
                "location": measurement.get("location"),
                "location_id": measurement.get("locationId"),
                "city": city,
                "country": measurement.get("country"),
                "latitude": measurement.get("coordinates", {}).get("latitude"),
                "longitude": measurement.get("coordinates", {}).get("longitude"),
                "parameter": measurement.get("parameter"),
                "value": measurement.get("value"),
                "unit": measurement.get("unit"),
                "date_utc": measurement.get("date", {}).get("utc"),
                "timestamp": datetime.fromisoformat(
                    measurement.get("date", {}).get("utc", "").replace("Z", "+00:00")
                ) if measurement.get("date", {}).get("utc") else None,
                "fetched_at": datetime.utcnow()
            }
        except Exception as e:
            print(f"⚠️  Error processing measurement: {e}")
            return None
    
    def collect_data(self):
        """Main data collection method"""
        print("\n" + "="*70)
        print("🌍 AIR QUALITY DATA COLLECTION")
        print("="*70)
        
        total_records = 0
        date_to = datetime.utcnow()
        date_from = date_to - timedelta(days=config.DAYS_TO_FETCH)
        
        date_from_str = date_from.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        date_to_str = date_to.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        
        print(f"\n📅 Date range: {date_from_str} to {date_to_str}")
        print(f"🏙️  Cities: {', '.join(config.CITIES)}\n")
        
        for city in config.CITIES:
            print(f"\n{'='*70}")
            print(f"📍 Processing: {city}")
            print(f"{'='*70}")
            
            locations = self.fetch_locations(city)
            print(f"Found {len(locations)} monitoring locations")
            
            if not locations:
                print(f"⚠️  No locations found for {city}")
                continue
            
            city_records = 0
            for idx, location in enumerate(locations, 1):
                location_id = location.get("id")
                location_name = location.get("name", "Unknown")
                
                print(f"\n  [{idx}/{len(locations)}] {location_name}")
                
                measurements = self.fetch_measurements(
                    location_id, date_from_str, date_to_str
                )
                
                if measurements:
                    documents = []
                    for m in measurements:
                        doc = self.process_measurement(m, city)
                        if doc:
                            documents.append(doc)
                    
                    if documents:
                        count = self.db.insert_many(config.COLLECTION_RAW, documents)
                        city_records += count
                        print(f"      ✅ Stored {count} measurements")
                else:
                    print(f"      ⚠️  No measurements found")
                
                time.sleep(0.5)  # Rate limiting
            
            print(f"\n✅ Total for {city}: {city_records} records")
            total_records += city_records
        
        print("\n" + "="*70)
        print(f"✅ DATA COLLECTION COMPLETE!")
        print(f"📊 Total records collected: {total_records}")
        print("="*70)
        
        self.print_summary()
        return total_records
    
    def print_summary(self):
        """Print data collection summary"""
        print("\n📊 Data Summary:")
        print("-" * 70)
        
        total = self.db.count(config.COLLECTION_RAW)
        print(f"Total measurements: {total}")
        
        print("\nBy City:")
        for city in config.CITIES:
            count = self.db.count(config.COLLECTION_RAW, {"city": city})
            print(f"  - {city}: {count}")
        
        print("\nBy Parameter:")
        pipeline = [
            {"$group": {"_id": "$parameter", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = self.db.aggregate(config.COLLECTION_RAW, pipeline)
        for r in results:
            print(f"  - {r['_id']}: {r['count']}")
        
        # Date range
        print("\nDate Range:")
        data = self.db.find(config.COLLECTION_RAW, 
                           projection={"timestamp": 1})
        if data:
            timestamps = [d.get('timestamp') for d in data if d.get('timestamp')]
            if timestamps:
                print(f"  From: {min(timestamps)}")
                print(f"  To: {max(timestamps)}")
    
    def close(self):
        self.db.close()

if __name__ == "__main__":
    collector = DataCollector()
    try:
        collector.collect_data()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.close()
        print("\n✅ Data collection script completed")