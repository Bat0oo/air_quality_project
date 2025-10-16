"""
INTERACTIVE DEMO SCRIPT FOR PROFESSOR
Shows complete workflow: Add Data → Process → Train → Predict

Run this script to demonstrate the entire pipeline!
"""

import pandas as pd
import sys
import os
sys.path.append('..')
from air_quality_project.database.db_manager import DatabaseManager
from air_quality_project import config
from datetime import datetime
import time

class ProfessorDemo:
    def __init__(self):
        self.db = DatabaseManager()
        print("\n" + "="*70)
        print("🎓 AIR QUALITY SYSTEM - LIVE DEMONSTRATION")
        print("="*70)
    
    def step_1_show_current_stats(self):
        """Step 1: Show current system statistics"""
        print("\n" + "="*70)
        print("STEP 1: Current System Status")
        print("="*70)
        
        stats = self.db.get_stats()
        
        print(f"\n📊 Database Statistics:")
        print(f"   Raw data:        {stats['raw']:,} measurements")
        print(f"   Processed data:  {stats['processed']:,} records")
        print(f"   Daily stats:     {stats['daily']:,} aggregations")
        print(f"   Predictions:     {stats['predictions']} forecasts")
        
        # Show data by city
        print(f"\n🏙️  Data by City:")
        pipeline = [
            {"$group": {"_id": "$city", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = self.db.aggregate(config.COLLECTION_RAW, pipeline)
        for r in results:
            print(f"   • {r['_id']}: {r['count']:,} measurements")
        
        input("\n➡️  Press ENTER to continue to Step 2...")
    
    def step_2_add_new_data(self):
        """Step 2: Demonstrate adding new data"""
        print("\n" + "="*70)
        print("STEP 2: Adding New Data")
        print("="*70)
        
        demo_folder = "air_quality_project/data/demo_csv"
        os.makedirs(demo_folder, exist_ok=True)
        
        print(f"\n📁 Demo CSV folder: {demo_folder}")
        print(f"\n💡 To add new data:")
        print(f"   1. Place CSV files in: {demo_folder}")
        print(f"   2. Name files: demo_belgrade.csv, demo_novi_sad.csv, demo_nis.csv")
        print(f"   3. Run the import script")
        
        # Check if demo files exist
        import glob
        demo_files = glob.glob(os.path.join(demo_folder, "*.csv"))
        
        if not demo_files:
            print(f"\n⚠️  No demo CSV files found!")
            print(f"\n📥 Please download demo CSVs from OpenAQ Explorer:")
            print(f"   → https://explore.openaq.org/")
            print(f"   → Search for Belgrade, Novi Sad, Nis")
            print(f"   → Download last 7 days of data")
            print(f"   → Place in {demo_folder}")
            
            choice = input(f"\n❓ Do you have CSV files ready? (yes/no): ").lower()
            if choice != 'yes':
                print("\n⏭️  Skipping data import for now...")
                input("\n➡️  Press ENTER to continue to Step 3...")
                return
        
        print(f"\n✅ Found {len(demo_files)} CSV file(s):")
        for f in demo_files:
            print(f"   • {os.path.basename(f)}")
        
        choice = input(f"\n❓ Import these files now? (yes/no): ").lower()
        
        if choice == 'yes':
            print(f"\n🔄 Importing data...")
            
            total_imported = 0
            for csv_file in demo_files:
                filename = os.path.basename(csv_file)
                city_name = os.path.splitext(filename)[0].replace('demo_', '').replace('_', ' ').title()
                
                print(f"\n   Processing {city_name}...")
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    documents = []
                    for _, row in df.iterrows():
                        try:
                            doc = {
                                "location": str(row.get('location_name', 'Demo Station')),
                                "city": city_name,
                                "country": "Serbia",
                                "parameter": str(row.get('parameter', 'pm25')),
                                "value": float(row.get('value', 0)),
                                "unit": str(row.get('unit', 'µg/m³')),
                                "timestamp": pd.to_datetime(row.get('datetimeUtc', datetime.utcnow())),
                                "fetched_at": datetime.utcnow(),
                                "demo_data": True  # Mark as demo data
                            }
                            documents.append(doc)
                        except Exception as e:
                            continue
                    
                    if documents:
                        count = self.db.insert_many(config.COLLECTION_RAW, documents)
                        total_imported += count
                        print(f"      ✅ Imported {count} measurements")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            print(f"\n✅ Total imported: {total_imported} new measurements")
        
        input("\n➡️  Press ENTER to continue to Step 3...")
    
    def step_3_process_data(self):
        """Step 3: Process the data"""
        print("\n" + "="*70)
        print("STEP 3: Data Processing Pipeline")
        print("="*70)
        
        print(f"\n🔄 Running data processor...")
        print(f"   This will:")
        print(f"   • Clean and validate data")
        print(f"   • Calculate Air Quality Index (AQI)")
        print(f"   • Generate hourly/daily aggregations")
        print(f"   • Perform temporal analysis")
        
        choice = input(f"\n❓ Run data processor now? (yes/no): ").lower()
        
        if choice == 'yes':
            print(f"\n⏳ Processing... (this takes 1-2 minutes)")
            
            # Import and run processor
            from air_quality_project.data.data_processor import DataProcessor
            
            processor = DataProcessor()
            processor.process_pipeline()
            processor.close()
            
            print(f"\n✅ Data processing complete!")
        else:
            print(f"\n⏭️  Skipped processing")
        
        input("\n➡️  Press ENTER to continue to Step 4...")
    
    def step_4_train_model(self):
        """Step 4: Train ML model"""
        print("\n" + "="*70)
        print("STEP 4: Machine Learning Training")
        print("="*70)
        
        print(f"\n🤖 Training Random Forest model...")
        print(f"   This will:")
        print(f"   • Extract features from processed data")
        print(f"   • Split into train/test sets (80/20)")
        print(f"   • Train Random Forest with 100 trees")
        print(f"   • Validate using cross-validation")
        print(f"   • Generate 24-hour predictions")
        
        choice = input(f"\n❓ Train model now? (yes/no): ").lower()
        
        if choice == 'yes':
            print(f"\n⏳ Training...")
            
            # Import and run ML trainer
            from air_quality_project.models.ml_trainer import MLTrainer
            
            trainer = MLTrainer()
            trainer.run_training_pipeline()
            trainer.close()
            
            print(f"\n✅ Model training complete!")
        else:
            print(f"\n⏭️  Skipped training")
        
        input("\n➡️  Press ENTER to continue to Step 5...")
    
    def step_5_show_results(self):
        """Step 5: Display results and predictions"""
        print("\n" + "="*70)
        print("STEP 5: Results & Predictions")
        print("="*70)
        
        # Get validation metrics
        validation_data = self.db.find("model_validation")
        
        if validation_data and len(validation_data) > 0:
            metrics = validation_data[0]
            
            print(f"\n🎯 Model Performance:")
            print(f"   R² Score:    {metrics.get('r2_score', 0):.4f} ({metrics.get('r2_score', 0)*100:.2f}%)")
            print(f"   RMSE:        {metrics.get('rmse', 0):.2f} μg/m³")
            print(f"   MAE:         {metrics.get('mae', 0):.2f} μg/m³")
            print(f"   Train size:  {metrics.get('train_size', 0):,} samples")
            print(f"   Test size:   {metrics.get('test_size', 0):,} samples")
        else:
            print(f"\n⚠️  No validation metrics found. Please train the model first.")
        
        # Show predictions
        predictions = self.db.find(config.COLLECTION_PREDICTIONS)
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            
            print(f"\n🔮 Generated Predictions:")
            print(f"   Total: {len(predictions)} predictions (24 hours × {len(config.CITIES)} cities)")
            
            print(f"\n   By City:")
            summary = pred_df.groupby('city')['predicted_pm25'].agg(['mean', 'min', 'max'])
            for city in summary.index:
                print(f"   • {city}:")
                print(f"      Mean: {summary.loc[city, 'mean']:.2f} μg/m³")
                print(f"      Range: {summary.loc[city, 'min']:.2f} - {summary.loc[city, 'max']:.2f} μg/m³")
            
            # Show sample predictions
            print(f"\n   Sample 6-hour forecast for {pred_df.iloc[0]['city']}:")
            sample = pred_df[pred_df['city'] == pred_df.iloc[0]['city']].head(6)
            for _, row in sample.iterrows():
                print(f"      +{row['hours_ahead']}h: {row['predicted_pm25']:.2f} μg/m³")
        else:
            print(f"\n⚠️  No predictions found. Please train the model first.")
        
        input("\n➡️  Press ENTER to continue to Step 6...")
    
    def step_6_launch_dashboard(self):
        """Step 6: Launch dashboard"""
        print("\n" + "="*70)
        print("STEP 6: Web Dashboard")
        print("="*70)
        
        print(f"\n🌐 Launch Interactive Dashboard")
        print(f"\n   The dashboard shows:")
        print(f"   • Real-time statistics")
        print(f"   • Model validation metrics")
        print(f"   • Latest air quality readings")
        print(f"   • Interactive charts (trends, patterns, predictions)")
        print(f"   • City comparison")
        
        choice = input(f"\n❓ Launch dashboard now? (yes/no): ").lower()
        
        if choice == 'yes':
            print(f"\n🚀 Starting Flask server...")
            print(f"\n   Dashboard will be available at:")
            print(f"   → http://localhost:5000")
            print(f"\n   Press Ctrl+C to stop the server\n")
            
            # Import and run Flask app
            from air_quality_project import app as flask_app
            flask_app.app.run(host='0.0.0.0', port=5000, debug=False)
        else:
            print(f"\n💡 To launch dashboard later, run: python app.py")
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            self.step_1_show_current_stats()
            self.step_2_add_new_data()
            self.step_3_process_data()
            self.step_4_train_model()
            self.step_5_show_results()
            self.step_6_launch_dashboard()
            
            print("\n" + "="*70)
            print("✅ DEMONSTRATION COMPLETE!")
            print("="*70)
            print(f"\n🎓 Summary:")
            print(f"   1. ✅ Showed current system status")
            print(f"   2. ✅ Added new data from CSV")
            print(f"   3. ✅ Processed data with Pandas")
            print(f"   4. ✅ Trained ML model with validation")
            print(f"   5. ✅ Generated predictions")
            print(f"   6. ✅ Launched interactive dashboard")
            
            print(f"\n💡 Key Demonstration Points:")
            print(f"   • Complete ETL pipeline (Extract, Transform, Load)")
            print(f"   • Big data processing with Pandas")
            print(f"   • Machine learning with proper validation")
            print(f"   • Real-time visualization dashboard")
            print(f"   • Scalable NoSQL database architecture")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Demo interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.db.close()

def main():
    """Main demo entry point"""
    demo = ProfessorDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()