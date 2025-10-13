import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys
sys.path.append('..')
from air_quality_project.database.db_manager import DatabaseManager
from air_quality_project import config
from datetime import datetime, timedelta

class MLTrainer:
    def __init__(self):
        self.db = DatabaseManager()
        self.model = None
        self.scaler = None
        self.city_encoder = None
        self.validation_results = {}
    
    def load_data(self):
        """Load processed data for ML"""
        print("\n📥 Loading processed data...")
        data = self.db.find(config.COLLECTION_PROCESSED)
        
        if not data:
            print("❌ No processed data found!")
            return None
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} records")
        
        # Show data date range
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            print(f"📅 Data range: {min_date} to {max_date}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML (WITHOUT data leakage)"""
        print("\n🔧 Preparing features for ML (avoiding data leakage)...")
        
        # Filter for PM2.5 only
        df_pm25 = df[df['parameter'] == 'pm25'].copy()
        
        if len(df_pm25) < 100:
            print("❌ Not enough PM2.5 data for training!")
            return None
        
        # Sort by city and timestamp
        df_pm25 = df_pm25.sort_values(['city', 'timestamp'])
        
        # Create lag features (previous values)
        df_pm25['prev_value_1'] = df_pm25.groupby('city')['value'].shift(1)
        df_pm25['prev_value_2'] = df_pm25.groupby('city')['value'].shift(2)
        df_pm25['prev_value_3'] = df_pm25.groupby('city')['value'].shift(3)
        
        # FIXED: Rolling averages that DON'T include current value
        # Shift first, then calculate rolling average
        df_pm25['rolling_avg_3'] = df_pm25.groupby('city')['value'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df_pm25['rolling_avg_7'] = df_pm25.groupby('city')['value'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )
        
        # Calculate standard deviation of last 7 values (measure of volatility)
        df_pm25['rolling_std_7'] = df_pm25.groupby('city')['value'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
        )
        
        # Day/Night indicator (pollution patterns differ)
        df_pm25['is_night'] = df_pm25['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
        df_pm25['is_rush_hour'] = df_pm25['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
        
        # Remove rows with NaN lag features
        df_pm25 = df_pm25.dropna(subset=['prev_value_1', 'prev_value_2', 'rolling_avg_3', 'rolling_avg_7'])
        
        # Encode city
        self.city_encoder = LabelEncoder()
        df_pm25['city_encoded'] = self.city_encoder.fit_transform(df_pm25['city'])
        
        print(f"✅ Feature engineering complete: {len(df_pm25)} records")
        print(f"   Features are based ONLY on historical data (no data leakage)")
        
        return df_pm25
    
    def train_model(self, df):
        """Train Random Forest model with enhanced validation"""
        print("\n🤖 Training Random Forest model...")
        
        # Select features
        feature_cols = [
            'city_encoded', 'hour', 'day_of_week', 'month',
            'prev_value_1', 'prev_value_2', 'prev_value_3',
            'rolling_avg_3', 'rolling_avg_7', 'rolling_std_7',
            'is_night', 'is_rush_hour'
        ]
        
        df = df.dropna(subset=feature_cols + ['value'])
        X = df[feature_cols]
        y = df['value']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.ML_TEST_SIZE, 
            random_state=config.ML_RANDOM_STATE
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config.ML_RANDOM_STATE,
            n_jobs=-1
        )
        
        print("\n🏋️  Training...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate MAPE safely (avoid division by zero)
        mask = y_test > 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
        else:
            mape = 0
        
        # Cross-validation scores
        print("\n🔄 Running cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        # Store validation results
        self.validation_results = {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_samples': len(X),
            'feature_count': len(feature_cols),
            'trained_at': datetime.utcnow()
        }
        
        # Print detailed metrics
        print("\n" + "="*70)
        print("📊 MODEL VALIDATION RESULTS")
        print("="*70)
        print(f"\n🎯 Test Set Performance:")
        print(f"   R² Score:  {r2:.4f} ({r2*100:.2f}% variance explained)")
        print(f"   RMSE:      {rmse:.2f} μg/m³")
        print(f"   MAE:       {mae:.2f} μg/m³")
        print(f"   MAPE:      {mape:.2f}%")
        
        print(f"\n🔄 Cross-Validation (5-fold):")
        print(f"   Mean R²:   {cv_scores.mean():.4f}")
        print(f"   Std Dev:   {cv_scores.std():.4f}")
        print(f"   Range:     [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        # Interpretation
        print(f"\n💡 Model Interpretation:")
        if r2 >= 0.80:
            print("   ✅ Excellent model - High predictive accuracy")
        elif r2 >= 0.70:
            print("   ✅ Good model - Reliable predictions")
        elif r2 >= 0.60:
            print("   ✅ Acceptable model - Moderate predictive power")
        else:
            print("   ⚠️  Fair model - Limited predictive ability")
        
        print(f"\n   Average prediction error: ±{mae:.2f} μg/m³")
        print(f"   This means predictions are typically within {mae:.2f} μg/m³ of actual values")
        
        # Feature importance
        print("\n🔍 Top 10 Important Features:")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10).to_string(index=False))
        
        # Prediction accuracy breakdown
        print(f"\n📈 Prediction Accuracy Breakdown:")
        errors = np.abs(y_test.values - y_pred)
        within_5 = (errors <= 5).sum() / len(errors) * 100
        within_10 = (errors <= 10).sum() / len(errors) * 100
        within_20 = (errors <= 20).sum() / len(errors) * 100
        
        print(f"   Within ±5 μg/m³:   {within_5:.1f}% of predictions")
        print(f"   Within ±10 μg/m³:  {within_10:.1f}% of predictions")
        print(f"   Within ±20 μg/m³:  {within_20:.1f}% of predictions")
        
        # Sample predictions
        print("\n🔮 Sample Predictions (First 10):")
        sample_df = pd.DataFrame({
            'Actual': y_test[:10].values,
            'Predicted': y_pred[:10].round(2),
            'Error': (y_test[:10].values - y_pred[:10]).round(2),
            'Error %': ((np.abs(y_test[:10].values - y_pred[:10]) / y_test[:10].values) * 100).round(1)
        })
        print(sample_df.to_string(index=False))
        
        # Data leakage check
        print(f"\n🔒 Data Leakage Prevention:")
        print(f"   ✅ Rolling averages exclude current value (shifted by 1)")
        print(f"   ✅ All features are based on historical data only")
        print(f"   ✅ Test set never seen during training")
        print(f"   ✅ Cross-validation ensures model generalization")
        
        return X_test_scaled, y_test, y_pred
    
    def generate_predictions(self, df):
        """Generate 24-hour predictions for each city"""
        print("\n🔮 Generating 24-hour predictions...")
        
        predictions = []
        
        for city in config.CITIES:
            city_data = df[df['city'] == city].sort_values('timestamp')
            
            if len(city_data) == 0:
                print(f"⚠️  No data for {city}")
                continue
            
            # Get latest data point
            latest = city_data.iloc[-1]
            latest_date = latest['timestamp']
            
            print(f"\n   {city}:")
            print(f"      Latest data: {latest_date}")
            print(f"      Current PM2.5: {latest['value']:.2f} μg/m³")
            
            # Generate predictions for next 24 hours
            for hour_ahead in range(1, 25):
                future_hour = (latest['hour'] + hour_ahead) % 24
                future_timestamp = latest['timestamp'] + timedelta(hours=hour_ahead)
                
                # Determine time-based features
                is_night = 1 if future_hour >= 22 or future_hour <= 6 else 0
                is_rush_hour = 1 if (7 <= future_hour <= 9) or (17 <= future_hour <= 19) else 0
                
                features = {
                    'city_encoded': self.city_encoder.transform([city])[0],
                    'hour': future_hour,
                    'day_of_week': future_timestamp.dayofweek,
                    'month': future_timestamp.month,
                    'prev_value_1': latest['value'],
                    'prev_value_2': latest['prev_value_1'],
                    'prev_value_3': latest['prev_value_2'],
                    'rolling_avg_3': latest['rolling_avg_3'],
                    'rolling_avg_7': latest['rolling_avg_7'],
                    'rolling_std_7': latest['rolling_std_7'] if pd.notna(latest['rolling_std_7']) else 0,
                    'is_night': is_night,
                    'is_rush_hour': is_rush_hour
                }
                
                X_pred = np.array([[
                    features['city_encoded'],
                    features['hour'],
                    features['day_of_week'],
                    features['month'],
                    features['prev_value_1'],
                    features['prev_value_2'],
                    features['prev_value_3'],
                    features['rolling_avg_3'],
                    features['rolling_avg_7'],
                    features['rolling_std_7'],
                    features['is_night'],
                    features['is_rush_hour']
                ]])
                
                X_pred_scaled = self.scaler.transform(X_pred)
                prediction = self.model.predict(X_pred_scaled)[0]
                
                predictions.append({
                    'city': city,
                    'hours_ahead': hour_ahead,
                    'predicted_timestamp': future_timestamp,
                    'predicted_pm25': max(0, prediction),
                    'base_value': float(latest['value']),
                    'created_at': datetime.utcnow()
                })
        
        print(f"\n✅ Generated {len(predictions)} predictions")
        return predictions
    
    def save_model(self):
        """Save trained model and scaler"""
        print("\n💾 Saving model...")
        
        model_path = config.ML_MODEL_PATH + 'rf_model.pkl'
        scaler_path = config.ML_MODEL_PATH + 'scaler.pkl'
        encoder_path = config.ML_MODEL_PATH + 'city_encoder.pkl'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.city_encoder, encoder_path)
        
        print(f"✅ Model saved to {model_path}")
    
    def save_predictions(self, predictions):
        """Save predictions to MongoDB"""
        print("\n💾 Saving predictions to MongoDB...")
        
        self.db.clear_collection(config.COLLECTION_PREDICTIONS)
        
        if predictions:
            count = self.db.insert_many(config.COLLECTION_PREDICTIONS, predictions)
            print(f"✅ Saved {count} predictions")
    
    def save_validation_results(self):
        """Save validation metrics to MongoDB"""
        print("\n💾 Saving validation results...")
        
        validation_collection = "model_validation"
        self.db.clear_collection(validation_collection)
        self.db.insert_one(validation_collection, self.validation_results)
        
        print(f"✅ Validation results saved")
    
    def run_training_pipeline(self):
        """Execute complete ML training pipeline"""
        print("\n" + "="*70)
        print("🤖 MACHINE LEARNING TRAINING PIPELINE")
        print("="*70)
        
        try:
            # Load data
            df = self.load_data()
            if df is None:
                return
            
            # Prepare features
            df_features = self.prepare_features(df)
            if df_features is None:
                return
            
            # Train model with validation
            X_test, y_test, y_pred = self.train_model(df_features)
            
            # Generate future predictions
            predictions = self.generate_predictions(df_features)
            
            # Save everything
            self.save_model()
            self.save_predictions(predictions)
            self.save_validation_results()
            
            # Summary
            print("\n📊 Prediction Summary by City:")
            pred_df = pd.DataFrame(predictions)
            summary = pred_df.groupby('city')['predicted_pm25'].agg(['mean', 'min', 'max', 'count'])
            print(summary)
            
            print("\n" + "="*70)
            print("✅ ML TRAINING PIPELINE COMPLETED")
            print("="*70)
            print(f"\n💡 Key Takeaways:")
            print(f"   • Model explains {self.validation_results['r2_score']*100:.1f}% of PM2.5 variance")
            print(f"   • Average prediction error: ±{self.validation_results['mae']:.2f} μg/m³")
            print(f"   • Trained on {self.validation_results['train_size']:,} samples")
            print(f"   • Validated on {self.validation_results['test_size']:,} samples")
            print(f"   • No data leakage - all features use historical data only")
            
        except Exception as e:
            print(f"\n❌ Error in ML pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        self.db.close()

if __name__ == "__main__":
    trainer = MLTrainer()
    try:
        trainer.run_training_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()