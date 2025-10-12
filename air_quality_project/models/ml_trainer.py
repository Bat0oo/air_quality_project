import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys
sys.path.append('..')
from database.db_manager import DatabaseManager
import config
from datetime import datetime, timedelta

class MLTrainer:
    def __init__(self):
        self.db = DatabaseManager()
        self.model = None
        self.scaler = None
        self.city_encoder = None
    
    def load_data(self):
        """Load processed data for ML"""
        print("\n📥 Loading processed data...")
        data = self.db.find(config.COLLECTION_PROCESSED)
        
        if not data:
            print("❌ No processed data found!")
            return None
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} records")
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML"""
        print("\n🔧 Preparing features for ML...")
        
        # Filter for PM2.5 only (our target variable)
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
        
        # Rolling averages
        df_pm25['rolling_avg_3'] = df_pm25.groupby('city')['value'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_pm25['rolling_avg_7'] = df_pm25.groupby('city')['value'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Remove rows with NaN lag features
        df_pm25 = df_pm25.dropna(subset=['prev_value_1', 'prev_value_2', 'rolling_avg_3'])
        
        # Encode city
        self.city_encoder = LabelEncoder()
        df_pm25['city_encoded'] = self.city_encoder.fit_transform(df_pm25['city'])
        
        print(f"✅ Feature engineering complete: {len(df_pm25)} records")
        return df_pm25
    
    def train_model(self, df):
        """Train Random Forest model"""
        print("\n🤖 Training Random Forest model...")
        
        # Select features
        feature_cols = [
            'city_encoded', 'hour', 'day_of_week', 'month',
            'prev_value_1', 'prev_value_2', 'prev_value_3',
            'rolling_avg_3', 'rolling_avg_7'
        ]
        
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
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print("\n📊 Model Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} μg/m³")
        print(f"  MAE: {mae:.2f} μg/m³")
        
        # Feature importance
        print("\n🔍 Top 5 Important Features:")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head())
        
        # Sample predictions
        print("\n🔮 Sample Predictions:")
        sample_df = pd.DataFrame({
            'Actual': y_test[:10].values,
            'Predicted': y_pred[:10],
            'Error': y_test[:10].values - y_pred[:10]
        })
        print(sample_df.to_string())
        
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
            
            # Generate predictions for next 24 hours
            for hour_ahead in range(1, 25):
                # Calculate future hour
                future_hour = (latest['hour'] + hour_ahead) % 24
                future_timestamp = latest['timestamp'] + timedelta(hours=hour_ahead)
                
                # Prepare features
                features = {
                    'city_encoded': self.city_encoder.transform([city])[0],
                    'hour': future_hour,
                    'day_of_week': future_timestamp.dayofweek,
                    'month': future_timestamp.month,
                    'prev_value_1': latest['value'],
                    'prev_value_2': latest['prev_value_1'],
                    'prev_value_3': latest['prev_value_2'],
                    'rolling_avg_3': latest['rolling_avg_3'],
                    'rolling_avg_7': latest['rolling_avg_7']
                }
                
                # Create feature array
                X_pred = np.array([[
                    features['city_encoded'],
                    features['hour'],
                    features['day_of_week'],
                    features['month'],
                    features['prev_value_1'],
                    features['prev_value_2'],
                    features['prev_value_3'],
                    features['rolling_avg_3'],
                    features['rolling_avg_7']
                ]])
                
                # Scale and predict
                X_pred_scaled = self.scaler.transform(X_pred)
                prediction = self.model.predict(X_pred_scaled)[0]
                
                predictions.append({
                    'city': city,
                    'hours_ahead': hour_ahead,
                    'predicted_timestamp': future_timestamp,
                    'predicted_pm25': max(0, prediction),  # Ensure non-negative
                    'created_at': datetime.utcnow()
                })
        
        print(f"✅ Generated {len(predictions)} predictions")
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
        
        # Clear old predictions
        self.db.clear_collection(config.COLLECTION_PREDICTIONS)
        
        # Insert new predictions
        if predictions:
            count = self.db.insert_many(config.COLLECTION_PREDICTIONS, predictions)
            print(f"✅ Saved {count} predictions")
    
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
            
            # Train model
            X_test, y_test, y_pred = self.train_model(df_features)
            
            # Generate future predictions
            predictions = self.generate_predictions(df_features)
            
            # Save model
            self.save_model()
            
            # Save predictions
            self.save_predictions(predictions)
            
            # Summary
            print("\n📊 Prediction Summary by City:")
            pred_df = pd.DataFrame(predictions)
            summary = pred_df.groupby('city')['predicted_pm25'].agg(['mean', 'min', 'max'])
            print(summary)
            
            print("\n" + "="*70)
            print("✅ ML TRAINING PIPELINE COMPLETED")
            print("="*70)
            
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