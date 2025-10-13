from flask import Flask, render_template, jsonify
from flask_cors import CORS
from air_quality_project.database.db_manager import DatabaseManager
from air_quality_project import config
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Initialize database
db = DatabaseManager()

@app.route('/')
def index():
    """Main dashboard page"""
    stats = db.get_stats()
    latest_readings = get_latest_readings()
    validation_metrics = get_validation_metrics()
    
    return render_template('index.html',
                         stats=stats,
                         latest_readings=latest_readings,
                         validation=validation_metrics,
                         cities=config.CITIES)

@app.route('/api/stats')
def api_stats():
    """Database statistics"""
    return jsonify(db.get_stats())

@app.route('/api/validation-metrics')
def api_validation_metrics():
    """Get ML model validation metrics"""
    return jsonify(get_validation_metrics())

@app.route('/api/latest-readings')
def api_latest_readings():
    """Latest readings for all cities"""
    return jsonify(get_latest_readings())

@app.route('/api/city-trends/<city>')
def api_city_trends(city):
    """7-day trend for a city"""
    data = db.find(config.COLLECTION_DAILY, 
                   {'city': city, 'parameter': 'pm25'})
    
    df = pd.DataFrame(data)
    if df.empty:
        return jsonify({'dates': [], 'avg_values': [], 'min_values': [], 'max_values': []})
    
    df = df.sort_values('date').tail(7)
    
    return jsonify({
        'dates': df['date'].astype(str).tolist(),
        'avg_values': df['avg_value'].round(2).tolist(),
        'min_values': df['min_value'].round(2).tolist(),
        'max_values': df['max_value'].round(2).tolist()
    })

@app.route('/api/hourly-pattern/<city>')
def api_hourly_pattern(city):
    """Hourly pattern for a city"""
    pipeline = [
        {'$match': {'city': city, 'parameter': 'pm25'}},
        {'$group': {
            '_id': '$hour',
            'avg_value': {'$avg': '$value'}
        }},
        {'$sort': {'_id': 1}}
    ]
    
    results = db.aggregate(config.COLLECTION_PROCESSED, pipeline)
    
    hours = [r['_id'] for r in results]
    values = [round(r['avg_value'], 2) for r in results]
    
    return jsonify({'hours': hours, 'values': values})

@app.route('/api/city-comparison')
def api_city_comparison():
    """Compare cities"""
    pipeline = [
        {'$match': {'parameter': 'pm25'}},
        {'$group': {
            '_id': '$city',
            'avg_value': {'$avg': '$value'}
        }},
        {'$sort': {'avg_value': -1}}
    ]
    
    results = db.aggregate(config.COLLECTION_PROCESSED, pipeline)
    
    cities = [r['_id'] for r in results]
    avg_values = [round(r['avg_value'], 2) for r in results]
    
    return jsonify({'cities': cities, 'avg_values': avg_values})

@app.route('/api/predictions/<city>')
def api_predictions(city):
    """Get predictions for a city"""
    data = db.find(config.COLLECTION_PREDICTIONS, 
                   {'city': city})
    
    df = pd.DataFrame(data)
    if df.empty:
        return jsonify({'hours_ahead': [], 'predicted_values': []})
    
    df = df.sort_values('hours_ahead').head(24)
    
    return jsonify({
        'hours_ahead': df['hours_ahead'].tolist(),
        'predicted_values': df['predicted_pm25'].round(2).tolist()
    })

@app.route('/api/weekly-pattern')
def api_weekly_pattern():
    """Weekly pattern across all cities"""
    pipeline = [
        {'$match': {'parameter': 'pm25'}},
        {'$group': {
            '_id': '$day_of_week',
            'avg_value': {'$avg': '$value'}
        }},
        {'$sort': {'_id': 1}}
    ]
    
    results = db.aggregate(config.COLLECTION_PROCESSED, pipeline)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_mapping = {i: days[i] for i in range(7)}
    
    day_names = [day_mapping.get(r['_id'], 'Unknown') for r in results]
    values = [round(r['avg_value'], 2) for r in results]
    
    return jsonify({'days': day_names, 'values': values})

@app.route('/api/parameter-comparison/<city>')
def api_parameter_comparison(city):
    """Compare different parameters for a city"""
    pipeline = [
        {'$match': {'city': city}},
        {'$group': {
            '_id': '$parameter',
            'avg_value': {'$avg': '$value'}
        }}
    ]
    
    results = db.aggregate(config.COLLECTION_PROCESSED, pipeline)
    
    parameters = [r['_id'] for r in results]
    values = [round(r['avg_value'], 2) for r in results]
    
    return jsonify({'parameters': parameters, 'values': values})

def get_latest_readings():
    """Get latest readings for each city"""
    pipeline = [
        {'$sort': {'timestamp': -1}},
        {'$group': {
            '_id': {'city': '$city', 'parameter': '$parameter'},
            'latest_value': {'$first': '$value'},
            'latest_time': {'$first': '$timestamp'},
            'unit': {'$first': '$unit'}
        }}
    ]
    
    results = db.aggregate(config.COLLECTION_PROCESSED, pipeline)
    
    cities_data = {}
    for r in results:
        city = r['_id']['city']
        param = r['_id']['parameter']
        
        if city not in cities_data:
            cities_data[city] = {}
        
        cities_data[city][param] = {
            'value': round(r['latest_value'], 2),
            'time': r['latest_time'].strftime('%Y-%m-%d %H:%M') if r['latest_time'] else 'N/A',
            'unit': r['unit']
        }
    
    return cities_data

def get_validation_metrics():
    """Get ML model validation metrics"""
    try:
        validation_collection = "model_validation"
        data = db.find(validation_collection)
        
        if data and len(data) > 0:
            metrics = data[0]
            return {
                'r2_score': round(metrics.get('r2_score', 0), 4),
                'r2_percent': round(metrics.get('r2_score', 0) * 100, 2),
                'rmse': round(metrics.get('rmse', 0), 2),
                'mae': round(metrics.get('mae', 0), 2),
                'mape': round(metrics.get('mape', 0), 2),
                'cv_mean': round(metrics.get('cv_mean', 0), 4),
                'cv_std': round(metrics.get('cv_std', 0), 4),
                'train_size': metrics.get('train_size', 0),
                'test_size': metrics.get('test_size', 0),
                'total_samples': metrics.get('total_samples', 0),
                'trained_at': metrics.get('trained_at', 'N/A')
            }
        else:
            return None
    except Exception as e:
        print(f"Error getting validation metrics: {e}")
        return None

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING AIR QUALITY DASHBOARD")
    print("="*70)
    
    stats = db.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Raw records: {stats['raw']}")
    print(f"   Processed records: {stats['processed']}")
    print(f"   Daily stats: {stats['daily']}")
    print(f"   Hourly stats: {stats['hourly']}")
    print(f"   Predictions: {stats['predictions']}")
    
    validation = get_validation_metrics()
    if validation:
        print(f"\n🤖 ML Model Performance:")
        print(f"   R² Score: {validation['r2_score']} ({validation['r2_percent']}%)")
        print(f"   RMSE: {validation['rmse']} μg/m³")
        print(f"   MAE: {validation['mae']} μg/m³")
    
    print(f"\n🌐 Dashboard URL: http://localhost:{config.FLASK_PORT}")
    print("="*70 + "\n")
    
    app.run(host=config.FLASK_HOST, 
            port=config.FLASK_PORT, 
            debug=config.FLASK_DEBUG)