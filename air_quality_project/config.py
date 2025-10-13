import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "air_quality_db"

# Collections
COLLECTION_RAW = "raw_data"
COLLECTION_PROCESSED = "processed_data"
COLLECTION_DAILY = "daily_stats"
COLLECTION_HOURLY = "hourly_stats"
COLLECTION_PREDICTIONS = "predictions"

# OpenAQ API
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", None)

# Cities to monitor
CITIES = ["Beograd", "Novi Sad", "Nis"]

# Air quality parameters
PARAMETERS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

# Data collection settings
DAYS_TO_FETCH = 90  # Historical data range
REQUEST_TIMEOUT = 15  # API timeout in seconds

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Machine Learning settings
ML_TEST_SIZE = 0.2  # 80-20 train-test split
ML_RANDOM_STATE = 42
ML_MODEL_PATH = "models/saved_models/"

# Create directories if they don't exist
os.makedirs(ML_MODEL_PATH, exist_ok=True)