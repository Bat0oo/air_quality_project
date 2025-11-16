# air_quality_project
A comprehensive air quality prediction system with ETL pipeline, Random Forest machine learning, and interactive Plotly visualizations for Serbian cities.

## 🔍 Overview
This is a full-featured air quality monitoring and prediction platform built with Python and Flask. The system implements a complete ETL (Extract, Transform, Load) pipeline for processing CSV data from multiple Serbian cities (Beograd, Novi Sad, Niš), uses Random Forest machine learning models for forecasting 6 pollutants (SO2, PM2.5, PM10, O3, NO2, CO), and provides rich interactive Plotly dashboards with multiple chart types. Data is managed in MongoDB with comprehensive data processing using pandas and numpy. The application includes EPA-standard AQI calculation with color-coded health categories, model performance metrics, and a complete demo showcasing the entire workflow from data loading through prediction visualization.

## 🧰 Project Structure
```
air_quality_project/
├── data/
│   ├── csv_data_collector.py    # Module to collect data from CSV files
│   └── data_processor.py         # Module to process and clean collected data
|   └── dataset/                      # CSV data files (beograd.csv, novi_sad.csv, nis.csv)
├── models/
│   └── ml_trainer.py             # Module to train Random Forest models
├── demo/
│   ├── generate_democsv.py       # Generate random demo CSV data
│   └── demo.py                   # Complete demo workflow
├── templates/
│   └── index.html                # Flask HTML template with Bootstrap 4
└── app.py                        # Main Flask application                    
```

Flask Backend & Web UI — Flask application with Jinja2 templates and Bootstrap 4 UI
ETL Pipeline — Modular architecture with separate data collection and processing modules
Random Forest Models — Separate trained model for each of 6 pollutants (SO2, PM2.5, PM10, O3, NO2, CO)
MongoDB Database — NoSQL storage for time-series air quality data
Interactive Dashboard — Multiple Plotly visualizations (bar charts, line charts, box plots)
Data Processing Layer — Pandas/numpy-based ETL with datetime extraction and one-hot encoding
AQI Calculation Engine — EPA-standard Air Quality Index with breakpoint interpolation
Demo Module — Complete end-to-end demonstration with random data generation
City & Time Selection — User input for city (Beograd/Novi Sad/Niš) and temporal features

## 🚀 Getting Started

### Standard Setup
1. Clone the repository
```bash
git clone https://github.com/Bat0oo/air_quality_project.git
cd air_quality_project
```

2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. Install all dependencies
```bash
pip install -r requirements.txt
# Required: pandas, flask, scikit-learn, plotly, pymongo
```

4. Configure MongoDB connection

5. Run the complete pipeline:
```bash
# Step 1: Collect data from CSV files
python -m air_quality_project.data.csv_data_collector

# Step 2: Process the collected data
python -m air_quality_project.data.data_processor

# Step 3: Train the machine learning models
python -m air_quality_project.models.ml_trainer

# Step 4: Start the Flask application
python -m air_quality_project.app
```

6. Access the web interface at http://localhost:5000

### Demo Mode (Complete Workflow Demonstration)
To run the full demo that generates random data and shows the entire process:

1. Ensure virtual environment is activated
```bash
venv\Scripts\activate
```

2. Generate demo CSV data
```bash
python -m air_quality_project.demo.generate_democsv
```

3. Run the complete demo (shows data generation → import → training → prediction → visualization)
```bash
python -m air_quality_project.demo.demo
```

### Alternative: Quick Start with Flask Only
If you already have trained models and processed data:
```bash
python app.py  # Run from root directory
```

## ✨ Key Features
Complete ETL pipeline: Extract from 3 CSV files → Transform (clean, parse datetime, feature engineering) → Load to MongoDB
6 Random Forest models (one per pollutant: SO2, PM2.5, PM10, O3, NO2, CO)
Interactive Plotly visualizations:
  - Average air quality by city (grouped bar chart)
  - Hourly patterns across 24 hours (multi-line chart)
  - Daily patterns by day of week (Mon-Sun line chart)
  - Monthly patterns across 12 months (line chart)
  - Parameter distribution (box plots for all pollutants)
Flask web interface with Bootstrap 4 responsive design
EPA-standard AQI calculation with 6 health categories (Good to Hazardous)
Color-coded AQI display with health descriptions
Overall AQI + individual pollutant AQIs
Main pollutant identification
Model performance table showing R² scores for each pollutant
City selection (Beograd, Novi Sad, Niš)
Time-based feature inputs (hour, day, month, day of week)
Simultaneous prediction of all 6 pollutants
MongoDB integration for scalable data storage
Datetime feature extraction (hour, day, month, day_of_week)
Modular architecture for easy maintenance and extension

## 🧩 Tech Stack
Backend & Frontend: Python, Flask
Database: MongoDB (NoSQL)
Machine Learning: Random Forest Regressor (scikit-learn), pandas, numpy
ETL: Custom pandas-based pipeline with datetime parsing
Visualization: Plotly Express & Plotly Graph Objects
Data Format: CSV files (3 cities: Beograd, Novi Sad, Niš)
Model Training: Train-test split (80/20), 100 trees per forest, parallel processing
Architecture: Modular package structure with separate data, models, and demo modules

## 📂 Use-Cases
Environmental research and air quality analysis for Serbia
Advanced air quality prediction and forecasting
Public health monitoring with AQI health impact categories
Learning ETL pipeline development with real-world datasets
Building ML-powered dashboards with Plotly and Flask
Understanding Random Forest ensemble methods
Data science projects with temporal feature engineering
Multi-target regression (6 pollutants)
Smart city environmental monitoring applications
Modular ML project architecture examples
