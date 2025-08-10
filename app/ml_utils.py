import pickle
import pandas as pd
from sqlalchemy.orm import Session
from . import crud, models
from datetime import datetime, timedelta

MODEL_PATH = "ml/model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("ML model loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}. Predictions will not work.")
    model = None

def get_features_for_soldier(db: Session, soldier_id: str) -> pd.DataFrame:
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=14)
    
    reports = crud.get_mood_reports_for_soldier_daterange(db, soldier_id, start_date, end_date)
    
    if len(reports) < 3:
        return None
        
    df = pd.DataFrame([(r.date, r.sleep_hours, r.stress_level) for r in reports], columns=['date', 'sleep_hours', 'stress_level'])
    features = {
        'sleep_avg_14d': [df['sleep_hours'].mean()],
        'stress_avg_14d': [df['stress_level'].mean()],
        'sleep_std_14d': [df['sleep_hours'].std()],
        'stress_std_14d': [df['stress_level'].std()],
        'reports_count_14d': [len(df)]
    }
    feature_df = pd.DataFrame(features).fillna(0)
    return feature_df

def predict_risk(features: pd.DataFrame) -> dict:
    if model is None:
        return {"score": 0.5, "label": "Unknown"}
    expected_features = ['sleep_avg_14d', 'stress_avg_14d', 'sleep_std_14d', 'stress_std_14d', 'reports_count_14d']
    features = features[expected_features]
    score = model.predict_proba(features)[0][1]
    if score >= 0.75:
        label = "High"
    elif score >= 0.4:
        label = "Medium"
    else:
        label = "Low"
        
    return {"score": score, "label": label}
