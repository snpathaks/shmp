import joblib
import pandas as pd
from schemas import HealthReportCreate

class RiskPredictor:
    def __init__(self, model_path = 'backend/risk_model.pkl'):
        self.model = joblib.load(model_path)
        self.features = ['sleep_hours', 'hrv_score', 'stress_level', 'activity_calories']
        print("Risk prediction model loaded.")

    def predict_risk(self, report: HealthReportCreate) -> dict:
        data = pd.DataFrame([{
            'sleep_hours': report.sleep_hours,
            'hrv_score': report.hrv_score,
            'stress_level': report.stress_level,
            'activity_calories': report.activity_calories
        }], columns = self.features)

        probability = self.model.predict_proba(data)[0][1] 
        
        level = "Low"
        color = "success"
        if probability > 0.7:
            level = "High"
            color = "danger"
        elif probability > 0.4:
            level = "Medium"
            color = "warning"
            
        return {
            "risk_probability": round(probability, 2),
            "risk_level": level,
            "risk_color": color
        }
predictor = RiskPredictor()