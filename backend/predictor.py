import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, List
from sqlalchemy.orm import Session
from schemas import HealthReportCreate
from backend.feature_engineering import feature_engineer

class RiskPredictor:
    def __init__(self, model_path='backend/risk_model.pkl'):
        self.model = joblib.load(model_path)
        self.feature_names = [
            'sleep_hours', 'hrv_score', 'stress_level', 'activity_calories',
            'sleep_7d_avg', 'hrv_7d_avg', 'stress_7d_avg', 'activity_7d_avg',
            'sleep_vs_avg', 'hrv_vs_avg', 'stress_vs_avg', 'activity_vs_avg',
            'sleep_volatility', 'hrv_volatility', 'resilience_score',
            'day_of_week', 'is_weekend'
        ]
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_available = True
            print("SHAP explainer initialized for model interpretability.")
        except Exception as e:
            print(f"SHAP initialization failed: {e}. Predictions will work without explanations.")
            self.shap_available = False
        
        print("Enhanced risk prediction model loaded with feature engineering.")
    
    def predict_risk(self, report: HealthReportCreate, db: Session) -> Dict:
        engineered_features = feature_engineer.engineer_features_for_prediction(
            soldier_id=report.soldier_id,
            current_data={
                'sleep_hours': report.sleep_hours,
                'hrv_score': report.hrv_score,
                'stress_level': report.stress_level,
                'activity_calories': report.activity_calories
            },
            db=db
        )
        
        feature_vector = []
        for feature_name in self.feature_names:
            value = engineered_features.get(feature_name, 0)
            if pd.isna(value):
                value = 0
            feature_vector.append(value)
        
        feature_array = np.array([feature_vector])
        probability = self.model.predict_proba(feature_array)[0][1]
        
        level = "Low"
        color = "success"
        if probability > 0.7:
            level = "High"
            color = "danger"
        elif probability > 0.4:
            level = "Medium"
            color = "warning"
        
        # Generate explanation if SHAP is available
        explanation = self._generate_explanation(feature_array, engineered_features)
        
        return {
            "risk_probability": round(probability, 3),
            "risk_level": level,
            "risk_color": color,
            "explanation": explanation,
            "engineered_features": engineered_features
        }
    
    def _generate_explanation(self, feature_array: np.ndarray, engineered_features: Dict) -> Dict:
        if not self.shap_available:
            return {"message": "Explanations not available - SHAP not initialized"}
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(feature_array)
            
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class (at-risk)
            
            # Create feature importance ranking
            feature_importance = []
            for i, feature_name in enumerate(self.feature_names):
                impact = shap_values[0][i]
                value = feature_array[0][i]
                
                feature_importance.append({
                    'feature': feature_name,
                    'value': round(value, 2),
                    'impact': round(impact, 4),
                    'abs_impact': abs(impact)
                })
            
            # Sort by absolute impact
            feature_importance.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            # Get top contributing factors
            top_factors = feature_importance[:5]
            
            # Generate human-readable explanations
            explanations = []
            for factor in top_factors:
                explanation = self._interpret_feature_impact(
                    factor['feature'], 
                    factor['value'], 
                    factor['impact'],
                    engineered_features
                )
                if explanation:
                    explanations.append(explanation)
            
            return {
                "top_factors": top_factors,
                "explanations": explanations,
                "shap_values": shap_values[0].tolist()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate explanation: {str(e)}"}
    
    def _interpret_feature_impact(self, feature: str, value: float, impact: float, features: Dict) -> str:
        impact_direction = "increases" if impact > 0 else "decreases"
        impact_strength = "significantly" if abs(impact) > 0.1 else "moderately"
        
        interpretations = {
            'sleep_hours': f"Sleep duration of {value:.1f} hours {impact_strength} {impact_direction} risk",
            'hrv_score': f"HRV score of {value:.0f} {impact_strength} {impact_direction} risk",
            'stress_level': f"Stress level of {value:.0f}/10 {impact_strength} {impact_direction} risk",
            'activity_calories': f"Activity level of {value:.0f} calories {impact_strength} {impact_direction} risk",
            'sleep_vs_avg': f"Sleep {'above' if value > 0 else 'below'} personal average by {abs(value):.1f}h {impact_strength} {impact_direction} risk",
            'hrv_vs_avg': f"HRV {'above' if value > 0 else 'below'} personal average by {abs(value):.0f} {impact_strength} {impact_direction} risk",
            'stress_vs_avg': f"Stress {'above' if value > 0 else 'below'} personal average by {abs(value):.1f} points {impact_strength} {impact_direction} risk",
            'sleep_volatility': f"Sleep pattern variability of {value:.1f}h {impact_strength} {impact_direction} risk",
            'hrv_volatility': f"HRV variability of {value:.1f} {impact_strength} {impact_direction} risk",
            'resilience_score': f"Overall resilience score of {value:.2f} {impact_strength} {impact_direction} risk",
            'is_weekend': f"{'Weekend' if value else 'Weekday'} timing {impact_strength} {impact_direction} risk"
        }
        
        return interpretations.get(feature, f"{feature}: {value} {impact_strength} {impact_direction} risk")
    
    def get_feature_importance_global(self) -> Dict:
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for i, feature in enumerate(self.feature_names):
                importance_dict[feature] = float(self.model.feature_importances_[i])
            
            # Sort by importance
            sorted_importance = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                'feature_importance': dict(sorted_importance),
                'top_features': [item[0] for item in sorted_importance[:5]]
            }
        
        return {"message": "Feature importance not available for this model type"}

enhanced_predictor = RiskPredictor()