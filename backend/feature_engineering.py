import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from models import HealthReport, User

class FeatureEngineer:
    def __init__(self):
        self.lookback_days = 7
        self.min_records_for_trends = 3
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['soldier_id', 'timestamp'])
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['hours_since_last'] = df.groupby('soldier_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['hours_since_last'] = df['hours_since_last'].fillna(24) 
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for soldier_id in df['soldier_id'].unique():
            mask = df['soldier_id'] == soldier_id
            soldier_data = df[mask].sort_values('timestamp')
            
            if len(soldier_data) >= self.min_records_for_trends:
                df.loc[mask, 'sleep_7d_avg'] = soldier_data['sleep_hours'].rolling(
                    window=min(7, len(soldier_data)), min_periods=1
                ).mean()
                
                df.loc[mask, 'hrv_7d_avg'] = soldier_data['hrv_score'].rolling(
                    window=min(7, len(soldier_data)), min_periods=1
                ).mean()
                
                df.loc[mask, 'stress_7d_avg'] = soldier_data['stress_level'].rolling(
                    window=min(7, len(soldier_data)), min_periods=1
                ).mean()
                
                df.loc[mask, 'activity_7d_avg'] = soldier_data['activity_calories'].rolling(
                    window=min(7, len(soldier_data)), min_periods=1
                ).mean()
                
                df.loc[mask, 'sleep_vs_avg'] = soldier_data['sleep_hours'] - df.loc[mask, 'sleep_7d_avg']
                df.loc[mask, 'hrv_vs_avg'] = soldier_data['hrv_score'] - df.loc[mask, 'hrv_7d_avg']
                df.loc[mask, 'stress_vs_avg'] = soldier_data['stress_level'] - df.loc[mask, 'stress_7d_avg']
                df.loc[mask, 'activity_vs_avg'] = soldier_data['activity_calories'] - df.loc[mask, 'activity_7d_avg']
                
                df.loc[mask, 'sleep_volatility'] = soldier_data['sleep_hours'].rolling(
                    window=min(7, len(soldier_data)), min_periods=1
                ).std()
                
                df.loc[mask, 'hrv_volatility'] = soldier_data['hrv_score'].rolling(
                    window=min(7, len(soldier_data)), min_periods=1
                ).std()
        
        rolling_cols = [
            'sleep_7d_avg', 'hrv_7d_avg', 'stress_7d_avg', 'activity_7d_avg',
            'sleep_vs_avg', 'hrv_vs_avg', 'stress_vs_avg', 'activity_vs_avg',
            'sleep_volatility', 'hrv_volatility'
        ]
        
        for col in rolling_cols:
            if col in df.columns:
                if 'vs_avg' in col:
                    df[col] = df[col].fillna(0) 
                elif 'volatility' in col:
                    df[col] = df[col].fillna(0) 
                else:
                    base_col = col.replace('_7d_avg', '')
                    if base_col in ['sleep', 'hrv', 'stress', 'activity']:
                        base_col = base_col + '_hours' if base_col == 'sleep' else base_col + '_score' if base_col == 'hrv' else base_col + '_level' if base_col == 'stress' else base_col + '_calories'
                        if base_col in df.columns:
                            df[col] = df[col].fillna(df[base_col])
        
        return df
    
    def create_risk_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['sleep_quality_score'] = (
            (df['sleep_hours'] / 8.0) * 0.7 + 
            (1 - df['sleep_volatility'].fillna(0) / 3.0) * 0.3
        ).clip(0, 1)
        
        df['recovery_score'] = (
            (df['hrv_score'] / 80.0) * 0.6 + 
            df['sleep_quality_score'] * 0.4
        ).clip(0, 1)
        
        df['stress_burden'] = (
            df['stress_level'] / 10.0 * 0.7 +
            (df['stress_vs_avg'].fillna(0) + 5) / 10.0 * 0.3 
        ).clip(0, 1)
        df['activity_adequacy'] = 1 - np.abs(df['activity_calories'] - 800) / 800
        df['activity_adequacy'] = df['activity_adequacy'].clip(0, 1)
        
        df['resilience_score'] = (
            df['recovery_score'] * 0.4 +
            (1 - df['stress_burden']) * 0.3 +
            df['activity_adequacy'] * 0.3
        )
        
        return df
    
    def engineer_features_for_prediction(self, soldier_id: str, current_data: Dict, db: Session) -> Dict:

        historical_reports = db.query(HealthReport).filter(
            HealthReport.soldier_id == soldier_id
        ).order_by(HealthReport.timestamp.desc()).limit(30).all()
        
        if not historical_reports:
            return {
                **current_data,
                'sleep_7d_avg': current_data['sleep_hours'],
                'hrv_7d_avg': current_data['hrv_score'],
                'stress_7d_avg': current_data['stress_level'],
                'activity_7d_avg': current_data['activity_calories'],
                'sleep_vs_avg': 0,
                'hrv_vs_avg': 0,
                'stress_vs_avg': 0,
                'activity_vs_avg': 0,
                'sleep_volatility': 0,
                'hrv_volatility': 0,
                'resilience_score': 0.5,
                'day_of_week': datetime.now().weekday(),
                'is_weekend': int(datetime.now().weekday() >= 5)
            }
        
        hist_data = []
        for report in historical_reports:
            hist_data.append({
                'soldier_id': report.soldier_id,
                'sleep_hours': report.sleep_hours,
                'hrv_score': report.hrv_score,
                'stress_level': report.stress_level,
                'activity_calories': report.activity_calories,
                'timestamp': report.timestamp
            })
        hist_data.append({
            'soldier_id': soldier_id,
            'sleep_hours': current_data['sleep_hours'],
            'hrv_score': current_data['hrv_score'],
            'stress_level': current_data['stress_level'],
            'activity_calories': current_data['activity_calories'],
            'timestamp': datetime.now()
        })
        
        df = pd.DataFrame(hist_data)
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df)
        df = self.create_risk_composite_features(df)
        latest_features = df.iloc[-1].to_dict()
        feature_keys = [
            'sleep_hours', 'hrv_score', 'stress_level', 'activity_calories',
            'sleep_7d_avg', 'hrv_7d_avg', 'stress_7d_avg', 'activity_7d_avg',
            'sleep_vs_avg', 'hrv_vs_avg', 'stress_vs_avg', 'activity_vs_avg',
            'sleep_volatility', 'hrv_volatility', 'resilience_score',
            'day_of_week', 'is_weekend'
        ]
        
        return {k: v for k, v in latest_features.items() if k in feature_keys}
    
    def prepare_training_data(self, db: Session) -> pd.DataFrame:
        all_reports = db.query(HealthReport).order_by(
            HealthReport.soldier_id, HealthReport.timestamp
        ).all()
        
        if not all_reports:
            raise ValueError("No historical data available for training")
        
        data = []
        for report in all_reports:
            data.append({
                'soldier_id': report.soldier_id,
                'sleep_hours': report.sleep_hours,
                'hrv_score': report.hrv_score,
                'stress_level': report.stress_level,
                'activity_calories': report.activity_calories,
                'timestamp': report.timestamp,
                'is_at_risk': 1 if report.risk_level == 'High' else 0
            })
        
        df = pd.DataFrame(data)
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df)
        df = self.create_risk_composite_features(df)        
        return df

feature_engineer = FeatureEngineer()