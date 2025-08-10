import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from datetime import datetime, timedelta
import os
import uuid

def generate_synthetic_data(num_soldiers=50, days=90):
    print("Generating synthetic data...")
    base_date = datetime.utcnow() - timedelta(days=days)
    soldiers = [{'id': uuid.uuid4(), 'name': f'Soldier_{i}', 'unit': f'Alpha-{i%4+1}'} for i in range(num_soldiers)]
    
    records = []
    for soldier in soldiers:
        baseline_sleep = np.random.normal(7, 1.5)
        baseline_stress = np.random.normal(2.5, 1)
        risk_propensity = np.random.rand()

        for day in range(days):
            is_bad_day = np.random.rand() < (0.1 + risk_propensity * 0.3)
            
            if is_bad_day:
                sleep = baseline_sleep - np.random.uniform(2, 4)
                stress = baseline_stress + np.random.uniform(1, 2.5)
            else:
                sleep = baseline_sleep + np.random.normal(0, 0.5)
                stress = baseline_stress - np.random.normal(0, 0.5)
            high_risk = 1 if (sleep < 5 and stress > 3.5) else 0
            
            records.append({
                'soldier_id': soldier['id'],
                'date': base_date + timedelta(days=day),
                'sleep_hours': round(np.clip(sleep, 2, 10), 2),
                'stress_level': int(np.clip(stress, 1, 5)),
                'high_risk_target': high_risk
            })
    
    df = pd.DataFrame(records)
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    soldiers_df = pd.DataFrame(soldiers)
    soldiers_df.to_csv(f'{data_dir}/synthetic_soldiers.csv', index=False)
    df.to_csv(f'{data_dir}/synthetic_mood_reports.csv', index=False)
    print(f"Generated and saved {len(df)} records to {data_dir}/")
    return df

def engineer_features(df):
    print("Engineering features...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['soldier_id', 'date'])
    
    features = []
    for soldier_id, group in df.groupby('soldier_id'):
        group['sleep_avg_14d'] = group['sleep_hours'].rolling(window=14, min_periods=3).mean()
        group['stress_avg_14d'] = group['stress_level'].rolling(window=14, min_periods=3).mean()
        group['sleep_std_14d'] = group['sleep_hours'].rolling(window=14, min_periods=3).std()
        group['stress_std_14d'] = group['stress_level'].rolling(window=14, min_periods=3).std()
        group['reports_count_14d'] = group['sleep_hours'].rolling(window=14, min_periods=1).count()
        features.append(group)
    
    feature_df = pd.concat(features).dropna().reset_index(drop=True)
    print(f"Engineered {len(feature_df)} feature rows.")
    return feature_df

def train_model(feature_df):
    print("Training LightGBM model...")
    feature_names = [
        'sleep_avg_14d', 'stress_avg_14d', 
        'sleep_std_14d', 'stress_std_14d', 
        'reports_count_14d'
    ]
    target_name = 'high_risk_target'
    
    X = feature_df[feature_names]
    y = feature_df[target_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    lgb_clf.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100, verbose=False)])
    
    print("\n--- Model Evaluation ---")
    y_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]
    y_pred = lgb_clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test Set ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = 'ml/model.pkl'
    if not os.path.exists('ml'):
        os.makedirs('ml')
        
    with open(model_path, 'wb') as f:
        pickle.dump(lgb_clf, f)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    raw_data = generate_synthetic_data()
    feature_data = engineer_features(raw_data)
    train_model(feature_data)
