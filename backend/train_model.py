import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

def train():
    df = pd.read_csv('backend/synthetic_health_data.csv')
    
    features = ['sleep_hours', 'hrv_score', 'stress_level', 'activity_calories']
    target = 'is_at_risk'
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    joblib.dump(model, 'backend/risk_model.pkl')
    print("Model trained and saved to backend/risk_model.pkl")

if __name__ == '__main__':
    train()