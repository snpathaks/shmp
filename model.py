import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

def create_and_train_models():
    """
    Generates enhanced synthetic data, trains KNN, Random Forest, SVM, and LightGBM models,
    and saves them to disk.
    """
    print("Generating enhanced synthetic soldier data...")
    # Create a more detailed synthetic dataset
    np.random.seed(42)
    num_records = 2000
    data = {
        'sleep_hours': np.random.uniform(3, 9, num_records),
        'stress_level': np.random.randint(1, 11, num_records), # Scale of 1-10
        'readiness_score': np.random.randint(40, 100, num_records), # Scale of 0-100
        'physical_exertion': np.random.randint(1, 11, num_records), # Scale of 1-10
        'hydration_level': np.random.uniform(0.5, 1.5, num_records) # 1.0 is optimal
    }
    df = pd.DataFrame(data)

    # Define risk with more complex logic
    # High risk: low sleep, high stress, high exertion
    # Medium risk: moderate levels or one severe indicator
    # Low risk: good sleep, low stress, normal exertion
    risk_score = (10 - df['stress_level']) + (df['sleep_hours'] * 1.5) + (df['readiness_score'] / 10) - (df['physical_exertion']) + (1 - abs(1 - df['hydration_level'])) * 5
    
    conditions = [
        risk_score < 10,  # High risk
        risk_score < 15, # Medium risk
    ]
    # Risk Levels: 2 = High, 1 = Medium, 0 = Low
    risk_levels = [2, 1]
    df['risk_level'] = np.select(conditions, risk_levels, default=0)

    print("Synthetic data generated:")
    print(df.head())
    print("\nRisk level distribution:")
    print(df['risk_level'].value_counts(normalize=True))

    # --- Model Training ---
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. K-Nearest Neighbors (KNN)
    print("\nTraining KNN model...")
    knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    knn_pipeline.fit(X_train, y_train)
    print(f"KNN Test Accuracy: {knn_pipeline.score(X_test, y_test):.3f}")
    joblib.dump(knn_pipeline, 'knn_model.pkl')
    print("KNN model saved as knn_model.pkl")

    # 2. Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    print(f"Random Forest Test Accuracy: {rf_model.score(X_test, y_test):.3f}")
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Random Forest model saved as rf_model.pkl")
    
    # 3. Support Vector Machine (SVM)
    print("\nTraining SVM model...")
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    svm_pipeline.fit(X_train, y_train)
    print(f"SVM Test Accuracy: {svm_pipeline.score(X_test, y_test):.3f}")
    joblib.dump(svm_pipeline, 'svm_model.pkl')
    print("SVM model saved as svm_model.pkl")

    # 4. LightGBM
    print("\nTraining LightGBM model...")
    lgbm_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    lgbm_model.fit(X_train, y_train)
    print(f"LightGBM Test Accuracy: {lgbm_model.score(X_test, y_test):.3f}")
    joblib.dump(lgbm_model, 'lgbm_model.pkl')
    print("LightGBM model saved as lgbm_model.pkl")

    print("\nAll models have been trained and saved successfully.")

if __name__ == '__main__':
    create_and_train_models()
