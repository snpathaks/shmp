import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline

print("Starting the unified training pipeline...")

try:
    df = pd.read_csv('training\wellness_data.csv')
    print("Dataset 'wellness_data.csv' loaded.")
except FileNotFoundError:
    print("\nFATAL ERROR: 'wellness_data.csv' not found.")
    exit()

X = df[['Age', 'Heart Rate', 'Body Temperature', 'SpO2', 'Steps Count']]
y = df['ThreatLevel']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Labels encoded. Classes: {list(label_encoder.classes_)}. Encoder saved.")

with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Model columns saved for prediction ordering.")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
print("Data split into training and testing sets.")
print("Defining individual model pipelines...")
clf1 = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3))])
clf2 = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=42))])
clf3 = RandomForestClassifier(random_state=42)
clf4 = lgb.LGBMClassifier(random_state=42)

ensemble_clf = VotingClassifier(
    estimators=[('knn', clf1), ('svm', clf2), ('rf', clf3), ('lgbm', clf4)],
    voting='soft'
)
print("Ensemble Voting Classifier defined.")
print("Training the complete ensemble model on data arrays (without feature names)...")
ensemble_clf.fit(X_train.values, y_train)
print("Training complete.")

accuracy = ensemble_clf.score(X_test.values, y_test)
print(f"\nModel accuracy on test data: {accuracy:.2%}")

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_clf, f)
print("Warning-free 'ensemble_model.pkl' has been created and saved.")