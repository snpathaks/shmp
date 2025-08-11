from flask import Flask, request, jsonify, render_template
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import hashlib
import os

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# --- Database Configuration ---
DATABASE = 'soldier_wellness.db'

def get_db():
    """Connect to the SQLite database."""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize the database with the schema."""
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# --- Machine Learning Model Loading ---
try:
    knn_model = joblib.load('knn_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    lgbm_model = joblib.load('lgbm_model.pkl') # Load new model
    print("ML models loaded successfully.")
except FileNotFoundError:
    print("Warning: Model files not found. Risk prediction will not work.")
    print("Run model.py to generate the models.")
    knn_model, rf_model, svm_model, lgbm_model = None, None, None, None


# --- Helper Functions ---
def log_audit(user, action, details):
    db = get_db()
    db.execute('INSERT INTO audit_log (user, action, details) VALUES (?, ?, ?)', (user, action, details))
    db.commit()

# --- Routes ---

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/soldiers', methods=['GET'])
def get_soldiers():
    db = get_db()
    soldiers_cursor = db.execute('SELECT id, name, rank, unit, role, risk_level FROM soldiers ORDER BY name')
    soldiers = soldiers_cursor.fetchall()
    return jsonify([dict(s) for s in soldiers])

# NEW: Endpoint to get detailed data for one soldier
@app.route('/api/soldier_details/<int:soldier_id>', methods=['GET'])
def get_soldier_details(soldier_id):
    db = get_db()
    # Fetch basic info
    soldier_cursor = db.execute('SELECT id, name, rank, unit, role, risk_level FROM soldiers WHERE id = ?', (soldier_id,))
    soldier = soldier_cursor.fetchone()
    if not soldier:
        return jsonify({'error': 'Soldier not found'}), 404

    # Fetch historical health data
    health_cursor = db.execute(
        'SELECT timestamp, sleep_hours, stress_level, readiness_score FROM health_data WHERE soldier_id = ? ORDER BY timestamp DESC LIMIT 30', (soldier_id,)
    )
    health_history = [dict(row) for row in health_cursor.fetchall()]
    
    # Fetch wellness reports
    wellness_cursor = db.execute(
        'SELECT timestamp, mood, fatigue, comments FROM wellness_reports WHERE soldier_id = ? ORDER BY timestamp DESC LIMIT 10', (soldier_id,)
    )
    wellness_history = [dict(row) for row in wellness_cursor.fetchall()]

    return jsonify({
        'info': dict(soldier),
        'health_history': health_history,
        'wellness_history': wellness_history
    })

@app.route('/api/add_soldier', methods=['POST'])
def add_soldier():
    data = request.get_json()
    name, rank, unit = data.get('name'), data.get('rank'), data.get('unit')
    if not all([name, rank, unit]):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    db = get_db()
    try:
        db.execute('INSERT INTO soldiers (name, rank, unit) VALUES (?, ?, ?)', (name, rank, unit))
        db.commit()
        log_audit('System', 'ADD_SOLDIER', f'Added: {name}, Rank: {rank}')
        return jsonify({'status': 'success', 'message': 'Soldier added'})
    except sqlite3.IntegrityError:
        return jsonify({'status': 'error', 'message': 'Soldier name must be unique'}), 409

@app.route('/api/log_data', methods=['POST'])
def log_data():
    data = request.get_json()
    # Add new fields
    payload = {
        'soldier_id': data.get('soldier_id'),
        'sleep_hours': data.get('sleep_hours'),
        'stress_level': data.get('stress_level'),
        'readiness_score': data.get('readiness_score'),
        'physical_exertion': data.get('physical_exertion'),
        'hydration_level': data.get('hydration_level')
    }
    if not all(payload.values()):
        return jsonify({'status': 'error', 'message': 'Missing data fields'}), 400

    db = get_db()
    db.execute(
        'INSERT INTO health_data (soldier_id, sleep_hours, stress_level, readiness_score, physical_exertion, hydration_level) VALUES (?, ?, ?, ?, ?, ?)',
        tuple(payload.values())
    )
    db.commit()
    log_audit('System', 'LOG_DATA', f'Logged health data for soldier ID: {payload["soldier_id"]}')
    return jsonify({'status': 'success', 'message': 'Data logged successfully'})

# NEW: Endpoint for soldiers to submit wellness reports
@app.route('/api/log_wellness', methods=['POST'])
def log_wellness():
    data = request.get_json()
    payload = {
        'soldier_id': data.get('soldier_id'),
        'mood': data.get('mood'),
        'fatigue': data.get('fatigue'),
        'comments': data.get('comments', '')
    }
    if not all([payload['soldier_id'], payload['mood'], payload['fatigue']]):
        return jsonify({'status': 'error', 'message': 'Missing wellness fields'}), 400
    
    db = get_db()
    db.execute(
        'INSERT INTO wellness_reports (soldier_id, mood, fatigue, comments) VALUES (?, ?, ?, ?)',
        (payload['soldier_id'], payload['mood'], payload['fatigue'], payload['comments'])
    )
    db.commit()
    log_audit(f"SoldierID:{payload['soldier_id']}", 'LOG_WELLNESS', 'Submitted wellness report.')
    return jsonify({'status': 'success', 'message': 'Wellness report submitted'})

@app.route('/api/risk_prediction/<int:soldier_id>', methods=['GET'])
def predict_risk(soldier_id):
    if not all([knn_model, rf_model, svm_model, lgbm_model]):
        return jsonify({'error': 'ML models are not loaded'}), 500

    db = get_db()
    data_cursor = db.execute(
        'SELECT sleep_hours, stress_level, readiness_score, physical_exertion, hydration_level FROM health_data WHERE soldier_id = ? ORDER BY timestamp DESC LIMIT 1',
        (soldier_id,)
    )
    latest_data = data_cursor.fetchone()

    if not latest_data:
        return jsonify({'risk_level': 'N/A', 'message': 'No data for prediction.'})

    features = pd.DataFrame([dict(latest_data)])
    
    try:
        knn_pred = knn_model.predict(features)[0]
        rf_pred = rf_model.predict(features)[0]
        svm_pred = svm_model.predict(features)[0]
        lgbm_pred = lgbm_model.predict(features)[0] # Predict with new model
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    predictions = [knn_pred, rf_pred, svm_pred, lgbm_pred]
    final_prediction = max(set(predictions), key=predictions.count)
    
    risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = risk_map.get(final_prediction, 'Unknown')

    db.execute('UPDATE soldiers SET risk_level = ? WHERE id = ?', (risk_level, soldier_id))
    
    # NEW: Create an alert if risk is High
    if risk_level == 'High':
        details = f"High risk detected. Models (KNN, RF, SVM, LGBM): {risk_map.get(knn_pred)}, {risk_map.get(rf_pred)}, {risk_map.get(svm_pred)}, {risk_map.get(lgbm_pred)}"
        db.execute('INSERT INTO alerts (soldier_id, risk_level, details) VALUES (?, ?, ?)', (soldier_id, risk_level, details))
        log_audit('ML_System', 'CREATE_ALERT', f'High risk alert for soldier ID {soldier_id}')

    db.commit()
    log_audit('ML_System', 'RISK_PREDICTION', f'Predicted risk for soldier ID {soldier_id}: {risk_level}')

    return jsonify({'risk_level': risk_level, 'details': {
        'knn': risk_map.get(knn_pred),
        'random_forest': risk_map.get(rf_pred),
        'svm': risk_map.get(svm_pred),
        'lightgbm': risk_map.get(lgbm_pred)
    }})

# NEW: Endpoints for alerts
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    db = get_db()
    alerts_cursor = db.execute('''
        SELECT a.id, a.timestamp, a.risk_level, a.details, a.is_acknowledged, s.name, s.rank, s.unit
        FROM alerts a
        JOIN soldiers s ON a.soldier_id = s.id
        WHERE a.is_acknowledged = 0
        ORDER BY a.timestamp DESC
    ''')
    alerts = [dict(row) for row in alerts_cursor.fetchall()]
    return jsonify(alerts)

@app.route('/api/alerts/acknowledge/<int:alert_id>', methods=['POST'])
def acknowledge_alert(alert_id):
    db = get_db()
    db.execute('UPDATE alerts SET is_acknowledged = 1 WHERE id = ?', (alert_id,))
    db.commit()
    log_audit('Commander', 'ACKNOWLEDGE_ALERT', f'Alert ID {alert_id} acknowledged.')
    return jsonify({'status': 'success'})

@app.route('/api/unit_wellness', methods=['GET'])
def get_unit_wellness():
    db = get_db()
    wellness_cursor = db.execute('''
        SELECT unit,
            SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_risk_count,
            SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END) as medium_risk_count,
            SUM(CASE WHEN risk_level = 'Low' THEN 1 ELSE 0 END) as low_risk_count,
            COUNT(id) as total_soldiers
        FROM soldiers WHERE role = 'Soldier'
        GROUP BY unit
    ''')
    return jsonify([dict(row) for row in wellness_cursor.fetchall()])

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        print("Database not found. Initializing...")
        init_db()
    model_files = ['knn_model.pkl', 'rf_model.pkl', 'svm_model.pkl', 'lgbm_model.pkl']
    if not all([os.path.exists(f) for f in model_files]):
         print("ML models not found. Please run model.py to create them.")
    app.run(debug=True, port=5001) # Use a different port to avoid conflicts