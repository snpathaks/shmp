# soldier-health-monitoring/app.py
# FINAL, COMPLETE, AND CORRECTED VERSION

import sqlite3
import pandas as pd
import pickle
import json
import shap
import plotly
import plotly.graph_objects as go
from flask import (Flask, render_template, request, redirect, url_for, flash, g,
                   abort)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                       login_required, current_user)
from cryptography.fernet import Fernet
from functools import wraps
import datetime

# --- App Configuration ---
app = Flask(__name__)
try:
    app.config.from_object('config.Config')
except ImportError:
    print("FATAL: config.py not found. Please create it.")
    exit()

# --- Encryption Setup ---
try:
    cipher = Fernet(app.config['ENCRYPTION_KEY'])
except Exception:
    print("FATAL: Invalid ENCRYPTION_KEY in config.py.")
    cipher = None

# --- Database Setup & Utilities ---
DATABASE = 'soldier_wellness.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- ML Model Loading ---
try:
    with open('ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    # Using the correct path for training data
    X_train_sample = pd.read_csv('training/wellness_data.csv')[model_columns].sample(3, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, X_train_sample.values)
    print("ML Models and SHAP Explainer loaded successfully.")
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    model, label_encoder, explainer = None, None, None


# --- User Authentication & Security ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, password_hash, role):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user_data = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if user_data:
        return User(id=user_data['id'], username=user_data['username'], password_hash=user_data['password_hash'], role=user_data['role'])
    return None

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                log_action("ACCESS_DENIED", f"User '{current_user.username}' attempted to access a '{role}'-only page.")
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- Utility Functions ---
def encrypt_data(data):
    if not cipher or data is None: return data
    return cipher.encrypt(str(data).encode()).decode()

def decrypt_data(encrypted_data):
    if not cipher or encrypted_data is None: return "Decryption Error"
    try:
        return cipher.decrypt(encrypted_data.encode()).decode()
    except:
        return "Invalid Data"

def log_action(action, details=""):
    try:
        db = get_db()
        user_id = current_user.id if current_user.is_authenticated else None
        db.execute('INSERT INTO audit_logs (user_id, action, details) VALUES (?, ?, ?)',
                   (user_id, action, details))
        db.commit()
    except Exception as e:
        print(f"Audit log failed: {e}")

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user_data = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(id=user_data['id'], username=user_data['username'], password_hash=user_data['password_hash'], role=user_data['role'])
            login_user(user)
            log_action("LOGIN_SUCCESS", f"User '{username}' logged in.")
            flash('Login successful.', 'success')
            return redirect(url_for('index'))
        else:
            log_action("LOGIN_FAIL", f"Failed login attempt for username '{username}'.")
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        password_hash = generate_password_hash(password)
        try:
            db = get_db()
            db.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       (username, password_hash, role))
            db.commit()
            flash(f"User '{username}' created successfully. Please log in.", 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash(f"Username '{username}' already exists.", 'error')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    log_action("LOGOUT", f"User '{current_user.username}' logged out.")
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        try:
            soldier_id = request.form['soldier_id']
            input_data_dict = {
                'Age': [int(request.form['age'])],
                'Heart Rate': [int(request.form['heart_rate'])],
                'Body Temperature': [float(request.form['body_temperature'])],
                'SpO2': [int(request.form['spo2'])],
                'Steps Count': [int(request.form['steps_count'])]
            }
            input_df = pd.DataFrame.from_dict(input_data_dict)[model_columns]
            prediction_proba = model.predict_proba(input_df.values)[0]
            prediction_encoded = prediction_proba.argmax()
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]
            shap_values = explainer.shap_values(input_df.values)
            shap_explanation = dict(zip(model_columns, shap_values[prediction_encoded][0]))
            shap_json = json.dumps(shap_explanation)
            db = get_db()
            db.execute('INSERT INTO wellness_checks (soldier_id, age, heart_rate, body_temperature, spo2, steps_count, prediction_result, shap_values) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                       (soldier_id, int(input_df['Age']), encrypt_data(int(input_df['Heart Rate'])), encrypt_data(float(input_df['Body Temperature'])), encrypt_data(int(input_df['SpO2'])), int(input_df['Steps Count']), prediction, shap_json))
            db.commit()
            log_action("WELLNESS_CHECK_ADD", f"Added check for {soldier_id}. Threat: {prediction}")
            flash(f'Wellness check for Soldier {soldier_id} added. Threat Level: {prediction}', 'success')
        except Exception as e:
            flash(f'An error occurred: {e}', 'error')
            log_action("ERROR", f"Error on index form submission: {e}")
        return redirect(url_for('index'))
    db = get_db()
    soldiers = db.execute('SELECT soldier_id, MAX(created_at) as last_check_in, prediction_result FROM wellness_checks GROUP BY soldier_id ORDER BY last_check_in DESC').fetchall()
    return render_template('index.html', soldiers=soldiers)

@app.route('/soldier/<string:soldier_id>')
@login_required
def soldier_detail(soldier_id):
    db = get_db()
    checks = db.execute('SELECT * FROM wellness_checks WHERE soldier_id = ? ORDER BY created_at ASC', (soldier_id,)).fetchall()
    if not checks:
        flash('No data found for this soldier.', 'error')
        return redirect(url_for('index'))
    history = []
    for check in checks:
        dec_check = dict(check)
        dec_check['heart_rate'] = decrypt_data(check['heart_rate'])
        dec_check['body_temperature'] = decrypt_data(check['body_temperature'])
        dec_check['spo2'] = decrypt_data(check['spo2'])
        dec_check['shap_values'] = json.loads(check['shap_values']) if check['shap_values'] else {}
        history.append(dec_check)
    df = pd.DataFrame(history)
    df['created_at'] = pd.to_datetime(df['created_at'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['created_at'], y=df['heart_rate'].astype(float), mode='lines+markers', name='Heart Rate'))
    fig.add_trace(go.Scatter(x=df['created_at'], y=df['body_temperature'].astype(float), mode='lines+markers', name='Temperature', yaxis='y2'))
    fig.update_layout(title='Health Metrics Over Time', template='plotly_dark', yaxis=dict(title='Heart Rate (bpm)'), yaxis2=dict(title='Temp (°C)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('soldier_detail.html', soldier_id=soldier_id, history=history, graph_json=graph_json)

@app.route('/mood-tracker', methods=['GET', 'POST'])
@login_required
def mood_tracker():
    db = get_db()
    if request.method == 'POST':
        try:
            soldier_id = request.form['soldier_id']
            mood_rating = int(request.form['mood_rating'])
            notes = request.form.get('notes', '')
            db.execute('INSERT INTO mood_logs (soldier_id, mood_rating, notes, logged_by_user_id) VALUES (?, ?, ?, ?)', (soldier_id, mood_rating, notes, current_user.id))
            db.commit()
            log_action("MOOD_LOG_ADD", f"Mood log added for Soldier {soldier_id}.")
            flash(f"Mood log for Soldier {soldier_id} has been successfully recorded.", 'success')
        except Exception as e:
            log_action("ERROR", f"Error on mood tracker form submission: {e}")
            flash(f"An error occurred while logging mood: {e}", 'error')
        return redirect(url_for('mood_tracker'))
    moods = db.execute('SELECT m.created_at, m.soldier_id, m.mood_rating, m.notes, u.username FROM mood_logs m JOIN users u ON m.logged_by_user_id = u.id ORDER BY m.created_at DESC LIMIT 20').fetchall()
    return render_template('mood_tracker.html', moods=moods)

@app.route('/audit-log')
@login_required
@role_required('doctor')
def audit_log():
    db = get_db()
    logs = db.execute('SELECT a.timestamp, u.username, a.action, a.details FROM audit_logs a LEFT JOIN users u ON u.id = a.user_id ORDER BY a.timestamp DESC').fetchall()
    return render_template('audit_log.html', logs=logs)

# Command to initialize the database
@app.cli.command('init-db')
def init_db_command():
    """Clears the existing data and creates new tables."""
    db = sqlite3.connect(DATABASE)
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    db.close()
    print('Database has been initialized.')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')