import sqlite3
import pandas as pd
import pickle
import json
import shap
import plotly
import plotly.graph_objects as go
from flask import (Flask, render_template, request, redirect, url_for, flash, g, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (LoginManager, UserMixin, login_user, logout_user, login_required, current_user)
from cryptography.fernet import Fernet
from functools import wraps
import datetime

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

def ensure_mood_logs_schema() -> None:
    try:
        db = get_db()
        columns = db.execute('PRAGMA table_info(mood_logs)').fetchall()
        existing_column_names = {col['name'] for col in columns}

        def add_column(column_def: str) -> None:
            try:
                db.execute(f'ALTER TABLE mood_logs ADD COLUMN {column_def}')
                db.commit()
            except Exception as e:
                print(f"Schema migration notice: {e}")

        if 'mood_rating' not in existing_column_names:
            add_column('mood_rating INTEGER DEFAULT 3')
        if 'stress_level' not in existing_column_names:
            add_column('stress_level INTEGER DEFAULT 3')
        if 'fatigue_level' not in existing_column_names:
            add_column('fatigue_level INTEGER DEFAULT 3')
        if 'sleep_quality' not in existing_column_names:
            add_column('sleep_quality INTEGER DEFAULT 3')
        if 'notes' not in existing_column_names:
            add_column('notes TEXT')
        if 'logged_by_user_id' not in existing_column_names:
            add_column('logged_by_user_id INTEGER')
        if 'created_at' not in existing_column_names:
            add_column('created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
    except Exception as e:
        print(f"Schema check failed (non-fatal): {e}")

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
            db.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",(username, password_hash, role))
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
            shap_for_label = shap_values[prediction_encoded]
            shap_explanation = {feature: float(value) for feature, value in zip(model_columns, shap_for_label)}
            shap_json = json.dumps(shap_explanation)

            age_val = int(input_df.iloc[0]['Age'])
            hr_val = int(input_df.iloc[0]['Heart Rate'])
            temp_val = float(input_df.iloc[0]['Body Temperature'])
            spo2_val = int(input_df.iloc[0]['SpO2'])
            steps_count_val = int(input_df.iloc[0]['Steps Count'])

            db = get_db()
            db.execute('INSERT INTO wellness_checks (soldier_id, age, heart_rate, body_temperature, spo2, steps_count, prediction_result, shap_values) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (soldier_id, age_val, encrypt_data(hr_val), encrypt_data(temp_val), encrypt_data(spo2_val), steps_count_val, prediction, shap_json))
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
        dec_check['created_at'] = datetime.datetime.strptime(check['created_at'], '%Y-%m-%d %H:%M:%S')
        history.append(dec_check)
        
    df = pd.DataFrame(history)
    df['created_at'] = pd.to_datetime(df['created_at'])

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(x=df['created_at'], y=df['heart_rate'].astype(float), mode='lines+markers', name='Heart Rate'))
    fig_metrics.add_trace(go.Scatter(x=df['created_at'], y=df['body_temperature'].astype(float), mode='lines+markers', name='Temperature', yaxis='y2'))
    fig_metrics.update_layout(
        title='Health Metrics Over Time', template='plotly_dark',
        yaxis=dict(title='Heart Rate (bpm)'),
        yaxis2=dict(title='Temp (°C)', overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graph_json_metrics = json.dumps(fig_metrics, cls=plotly.utils.PlotlyJSONEncoder)

    shap_graph_json = None
    latest_check = history[-1] if history else None
    if latest_check and latest_check['shap_values']:
        shap_df = pd.DataFrame(list(latest_check['shap_values'].items()), columns=['Feature', 'SHAP Value']).sort_values(by='SHAP Value', ascending=True)
        shap_df['Color'] = shap_df['SHAP Value'].apply(lambda x: '#FF6347' if x > 0 else '#4682B4')
        
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=shap_df['Color']
        ))
        fig_shap.update_layout(
            title=f"Why the prediction was '{latest_check['prediction_result']}' (Latest Check)",
            xaxis_title="Contribution to Threat Level (SHAP Value)",
            yaxis_title="Health Feature",
            template='plotly_dark',
            showlegend=False
        )
        shap_graph_json = json.dumps(fig_shap, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('soldier_detail.html', soldier_id=soldier_id, history=history, graph_json_metrics=graph_json_metrics, shap_graph_json=shap_graph_json)


@app.route('/mood-tracker', methods=['GET', 'POST'])
@login_required
def mood_tracker():
    ensure_mood_logs_schema()
    db = get_db()
    if request.method == 'POST':
        try:
            soldier_id = request.form['soldier_id']
            mood_rating = int(request.form['mood_rating'])
            stress_level = int(request.form['stress_level'])
            fatigue_level = int(request.form['fatigue_level'])
            sleep_quality = int(request.form['sleep_quality'])
            notes = request.form.get('notes', '')

            db.execute('''
                INSERT INTO mood_logs 
                (soldier_id, mood_rating, stress_level, fatigue_level, sleep_quality, notes, logged_by_user_id) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (soldier_id, mood_rating, stress_level, fatigue_level, sleep_quality, notes, current_user.id))
            
            db.commit()
            log_action("MOOD_LOG_ADD", f"Detailed mood log added for Soldier {soldier_id}.")
            flash(f"Mood log for Soldier {soldier_id} has been successfully recorded.", 'success')
            return redirect(url_for('mood_tracker', soldier_id=soldier_id))
        except Exception as e:
            log_action("ERROR", f"Error on mood tracker form submission: {e}")
            flash(f"An error occurred while logging mood: {e}", 'error')
    
    mood_graph_json = None
    notes = []
    selected_soldier_id = request.args.get('soldier_id')

    if selected_soldier_id:
        mood_data = db.execute('''
            SELECT created_at, mood_rating, stress_level, fatigue_level, sleep_quality, notes 
            FROM mood_logs WHERE soldier_id = ? ORDER BY created_at ASC
        ''', (selected_soldier_id,)).fetchall()

        if mood_data:
            rows = [dict(r) for r in mood_data]
            df = pd.DataFrame.from_records(rows)

            timestamps = None
            if 'created_at' in df.columns:
                timestamps = pd.to_datetime(df['created_at'], errors='coerce')
            x_values = timestamps if (timestamps is not None and timestamps.notna().any()) else list(range(1, len(df) + 1))

            for col in ['mood_rating', 'stress_level', 'fatigue_level', 'sleep_quality']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            def compute_wellness(row):
                stress_component = 6 - (row.get('stress_level', 3) or 3)
                fatigue_component = 6 - (row.get('fatigue_level', 3) or 3)
                sleep_component = row.get('sleep_quality', 3) or 3
                mood_component = row.get('mood_rating', 3) or 3
                total = stress_component + fatigue_component + sleep_component + mood_component
                return round((total / 20) * 100, 2)

            df['wellness_score'] = df.apply(compute_wellness, axis=1)
            fig = go.Figure()
            if 'stress_level' in df.columns:
                fig.add_trace(go.Scatter(x=x_values, y=df['stress_level'], mode='lines+markers', name='Stress Level'))
            if 'fatigue_level' in df.columns:
                fig.add_trace(go.Scatter(x=x_values, y=df['fatigue_level'], mode='lines+markers', name='Fatigue Level'))
            if 'sleep_quality' in df.columns:
                fig.add_trace(go.Scatter(x=x_values, y=df['sleep_quality'], mode='lines+markers', name='Sleep Quality'))
            if 'mood_rating' in df.columns:
                fig.add_trace(go.Scatter(x=x_values, y=df['mood_rating'], mode='lines+markers', name='Overall Mood'))

            fig.add_trace(go.Scatter(x=x_values, y=df['wellness_score'], mode='lines+markers', name='Wellness Score (0-100)', yaxis='y2'))

            fig.update_layout(
                title=f'Mental Wellness Trends for Soldier {selected_soldier_id}',
                xaxis_title='Date' if isinstance(x_values, pd.Series) else 'Entry #',
                yaxis=dict(title='Level (1-5)'),
                yaxis2=dict(title='Wellness (0-100)', overlaying='y', side='right', rangemode='tozero'),
                template='plotly_dark',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            mood_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            insights = []
            if len(df) >= 2:
                def trend(col):
                    if col in df.columns and df[col].notna().any():
                        return df[col].iloc[-1] - df[col].iloc[0]
                    return 0
                stress_trend = trend('stress_level')
                fatigue_trend = trend('fatigue_level')
                sleep_trend = trend('sleep_quality')
                mood_trend = trend('mood_rating')
                insights.append(f"Stress is {'up' if stress_trend>0 else 'down' if stress_trend<0 else 'stable'} by {abs(round(stress_trend,1))}.")
                insights.append(f"Fatigue is {'up' if fatigue_trend>0 else 'down' if fatigue_trend<0 else 'stable'} by {abs(round(fatigue_trend,1))}.")
                insights.append(f"Sleep quality is {'up' if sleep_trend>0 else 'down' if sleep_trend<0 else 'stable'} by {abs(round(sleep_trend,1))}.")
                insights.append(f"Mood is {'up' if mood_trend>0 else 'down' if mood_trend<0 else 'stable'} by {abs(round(mood_trend,1))}.")
            else:
                insights.append("Not enough historical data to compute trends yet. Log more entries.")

            latest = df.iloc[-1]
            latest_summary = {
                'mood_rating': latest.get('mood_rating', None),
                'stress_level': latest.get('stress_level', None),
                'fatigue_level': latest.get('fatigue_level', None),
                'sleep_quality': latest.get('sleep_quality', None),
                'wellness_score': latest.get('wellness_score', None)
            }

            if 'created_at' in df.columns and 'notes' in df.columns:
                notes = df[['created_at', 'notes']].to_dict('records')
            else:
                notes = []
        else:
            insights = []
            latest_summary = None

    all_soldiers = db.execute('SELECT DISTINCT soldier_id FROM mood_logs ORDER BY soldier_id ASC').fetchall()

    return render_template('mood_tracker.html', mood_graph_json=mood_graph_json, notes=notes, insights=locals().get('insights', []), latest_summary=locals().get('latest_summary', None), all_soldiers=all_soldiers, selected_soldier_id=selected_soldier_id)


@app.route('/soldier/<string:soldier_id>/mood_analysis')
@login_required
def mood_analysis(soldier_id):
    ensure_mood_logs_schema()
    db = get_db()
    mood_data = db.execute('''
        SELECT created_at, mood_rating, stress_level, fatigue_level, sleep_quality, notes 
        FROM mood_logs 
        WHERE soldier_id = ? 
        ORDER BY created_at ASC
    ''', (soldier_id,)).fetchall()

    if not mood_data:
        flash('No mood data found for this soldier to analyze.', 'warning')
        return redirect(url_for('soldier_detail', soldier_id=soldier_id))

    rows = [dict(r) for r in mood_data]
    df = pd.DataFrame.from_records(rows)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['created_at'], y=df['stress_level'], mode='lines+markers', name='Stress Level'))
    fig.add_trace(go.Scatter(x=df['created_at'], y=df['fatigue_level'], mode='lines+markers', name='Fatigue Level'))
    fig.add_trace(go.Scatter(x=df['created_at'], y=df['sleep_quality'], mode='lines+markers', name='Sleep Quality'))

    fig.update_layout(
        title=f'Mental Wellness Trends for Soldier {soldier_id}',
        xaxis_title='Date',
        yaxis_title='Level (1-5)',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    mood_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    notes_records = df[['created_at', 'notes']].to_dict('records') if 'created_at' in df.columns and 'notes' in df.columns else []
    return render_template('mood_analysis.html', soldier_id=soldier_id, mood_graph_json=mood_graph_json, notes=notes_records)

@app.route('/audit-log')
@login_required
@role_required('doctor')
def audit_log():
    db = get_db()
    logs = db.execute('SELECT a.timestamp, u.username, a.action, a.details FROM audit_logs a LEFT JOIN users u ON u.id = a.user_id ORDER BY a.timestamp DESC').fetchall()
    return render_template('audit_log.html', logs=logs)

@app.cli.command('init-db')
def init_db_command():
    db = sqlite3.connect(DATABASE)
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    db.close()
    print('Database has been initialized.')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')