DROP TABLE IF EXISTS wellness_checks;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS mood_logs;
DROP TABLE IF EXISTS alerts;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'analyst'
);

INSERT INTO users (username, password_hash, role) VALUES 
('admin', 'pbkdf2:sha256:260000$Csm8Hp9vI8PmIxzG$6127c35db8be936e489e2ebcc6e2e50f30734236aa876ef63b2ce88354de242b', 'doctor');

CREATE TABLE wellness_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id TEXT NOT NULL,
    age INTEGER,
    heart_rate TEXT, 
    body_temperature TEXT,
    spo2 TEXT,
    steps_count INTEGER,
    prediction_result TEXT NOT NULL,
    shap_values TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE mood_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id TEXT NOT NULL,
    mood_rating INTEGER NOT NULL, -- Overall mood (1-5)
    stress_level INTEGER NOT NULL DEFAULT 3, -- Specific stress level (1-5)
    fatigue_level INTEGER NOT NULL DEFAULT 3, -- Specific fatigue level (1-5)
    sleep_quality INTEGER NOT NULL DEFAULT 3, -- Sleep quality (1-5)
    notes TEXT,
    logged_by_user_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (logged_by_user_id) REFERENCES users (id)
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id TEXT NOT NULL,
    alert_message TEXT NOT NULL,
    is_acknowledged BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_by_user_id INTEGER,
    acknowledged_at TIMESTAMP,
    FOREIGN KEY (acknowledged_by_user_id) REFERENCES users (id)
);