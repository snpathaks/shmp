DROP TABLE IF EXISTS soldiers;
DROP TABLE IF EXISTS health_data;
DROP TABLE IF EXISTS mission_logs;
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS wellness_reports;
DROP TABLE IF EXISTS alerts;

CREATE TABLE soldiers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    rank TEXT NOT NULL,
    unit TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'Soldier',
    risk_level TEXT DEFAULT 'N/A',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE health_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sleep_hours REAL NOT NULL,
    stress_level INTEGER NOT NULL,
    readiness_score INTEGER NOT NULL,
    physical_exertion INTEGER,
    hydration_level REAL,
    FOREIGN KEY (soldier_id) REFERENCES soldiers (id) ON DELETE CASCADE
);

CREATE TABLE mission_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mission_name TEXT DEFAULT 'Routine Patrol',
    performance_score REAL NOT NULL,
    notes TEXT,
    FOREIGN KEY (soldier_id) REFERENCES soldiers (id) ON DELETE CASCADE
);

CREATE TABLE wellness_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mood INTEGER NOT NULL, -- e.g.,
    fatigue INTEGER NOT NULL,
    comments TEXT,
    FOREIGN KEY (soldier_id) REFERENCES soldiers (id) ON DELETE CASCADE
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    soldier_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_level TEXT NOT NULL,
    details TEXT NOT NULL,
    is_acknowledged INTEGER DEFAULT 0,
    FOREIGN KEY (soldier_id) REFERENCES soldiers (id) ON DELETE CASCADE
);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT
);

INSERT INTO soldiers (name, rank, unit, role) VALUES
('David Miller', 'Captain', 'HQ', 'Commander');

INSERT INTO soldiers (name, rank, unit) VALUES
('John Doe', 'Sergeant', 'Alpha Company'),
('Jane Smith', 'Corporal', 'Alpha Company'),
('Mike Ross', 'Private', 'Bravo Company'),
('Sarah Connor', 'Sergeant', 'Bravo Company'),
('Alex Ray', 'Private', 'Alpha Company');

INSERT INTO health_data (soldier_id, sleep_hours, stress_level, readiness_score, physical_exertion, hydration_level) VALUES
(2, 7.5, 4, 88, 5, 1.1);
