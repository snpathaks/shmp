import sqlite3
from werkzeug.security import generate_password_hash

DATABASE = 'soldier_wellness.db'

def add_user():
    print("--- Create a New User ---")
    username = input("Enter username: ")
    password = input("Enter password: ")
    role = input("Enter role (doctor/analyst): ")

    if not username or not password or role not in ['doctor', 'analyst']:
        print("Invalid input. Aborting.")
        return

    password_hash = generate_password_hash(password)
    try:
        con = sqlite3.connect(DATABASE)
        cur = con.cursor()
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, password_hash, role))
        con.commit()
        con.close()
        print(f"\nSUCCESS: User '{username}' with role '{role}' was created.")
        print("You may now run the Flask application and log in.")
    except sqlite3.IntegrityError:
        print(f"\nERROR: Username '{username}' already exists.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    add_user()