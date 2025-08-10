# Soldier Health Monitoring Platform (SHMP)
A secure, scalable system designed to monitor soldier health, behavior, and mission performance, predicting injury, fatigue, or psychological risk using a state-of-the-art machine learning pipeline.

## High-Level Architecture
This MVP uses a straightforward, containerized setup:

- Web (FastAPI): The backend API serving data and web pages.
- PostgreSQL: The database for all application data.

All services are managed with a simple `docker-compose.yml` file.

## Core Features
- Secure Authentication: JWT-based login with password hashing and basic roles (e.g., 'commander', 'soldier').

- Simple Offline Data Entry: A mobile-friendly web page that saves health and mood reports to the browser's local storage (IndexedDB). A manual "Sync" button sends the data to the server when online.

- Effective ML Model: A LightGBM (Gradient Boosted Tree) model predicts risk scores. This is a powerful and widely-used model that's much simpler than a deep learning pipeline.

- Clear Dashboard: A clean dashboard for commanders to view a list of their soldiers and their latest risk status.

- Basic Alerting: High-risk scores automatically create simple alerts in the system.

## Tech Stack
- Backend: Python, FastAPI
- Database: PostgreSQL (with SQLAlchemy)
- Frontend: Jinja2 templates, TailwindCSS, Alpine.js
- Offline Storage: Dexie.js (a simple wrapper for IndexedDB)
- ML: scikit-learn, LightGBM, Pandas
- Containerization: Docker, Docker Compose