import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-hard-to-guess-string'
    ENCRYPTION_KEY = b'w8JHDj6AYcuNHsNljZwi34-FuUTXZ7gCQtv9QsneyS0='
    SQLALCHEMY_DATABASE_URI = 'sqlite:///soldier_wellness.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False