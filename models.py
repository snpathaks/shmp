from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
import enum

class UserRole(str, enum.Enum):
    soldier = "soldier"
    commander = "commander"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    soldier_id = Column(String, unique=True, index=True, nullable=True)
    unit = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    health_reports = relationship("HealthReport", back_populates="user")

class HealthReport(Base):
    __tablename__ = "health_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    soldier_id = Column(String, index=True, nullable=False)
    sleep_hours = Column(Float, nullable=False)
    hrv_score = Column(Integer, nullable=False)
    stress_level = Column(Integer, nullable=False)
    activity_calories = Column(Integer, nullable=False)
    risk_probability = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    risk_color = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="health_reports")