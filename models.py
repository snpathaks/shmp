from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

class HealthReport(Base):
    __tablename__ = "health_reports"

    id = Column(Integer, primary_key=True, index=True)
    soldier_id = Column(String, index=True, nullable=False)
    sleep_hours = Column(Float, nullable=False)
    hrv_score = Column(Integer, nullable=False)
    stress_level = Column(Integer, nullable=False)
    activity_calories = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())