import uuid
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Enum as PyEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()

class Role(str, enum.Enum):
    soldier = "soldier"
    commander = "commander"

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(PyEnum(Role), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Soldier(Base):
    __tablename__ = "soldiers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    unit = Column(String, index=True)
    
    mood_reports = relationship("MoodReport", back_populates="soldier")
    risk_scores = relationship("RiskScore", back_populates="soldier", order_by="desc(RiskScore.ts)")

class MoodReport(Base):
    __tablename__ = "mood_reports"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    soldier_id = Column(UUID(as_uuid=True), ForeignKey('soldiers.id'), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    stress_level = Column(Integer) # 1-5
    sleep_hours = Column(Float)
    
    soldier = relationship("Soldier", back_populates="mood_reports")

class RiskScore(Base):
    __tablename__ = "risk_scores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    soldier_id = Column(UUID(as_uuid=True), ForeignKey('soldiers.id'), nullable=False)
    ts = Column(DateTime(timezone=True), server_default=func.now())
    score = Column(Float, nullable=False) 
    label = Column(String, nullable=False)
    
    soldier = relationship("Soldier", back_populates="risk_scores")

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    soldier_id = Column(UUID(as_uuid=True), ForeignKey('soldiers.id'), nullable=False)
    message = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Integer, default=1)
    
    soldier = relationship("Soldier")
