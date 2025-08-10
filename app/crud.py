from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from typing import List
from datetime import datetime
import uuid

from . import models, schemas

def get_user_by_username(db: Session, username: str) -> models.User:
    return db.query(models.User).filter(models.User.username == username).first()

def get_soldiers_with_risk_scores(db: Session, skip: int = 0, limit: int = 100) -> List[models.Soldier]:
    return (
        db.query(models.Soldier)
        .options(joinedload(models.Soldier.risk_scores))
        .order_by(models.Soldier.name)
        .offset(skip)
        .limit(limit)
        .all()
    )

def create_mood_report(db: Session, report: schemas.MoodReportCreate) -> models.MoodReport:
    db_report = models.MoodReport(**report.model_dump())
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_mood_reports_for_soldier_daterange(
    db: Session, soldier_id: uuid.UUID, start_date: datetime, end_date: datetime) -> List[models.MoodReport]:
    return (
        db.query(models.MoodReport)
        .filter(
            models.MoodReport.soldier_id == soldier_id,
            models.MoodReport.date >= start_date,
            models.MoodReport.date <= end_date,
        )
        .order_by(models.MoodReport.date)
        .all()
    )

def create_risk_score(db: Session, score: schemas.RiskScoreCreate) -> models.RiskScore:
    db_score = models.RiskScore(**score.model_dump())
    db.add(db_score)
    db.commit()
    db.refresh(db_score)
    return db_score

def create_alert(db: Session, soldier_id: uuid.UUID, message: str) -> models.Alert:
    db_alert = models.Alert(soldier_id=soldier_id, message=message)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_active_alerts(db: Session, limit: int = 10) -> List[models.Alert]:
    return (
        db.query(models.Alert)
        .options(joinedload(models.Alert.soldier))
        .filter(models.Alert.is_active == 1)
        .order_by(desc(models.Alert.created_at))
        .limit(limit)
        .all()
    )
