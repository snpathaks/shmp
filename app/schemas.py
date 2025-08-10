import uuid
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TunedModel(BaseModel):
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class MoodReportBase(TunedModel):
    soldier_id: uuid.UUID
    date: datetime
    stress_level: int = Field(..., ge=1, le=5)
    sleep_hours: float = Field(..., ge=0, le=24)

class MoodReportCreate(MoodReportBase):
    pass

class MoodReportPublic(MoodReportBase):
    id: uuid.UUID

class SyncPayload(BaseModel):
    mood_reports: List[MoodReportCreate] = []

class RiskScoreBase(TunedModel):
    soldier_id: uuid.UUID
    score: float
    label: str

class RiskScoreCreate(RiskScoreBase):
    pass

class RiskScorePublic(RiskScoreBase):
    id: uuid.UUID
    ts: datetime

class SoldierBase(TunedModel):
    id: uuid.UUID
    name: str
    unit: str

class SoldierPublic(SoldierBase):
    latest_risk_score: Optional[RiskScorePublic] = None

class AlertPublic(TunedModel):
    id: uuid.UUID
    soldier: SoldierPublic
    message: str
    created_at: datetime