from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., regex="^(soldier|commander)$")
    soldier_id: Optional[str] = None
    unit: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    soldier_id: Optional[str] = None
    unit: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class HealthReportCreate(BaseModel):
    soldier_id: str = Field(..., example="SGT-117")
    sleep_hours: float = Field(..., gt=0, lt=16, example=7.5)
    hrv_score: int = Field(..., gt=0, example=55)
    stress_level: int = Field(..., ge=1, le=10, example=4)
    activity_calories: int = Field(..., gt=0, example=800)

class HealthReportDisplay(BaseModel):
    id: int
    soldier_id: str
    sleep_hours: float
    hrv_score: int
    stress_level: int
    activity_calories: int
    risk_probability: float
    risk_level: str
    risk_color: str
    timestamp: datetime

    class Config:
        from_attributes = True

class DashboardData(BaseModel):
    reports: List[HealthReportDisplay]
    user_info: UserResponse