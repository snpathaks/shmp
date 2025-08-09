from pydantic import BaseModel, Field

class HealthReportCreate(BaseModel):
    soldier_id: str = Field(..., example = "SGT-117")
    sleep_hours: float = Field(..., gt = 0, lt = 16, example = 7.5)
    hrv_score: int = Field(..., gt = 0, example = 55)
    stress_level: int = Field(..., ge = 1, le = 10, example = 4)
    activity_calories: int = Field(..., gt = 0, example = 800)

class HealthReportDisplay(BaseModel):
    soldier_id: str
    risk_probability: float
    risk_level: str
    risk_color: str