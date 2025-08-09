from datetime import timedelta
from typing import List, Optional
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from schemas import (
    HealthReportCreate, HealthReportDisplay, DashboardData,
    UserCreate, UserLogin, UserResponse, Token
)
from backend.predictor import predictor
from database import get_db, engine
from models import Base, User, HealthReport
from auth import (
    authenticate_user, create_access_token, get_password_hash,
    get_current_user, get_current_commander, get_current_soldier,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Project Sentinel")
templates = Jinja2Templates(directory="templates")

@app.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if user.role == "soldier" and not user.soldier_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Soldier ID is required for soldier role"
        )
    
    if user.soldier_id:
        existing_soldier = db.query(User).filter(User.soldier_id == user.soldier_id).first()
        if existing_soldier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Soldier ID already exists"
            )
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,
        soldier_id=user.soldier_id,
        unit=user.unit
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/login", response_model=Token)
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    authenticated_user = authenticate_user(db, user.email, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user.email}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/report", response_model=HealthReportDisplay)
async def create_report(
    report: HealthReportCreate,
    current_user: User = Depends(get_current_soldier),
    db: Session = Depends(get_db)
):
    if current_user.soldier_id != report.soldier_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only submit reports for your own soldier ID"
        )
    
    prediction_result = predictor.predict_risk(report)
    db_report = HealthReport(
        user_id=current_user.id,
        soldier_id=report.soldier_id,
        sleep_hours=report.sleep_hours,
        hrv_score=report.hrv_score,
        stress_level=report.stress_level,
        activity_calories=report.activity_calories,
        risk_probability=prediction_result["risk_probability"],
        risk_level=prediction_result["risk_level"],
        risk_color=prediction_result["risk_color"]
    )
    
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    return db_report

@app.get("/my-reports", response_model=List[HealthReportDisplay])
async def get_my_reports(
    current_user: User = Depends(get_current_soldier),
    db: Session = Depends(get_db)
):
    reports = db.query(HealthReport).filter(
        HealthReport.user_id == current_user.id
    ).order_by(HealthReport.timestamp.desc()).all()
    
    return reports

@app.get("/unit-reports", response_model=List[HealthReportDisplay])
async def get_unit_reports(
    current_user: User = Depends(get_current_commander),
    db: Session = Depends(get_db)
):
    soldiers_in_unit = db.query(User).filter(
        User.unit == current_user.unit,
        User.role == "soldier"
    ).all()
    
    soldier_ids = [soldier.id for soldier in soldiers_in_unit]
    reports = db.query(HealthReport).filter(
        HealthReport.user_id.in_(soldier_ids)
    ).order_by(HealthReport.timestamp.desc()).all()
    
    return reports

@app.get("/", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/soldier-dashboard", response_class=HTMLResponse)
async def get_soldier_dashboard(
    request: Request,
    current_user: User = Depends(get_current_soldier),
    db: Session = Depends(get_db)
):
    reports = db.query(HealthReport).filter(
        HealthReport.user_id == current_user.id
    ).order_by(HealthReport.timestamp.desc()).limit(10).all()
    
    return templates.TemplateResponse(
        "soldier_dashboard.html", 
        {"request": request, "user": current_user, "reports": reports}
    )

@app.get("/commander-dashboard", response_class=HTMLResponse)
async def get_commander_dashboard(
    request: Request,
    current_user: User = Depends(get_current_commander),
    db: Session = Depends(get_db)
):
    soldiers_in_unit = db.query(User).filter(
        User.unit == current_user.unit,
        User.role == "soldier"
    ).all()
    
    soldier_ids = [soldier.id for soldier in soldiers_in_unit]
    latest_reports = []
    for soldier in soldiers_in_unit:
        latest_report = db.query(HealthReport).filter(
            HealthReport.user_id == soldier.id
        ).order_by(HealthReport.timestamp.desc()).first()
        
        if latest_report:
            latest_reports.append(latest_report)
    
    return templates.TemplateResponse(
        "commander_dashboard.html",
        {"request": request, "user": current_user, "reports": latest_reports, "soldiers": soldiers_in_unit}
    )

@app.post("/register-form")
async def register_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    soldier_id: Optional[str] = Form(None),
    unit: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        user_data = UserCreate(
            email=email,
            password=password,
            role=role,
            soldier_id=soldier_id,
            unit=unit
        )
        await register_user(user_data, db)
        return RedirectResponse(url="/", status_code=303)
    except HTTPException as e:
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": e.detail}
        )

@app.post("/login-form")
async def login_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        user_data = UserLogin(email=email, password=password)
        token = await login_user(user_data, db)
        
        user = db.query(User).filter(User.email == email).first()
        if user.role == "soldier":
            response = RedirectResponse(url="/soldier-dashboard", status_code=303)
        else:
            response = RedirectResponse(url="/commander-dashboard", status_code=303)
        response.set_cookie(key="access_token", value=f"Bearer {token['access_token']}")
        return response
        
    except HTTPException as e:
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": e.detail}
        )

@app.post("/submit-report-form")
async def submit_report_form(
    request: Request,
    sleep_hours: float = Form(...),
    hrv_score: int = Form(...),
    stress_level: int = Form(...),
    activity_calories: int = Form(...),
    current_user: User = Depends(get_current_soldier),
    db: Session = Depends(get_db)
):
    try:
        report_data = HealthReportCreate(
            soldier_id=current_user.soldier_id,
            sleep_hours=sleep_hours,
            hrv_score=hrv_score,
            stress_level=stress_level,
            activity_calories=activity_calories
        )
        
        await create_report(report_data, current_user, db)
        return RedirectResponse(url="/soldier-dashboard", status_code=303)
        
    except HTTPException as e:
        reports = db.query(HealthReport).filter(
            HealthReport.user_id == current_user.id
        ).order_by(HealthReport.timestamp.desc()).limit(10).all()
        
        return templates.TemplateResponse(
            "soldier_dashboard.html",
            {"request": request, "user": current_user, "reports": reports, "error": e.detail}
        )