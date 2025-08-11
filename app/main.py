from fastapi import FastAPI, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List

from . import crud, models, schemas, auth, ml_utils
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="SHMP")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    if user.role != 'commander':
        raise HTTPException(status_code=403, detail="Access denied. Commander role required.")
    soldiers = crud.get_soldiers_with_risk_scores(db, limit=100)
    alerts = crud.get_active_alerts(db, limit=10)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "soldiers": soldiers,
        "alerts": alerts
    })

@app.get("/mobile-form", response_class=HTMLResponse)
async def mobile_form_page(request: Request):
    """Serves the mobile data entry form."""
    return templates.TemplateResponse("mobile_form.html", {"request": request})


@app.post("/api/login", response_model=schemas.Token)
async def login_for_access_token(db: Session = Depends(get_db), form_data: auth.OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username, "role": user.role})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.post("/api/sync")
def sync_data(
    payload: schemas.SyncPayload,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user)
):
    soldiers_to_rescore = set()
    for report_data in payload.mood_reports:
        crud.create_mood_report(db, report_data)
        soldiers_to_rescore.add(report_data.soldier_id)

    for soldier_id in soldiers_to_rescore:
        features = ml_utils.get_features_for_soldier(db, soldier_id)
        if features is None:
            continue

        prediction = ml_utils.predict_risk(features)
        score_data = schemas.RiskScoreCreate(
            soldier_id=soldier_id,
            score=prediction['score'],
            label=prediction['label']
        )
        crud.create_risk_score(db, score_data)
        if prediction['label'] == 'High':
            crud.create_alert(
                db=db,
                soldier_id=soldier_id,
                message=f"High risk detected with score {prediction['score']:.2f}"
            )

    return {"status": "success", "processed_reports": len(payload.mood_reports)}