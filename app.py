from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from schemas import HealthReportCreate, HealthReportDisplay
from backend.predictor import predictor 

app = FastAPI(title = "Project Sentinel")
templates = Jinja2Templates(directory = "templates")

REPORTS_DB = []

@app.get("/", response_class = HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/report", response_model = HealthReportDisplay)
async def create_report(report: HealthReportCreate):
    prediction_result = predictor.predict_risk(report)
    
    display_data = {
        "soldier_id": report.soldier_id,
        **prediction_result
    }
    REPORTS_DB.append(display_data)
    return display_data

@app.get("/dashboard", response_class = HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "reports": REPORTS_DB})