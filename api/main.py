# api/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import pandas as pd
import os
import uuid
import json

# Ensure needed directories exist
os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# Monitoring
from api.monitor_routes import router as monitor_router
from api.monitoring import sse_event_generator, read_latest

# Logging
try:
    from logging_config import setup_logging
    logger = setup_logging()
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("automind")

# Master Agent
from api.agents.master_agent import AutoMindMasterAgent

app = FastAPI(
    title="AutoMind DS-Agent",
    version="1.0"
)

# Add monitor endpoints
app.include_router(monitor_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def home():
    return {"status": "AutoMind backend running"}

# Health
@app.get("/health")
def health():
    return {"status": "ok"}

# Main pipeline
@app.post("/run_agent")
async def run_agent(
    file: UploadFile = File(...),
    autodetect: str = Form("True"),
    target: str = Form(""),
    tune: str = Form("10"),
    advanced_fe: str = Form("False"),
    sample_frac: str = Form("1.0")
):

    run_id = str(uuid.uuid4())[:8]
    logger.info(f"[{run_id}] Pipeline started")

    # load CSV into pandas safely (support file-like)
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        logger.error(f"[{run_id}] Failed to read CSV: {e}")
        return JSONResponse({"error": "failed to read uploaded CSV", "detail": str(e)}, status_code=400)

    agent = AutoMindMasterAgent(run_id=run_id)

    try:
        results = agent.run_pipeline(
            df=df,
            autodetect_target=(autodetect.lower() == "true"),
            target_override=target if target.strip() else None,
            tune_rounds=int(tune),
            advanced_fe=(advanced_fe.lower() == "true"),
            sample_frac=float(sample_frac),
            notebook=True
        )
    except Exception as e:
        logger.error(f"[{run_id}] Pipeline failed: {e}")
        return JSONResponse({"error": "Internal Server Error", "detail": str(e)}, status_code=500)

    return JSONResponse(results)

# Download artifact ZIP
@app.get("/artifact/{run_id}")
def download_artifact(run_id: str):
    path = f"artifacts/artifact_{run_id}.zip"
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="application/zip")

# View report
@app.get("/reports/{run_id}")
def view_report(run_id: str):
    path = f"reports/report_{run_id}.html"
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")

# Runs list (UI calls this to populate run history)
@app.get("/runs")
def get_runs():
    hist_path = "run_history.json"
    if not os.path.exists(hist_path):
        return JSONResponse({"runs": []})
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse({"runs": data})
    except Exception:
        return JSONResponse({"runs": []})

# SSE stream (fallback)
@app.get("/logs/stream")
def logs_stream(last_event_id: int = 0):
    return StreamingResponse(sse_event_generator(last_event_id), media_type="text/event-stream")

@app.get("/logs/latest")
def logs_latest(n: int = 100):
    return {"events": read_latest(n)}
