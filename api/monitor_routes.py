# api/monitor_routes.py
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Generator
from api.monitoring import sse_event_generator, read_latest

router = APIRouter(prefix="/monitor", tags=["monitor"])

@router.get("/logs/stream")
def logs_stream(last_event_id: int = Query(0)):
    gen: Generator[str, None, None] = sse_event_generator(last_event_id)
    return StreamingResponse(gen, media_type="text/event-stream")

@router.get("/logs/latest")
def logs_latest(n: int = Query(100)):
    return JSONResponse({"events": read_latest(n)})

@router.get("/health")
def monitor_health():
    return JSONResponse({"status": "ok"})
