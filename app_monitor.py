# app_monitor.py — Streamlit Monitoring Dashboard for AutoMind
# Run with: streamlit run app_monitor.py

import streamlit as st
import requests
import json
import time
from datetime import datetime

BACKEND_URL = st.secrets.get("backend_url", "http://127.0.0.1:8000")

st.set_page_config(page_title="AutoMind Monitor", layout="wide")
st.title("AutoMind Monitoring Dashboard")

col1, col2 = st.columns([3,1])

with col2:
    st.header("Controls")
    refresh = st.button("Refresh Now")
    live = st.checkbox("Live Stream", value=False)
    limit = st.slider("Events to show", 10, 500, 100)

with col1:
    st.header("Recent Execution Events")
    event_box = st.empty()

def fetch_latest(n=100):
    try:
        r = requests.get(f"{BACKEND_URL}/logs/latest?n={n}")
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
    return []

# Show initial events
events = fetch_latest(limit)

def render_events(events):
    lines = []
    for e in events[::-1]:
        ts = e.get("timestamp")
        step = e.get("step")
        status = e.get("status")
        details = e.get("details")
        lines.append(f"**[{ts}] {step}** — {status} — `{details}`")
    event_box.markdown("\n\n".join(lines))

render_events(events)

# Live stream via SSE client in Python (fallback polling if SSE not available)
if live:
    st.info("Live streaming enabled — this polls the /logs/stream SSE endpoint.")
    try:
        import sseclient
        resp = requests.get(f"{BACKEND_URL}/logs/stream", stream=True)
        client = sseclient.SSEClient(resp)
        for event in client.events():
            try:
                data = json.loads(event.data)
                events.insert(0, data)
                if len(events) > limit:
                    events = events[:limit]
                render_events(events)
            except Exception:
                continue
    except Exception:
        st.warning("sseclient not available or SSE failed. Falling back to polling every 1s.")
        while True:
            events = fetch_latest(limit)
            render_events(events)
            time.sleep(1)

# Manual refresh
if refresh and not live:
    events = fetch_latest(limit)
    render_events(events)

# Simple metrics summary
st.markdown("---")
st.header("Metrics Summary (recent events)")

if events:
    statuses = {}
    for e in events:
        statuses[e.get("status")] = statuses.get(e.get("status"), 0) + 1
    st.write(statuses)
else:
    st.write("No events yet.")
