# test_backend.py
import requests, json

BACKEND = "http://127.0.0.1:8000"

print("Health:", requests.get(f"{BACKEND}/health").json())

csv = b"age,income,target\n25,50000,1\n40,90000,0\n33,62000,1\n"
files = {"file": ("sample.csv", csv, "text/csv")}
data = {"autodetect": "True", "target": "", "tune": "3", "advanced_fe": "False", "sample_frac": "1.0"}

r = requests.post(f"{BACKEND}/run_agent", files=files, data=data, timeout=600)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print("Non-JSON response:", r.text)
