# Dockerfile for AutoMind DS-Agent (Python 3.11)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates libxml2-dev libxslt-dev \
    libffi-dev libssl-dev gfortran libblas-dev liblapack-dev libatlas-base-dev libgomp1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
