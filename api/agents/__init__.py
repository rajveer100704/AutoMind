# api/agents/__init__.py

"""
AutoMind DS-Agent — Modular Agent Package

This package contains all autonomous agents used in the AutoMind Master Pipeline:

- Data Loading
- Target Detection
- Problem Type Classification
- EDA Generation
- Preprocessing & Cleaning
- Feature Engineering (Basic + Advanced)
- Sampling for Large Datasets
- Model Selection & Training
- Time Series Handling
- Evaluation
- Explainability
- Narrative Generation
- Notebook Generation
- Reporting Engine

All agents are orchestrated by AutoMindMasterAgent.
"""

from .master_agent import AutoMindMasterAgent
