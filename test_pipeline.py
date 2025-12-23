# test_pipeline.py
"""
Tests AutoMind backend end-to-end using local functions,
NOT FastAPI endpoints.
"""

import pandas as pd
from api.agents.master_agent import AutoMindMasterAgent


def test_small_classification():
    print("\n=== TEST: Classification ===")

    df = pd.DataFrame({
        "age": [25, 32, 40, 50, 22],
        "salary": [30, 45, 60, 80, 25],
        "buy": [0, 1, 1, 1, 0]
    })

    agent = AutoMindMasterAgent(run_id="test_clf")
    results = agent.run_pipeline(df)

    print("Metrics:", results["metrics"])
    print("Report:", results["report"])
    print("Artifact:", results["artifact"])
    print("OK\n")


def test_regression():
    print("\n=== TEST: Regression ===")

    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [3, 5, 7, 9, 11]
    })

    agent = AutoMindMasterAgent(run_id="test_reg")
    results = agent.run_pipeline(df)

    print("Metrics:", results["metrics"])
    print("Report:", results["report"])
    print("OK\n")


def test_timeseries():
    print("\n=== TEST: Time Series ===")

    df = pd.DataFrame({
        "ds": pd.date_range(start="2023-01-01", periods=20, freq="D"),
        "y": [i + (i % 3) for i in range(20)]
    })

    df = df.rename(columns={"ds": "timestamp"})

    agent = AutoMindMasterAgent(run_id="test_ts")
    results = agent.run_pipeline(df)

    print("Report:", results["report"])
    print("OK\n")


if __name__ == "__main__":
    test_small_classification()
    test_regression()
    test_timeseries()

    print("\n✔ All tests completed.\n")
