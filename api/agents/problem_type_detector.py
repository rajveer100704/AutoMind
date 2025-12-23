# api/agents/problem_type_detector.py

import pandas as pd


class ProblemTypeDetectorAgent:
    """
    Predict task type:
    - Time series → datetime index or timestamp col
    - Classification → target has <=20 unique values
    - Regression → default
    """

    def detect(self, df: pd.DataFrame, target: str) -> str:

        # Time-series check 1: datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            return "timeseries"

        # Time-series check 2: timestamp column
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return "timeseries"

        # Classification
        if df[target].nunique() <= 20:
            return "classification"

        return "regression"
