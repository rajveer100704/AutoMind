# api/agents/evaluation_agent.py

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


class EvaluationAgent:

    def evaluate(self, model, df, target, task):
        try:
            X = df.drop(columns=[target])
            y = df[target]
        except:
            return {"error": "Target column missing"}

        try:
            pred = model.predict(X)
        except:
            return {"error": "Model failed to predict"}

        if task == "classification":
            return {
                "accuracy": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, average="weighted")),
                "precision": float(precision_score(y, pred, average="weighted")),
                "recall": float(recall_score(y, pred, average="weighted")),
            }

        if task == "regression":
            return {
                "r2": float(r2_score(y, pred)),
                "rmse": float(mean_squared_error(y, pred, squared=False)),
                "mae": float(mean_absolute_error(y, pred)),
            }

        return {"status": "no metrics for time series"}
