# api/agents/explainability_agent.py

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

import shap
import base64
from io import BytesIO
import pandas as pd

class ExplainabilityAgent:
    def _fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def explain(self, model, df: pd.DataFrame, target: str):
        X = df.drop(columns=[target])

        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            fig = shap.plots.bar(shap_values, show=False)
            enc = self._fig_to_base64(fig.figure)
            return [enc]
        except Exception as e:
            print("SHAP failed:", e)
            return []
