# api/agents/model_agent.py
# Updated for PyCaret 3.3.2 (silent=True removed)

import pandas as pd
from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare,
    tune_model as clf_tune,
    finalize_model as clf_finalize,
    pull as clf_pull,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    tune_model as reg_tune,
    finalize_model as reg_finalize,
    pull as reg_pull,
)


class ModelTrainingAgent:
    """
    AutoML Model Training using PyCaret 3.x
    Supports classification & regression.
    """

    def __init__(self, run_id):
        self.run_id = run_id

    def train(self, df: pd.DataFrame, target: str, task: str, tune_rounds: int = 10):

        if task == "classification":
            return self._classification(df, target, tune_rounds)
        else:
            return self._regression(df, target, tune_rounds)

    # ---------------- Classification ---------------- #
    def _classification(self, df, target, tune_rounds):

        clf_setup(
            data=df,
            target=target,
            session_id=42,
            preprocess=True,
            verbose=False
        )

        best_model = clf_compare()
        leaderboard = clf_pull().to_dict(orient="records")

        try:
            tuned = clf_tune(best_model, n_iter=tune_rounds)
            final_model = clf_finalize(tuned)
        except Exception:
            final_model = clf_finalize(best_model)

        return final_model, leaderboard

    # ---------------- Regression ---------------- #
    def _regression(self, df, target, tune_rounds):

        reg_setup(
            data=df,
            target=target,
            session_id=42,
            preprocess=True,
            verbose=False
        )

        best_model = reg_compare()
        leaderboard = reg_pull().to_dict(orient="records")

        try:
            tuned = reg_tune(best_model, n_iter=tune_rounds)
            final_model = reg_finalize(tuned)
        except Exception:
            final_model = reg_finalize(best_model)

        return final_model, leaderboard
