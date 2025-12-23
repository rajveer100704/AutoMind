# api/agents/timeseries_agent.py

import pandas as pd
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from autots import AutoTS


class TimeSeriesAgent:
    """
    Time series forecasting using:
    1) Prophet
    2) ETS
    3) AutoTS fallback
    """

    def _ensure_datetime(self, df: pd.DataFrame):
        df = df.copy()

        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df = df.sort_values(dt_cols[0]).reset_index(drop=True)
            df = df.set_index(dt_cols[0])
            return df

        # No datetime column → use index
        df.index = pd.to_datetime(df.index)
        return df

    def fit(self, df: pd.DataFrame, target: str):

        df = self._ensure_datetime(df)
        data = df[[target]].reset_index()
        data.columns = ["ds", "y"]

        # ---------------- Prophet ----------------
        try:
            m = Prophet()
            m.fit(data)
            return m, [{"model": "prophet", "status": "success"}]
        except:
            pass

        # ---------------- ETS ----------------
        try:
            ets = ExponentialSmoothing(df[target], trend='add').fit()
            return ets, [{"model": "ets", "status": "success"}]
        except:
            pass

        # ---------------- AutoTS ----------------
        try:
            model = AutoTS(
                forecast_length=12,
                frequency='infer',
                ensemble='simple'
            )
            model = model.fit(df[target])
            return model, [{"model": "autots", "status": "success"}]
        except:
            return None, [{"model": "none", "status": "failed"}]
