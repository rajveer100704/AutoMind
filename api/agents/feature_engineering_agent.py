# api/agents/feature_engineering_agent.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineeringAgent:
    def transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.copy()

        # ---- Date part extraction ----
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dow"] = df[col].dt.dayofweek

        # ---- Polynomial features ----
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

        if len(num_cols) > 0:
            try:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_vals = poly.fit_transform(df[num_cols])
                poly_df = pd.DataFrame(poly_vals, columns=poly.get_feature_names_out(num_cols))
                df = pd.concat([df, poly_df], axis=1)
            except:
                pass

        # Remove duplicates
        df = df.loc[:, ~df.columns.duplicated()]

        return df
