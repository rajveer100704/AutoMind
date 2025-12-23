# api/agents/sampling_policy.py

import pandas as pd
from sklearn.model_selection import train_test_split


class SamplingPolicyAgent:

    def sample(self, df: pd.DataFrame, target: str, frac=1.0) -> pd.DataFrame:
        n = len(df)

        if frac < 1.0:
            return df.sample(frac=frac, random_state=42).reset_index(drop=True)

        if isinstance(df.index, pd.DatetimeIndex):
            return df

        if n <= 200_000:
            return df

        # classification
        if df[target].nunique() <= 20:
            frac = 0.2 if n <= 500_000 else 0.1
            df_small, _ = train_test_split(df, stratify=df[target], train_size=frac, random_state=42)
            return df_small.reset_index(drop=True)

        # regression
        frac = 0.2 if n <= 500_000 else 0.1
        df_small = df.sample(frac=frac, random_state=42)
        return df_small.reset_index(drop=True)
