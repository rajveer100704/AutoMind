# api/agents/advanced_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer


class AdvancedFeatureEngineeringAgent:

    def _target_encode(self, df, target):
        """
        SAFE target encoding — NOT 10k+ features.
        """
        df = df.copy()
        y = df[target]

        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            if col == target:
                continue

            enc = pd.Series(index=df.index, dtype=float)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train, val in kf.split(df):
                enc.iloc[val] = y.iloc[train].mean()

            df[f"{col}_target_enc"] = enc

        return df

    def _text_features(self, df):
        df = df.copy()
        text_cols = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 12]

        for col in text_cols:
            try:
                tfidf = TfidfVectorizer(max_features=40)
                mat = tfidf.fit_transform(df[col].fillna(""))
                tfidf_df = pd.DataFrame(mat.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(mat.shape[1])])
                df = pd.concat([df, tfidf_df], axis=1)
            except:
                pass

        return df

    def enhance(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.copy()

        df = self._target_encode(df, target)
        df = self._text_features(df)

        # SAFE interactions: only between top 10 numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns[:10]

        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                df[f"int_{c1}_{c2}"] = df[c1] * df[c2]

        return df.loc[:, ~df.columns.duplicated()]
