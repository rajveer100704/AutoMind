# api/agents/preprocessing_agent.py
# Safe preprocessing with datetime protection for AutoMind v1.0

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class PreprocessingAgent:
    """
    Safe preprocessing for AutoMind v1.0
    - Converts datetime columns to UNIX integer timestamps (seconds)
    - Imputes numeric and categorical separately
    - One-hot encodes categorical variables (drop_first=True)
    - Scales numeric columns with StandardScaler
    """

    def process(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.copy()

        # ---------------- DATETIME HANDLING ----------------
        datetime_cols = []
        for col in df.columns:
            # Detect datetime dtypes (including pandas nullable datetimelike)
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_datetime64_ns_dtype(df[col]):
                datetime_cols.append(col)
                # convert to integer seconds (safe for sklearn)
                # coerce errors so NaT becomes NaN (and imputer can handle)
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].view("int64") // 10**9  # nan stays as NaN

        # ---------------- Separate numeric and categorical ----------------
        # After datetime conversion, numeric columns include converted datetime ints
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Ensure target not included in features lists
        if target in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target]
        if target in cat_cols:
            cat_cols = [c for c in cat_cols if c != target]

        # ---------------- NUMERIC IMPUTATION ----------------
        if numeric_cols:
            num_imputer = SimpleImputer(strategy="median")
            # fit_transform returns ndarray
            try:
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            except Exception:
                # fallback: impute column-by-column
                for col in numeric_cols:
                    try:
                        imp = SimpleImputer(strategy="median")
                        df[[col]] = imp.fit_transform(df[[col]])
                    except Exception:
                        pass

        # ---------------- CATEGORICAL IMPUTATION ----------------
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            try:
                df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            except Exception:
                # fallback column-by-column
                for col in cat_cols:
                    try:
                        imp = SimpleImputer(strategy="most_frequent")
                        df[[col]] = imp.fit_transform(df[[col]])
                    except Exception:
                        pass

        # ---------------- ONE-HOT ENCODING ----------------
        if cat_cols:
            try:
                df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            except Exception:
                # fallback: simple label conversion of cat columns
                df_encoded = df.copy()
                for col in cat_cols:
                    try:
                        df_encoded[col] = df_encoded[col].astype(str)
                    except Exception:
                        pass
                df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
        else:
            df_encoded = df.copy()

        # ---------------- SCALING NUMERIC ----------------
        final_numeric = [c for c in df_encoded.columns if df_encoded[c].dtype in [np.float64, np.int64]]
        if target in final_numeric:
            final_numeric = [c for c in final_numeric if c != target]

        if final_numeric:
            scaler = StandardScaler()
            try:
                df_encoded[final_numeric] = scaler.fit_transform(df_encoded[final_numeric])
            except Exception:
                # if scaling fails for any reason, leave as-is (avoid crash)
                pass

        # Reset index to ensure downstream consistency
        df_encoded = df_encoded.reset_index(drop=True)

        return df_encoded
