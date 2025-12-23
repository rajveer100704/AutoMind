# api/agents/target_detector.py

import pandas as pd


class TargetDetectorAgent:
    """
    Auto-detects target column using heuristics.
    """

    COMMON_NAMES = ["target", "label", "class", "output", "y"]

    def detect(self, df: pd.DataFrame, override=None, autodetect=True):

        # Manual user override
        if override and override in df.columns:
            return override

        if not autodetect:
            raise ValueError("Target must be specified if autodetect=False")

        lower_map = {c.lower(): c for c in df.columns}

        # 1. Look for common names
        for name in self.COMMON_NAMES:
            if name in lower_map:
                return lower_map[name]

        # 2. Column with lowest unique values
        nunique = df.nunique().sort_values()
        return nunique.index[0]
