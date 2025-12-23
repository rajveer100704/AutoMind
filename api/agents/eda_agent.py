# api/agents/eda_agent.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")    # PREVENTS Tkinter errors
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class EDAAgent:
    def _fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        fig.clear()
        return base64.b64encode(buf.read()).decode()

    def analyze(self, df: pd.DataFrame, target: str):
        summary, images = {}, []

        # Summary stats
        try:
            summary["describe"] = df.describe(include="all").to_dict()
        except:
            summary["describe"] = {}

        # Missing % summary
        miss = df.isnull().sum()
        summary["missing_values"] = miss[miss > 0].to_dict()

        # Correlation heatmap
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax)
            images.append(self._fig_to_base64(fig))
            plt.close(fig)

        # Distribution of target
        if target in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                sns.histplot(df[target], kde=True, ax=ax)
                images.append(self._fig_to_base64(fig))
            except:
                pass
            plt.close(fig)

        return summary, images
