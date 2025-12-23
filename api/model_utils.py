# api/model_utils.py
import pandas as pd
from api.agents.master_agent import AutoMindMasterAgent

def train_from_dataframe(df: pd.DataFrame, autodetect_target=True, target_col=None, tune_rounds=10):
    agent = AutoMindMasterAgent(run_id="compat")
    result = agent.run_pipeline(
        df=df,
        autodetect_target=autodetect_target,
        target_override=target_col,
        tune_rounds=tune_rounds,
        advanced_fe=False,
        sample_frac=1.0,
        notebook=False
    )
    return result
