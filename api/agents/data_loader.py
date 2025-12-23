# api/agents/data_loader.py

import pandas as pd


class DataLoaderAgent:
    """
    Loads raw dataframe, fixes types, removes empty columns.
    """

    def load(self, df: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df = df.reset_index(drop=True)

        # Drop columns entirely empty
        df = df.dropna(axis=1, how='all')

        # Best-effort numeric conversion
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        return df
