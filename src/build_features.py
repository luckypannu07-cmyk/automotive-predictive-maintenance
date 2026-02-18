import pandas as pd

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df = df.merge(max_cycle.rename("max_cycle"), on="engine_id")

    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    return df
