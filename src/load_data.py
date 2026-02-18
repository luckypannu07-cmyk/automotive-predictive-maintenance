import pandas as pd

COLUMN_NAMES = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )
    df.columns = COLUMN_NAMES
    return df
