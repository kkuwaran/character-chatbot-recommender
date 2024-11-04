from typing import List
import pandas as pd



COLUMN_NAMES = ["Name", "Medium", "Setting", "Description"]


def combine_columns(row: pd.Series, column_names: List[str] = COLUMN_NAMES) -> str:
    """Combine the columns into a single text string"""

    text = ""
    for column in column_names:
        assert column in row, f"Column '{column}' not found in the DataFrame."
        text += f"{column}: {row[column]}; "
    return text.strip()


def preprocess_data(df: pd.DataFrame, column_names: List[str] = COLUMN_NAMES) -> pd.DataFrame:
    """Preprocess the input DataFrame by combining columns and dropping unnecessary columns."""

    # Combine columns into a single text column
    df["text"] = df.apply(combine_columns, axis=1)
    df = df.drop(columns=column_names)
    return df