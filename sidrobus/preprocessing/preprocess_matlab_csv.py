"""Preprocess MATLAB CSV data for Sidrobus."""

import pandas as pd


def preprocess_matlab(matlab_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess MATLAB data.

    Preprocesses a DataFrame containing MATLAB-exported data by converting the
    'Timestamp' column to datetime, computing the elapsed time in seconds from the
    earliest timestamp, and returning the modified DataFrame without the original
    'Timestamp' column.

    Args:
        matlab_data (pd.DataFrame): Input DataFrame with a 'Timestamp' column containing
            date/time strings.

    Returns:
        pd.DataFrame: DataFrame with a new 'time' column representing seconds elapsed
            from the minimum timestamp, and without the original 'Timestamp' column.
    """
    matlab_data["Timestamp"] = pd.to_datetime(matlab_data["Timestamp"])
    matlab_data["time"] = pd.to_datetime(matlab_data["Timestamp"])
    matlab_data["time"] = (
        matlab_data["time"] - matlab_data["time"].min()
    ).dt.total_seconds()
    return matlab_data.drop(columns=["Timestamp"])
