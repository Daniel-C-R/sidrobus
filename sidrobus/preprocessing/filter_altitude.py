"""Functions to filter altitude data in route dataframes."""

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def gaussian_filter(route_data: pd.DataFrame, sigma: float = 1.0) -> pd.DataFrame:
    """Applies a 1D Gaussian filter to the 'altitude' column of a DataFrame.

    Parameters:
        route_data (pd.DataFrame): Input DataFrame containing at least an 'altitude'
            column.
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 1.0.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the 'altitude' column smoothed
            by the Gaussian filter.
    """
    filtered_data = route_data.copy()
    if "altitude" in filtered_data.columns:
        filtered_data["altitude"] = gaussian_filter1d(
            filtered_data["altitude"].values, sigma=sigma, mode="nearest"
        )
    return filtered_data


def savgol_filter_route(
    route_data: pd.DataFrame, window_length: int | None = None, polyorder: int = 2
) -> pd.DataFrame:
    """Apply Savitzky-Golay filter to the altitude data in the route DataFrame.

    Args:
        route_data (pd.DataFrame): DataFrame containing route data with an 'altitude'
            column.
        window_length (int | None): The length of the filter window. If None, it
            defaults to one-tenth of the number of samples in the DataFrame.
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        pd.DataFrame: A new DataFrame with the filtered altitude data.
    """
    if window_length is None:
        window_length = route_data.shape[0] // 10

    filtered_data = route_data.copy()
    if "altitude" in filtered_data.columns:
        filtered_data["altitude"] = savgol_filter(
            filtered_data["altitude"].values, window_length, polyorder
        )

    return filtered_data
