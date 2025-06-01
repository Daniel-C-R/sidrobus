"""Module for representing a bus route as a polyline in 3D space.

The Route class stores geographical coordinates (longitude, latitude) and altitudes
above sea level for each point along the route.
"""

import numpy as np
import pandas as pd
from numpy import typing as npt

from sidrobus.constants import EARTH_RADIUS


class Route:
    """Represents a bus route as a polyline.

    Reperesnts a bus route as a polyline in 3D space, with each point represented
    longitude, latitude and height above sea level. For each point, the time of arrival
    and the volocity is included.
    """

    _times: npt.NDArray[np.float64]
    _longitudes: npt.NDArray[np.float64]
    _latitudes: npt.NDArray[np.float64]
    _altitudes: npt.NDArray[np.float64]
    _speeds: npt.NDArray[np.float64]

    def __init__(
        self,
        times: npt.NDArray[np.float64],
        longitudes: npt.NDArray[np.float64],
        latitudes: npt.NDArray[np.float64],
        altitudes: npt.NDArray[np.float64],
        speeds: npt.NDArray[np.float64],
    ) -> None:
        """Initializes a Route object with geographical coordinates and altitudes.

        Args:
            times (npt.NDArray[np.float64]): Array of time values.
            longitudes (npt.NDArray[np.float64]): Array of longitude values.
            latitudes (npt.NDArray[np.float64]): Array of latitude values.
            altitudes (npt.NDArray[np.float64]): Array of height values.
            speeds (npt.NDArray[np.float64]): Array of velocity values.

        Returns:
            None

        """
        self._times = times
        self._longitudes = longitudes
        self._latitudes = latitudes
        self._altitudes = altitudes
        self._speeds = speeds

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Route":
        """Create a Route object from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing columns 'time', 'longitude',
                'latitude', 'altitude', and 'speed'.

        Returns:
            Route: A Route object initialized with the data from the DataFrame.
        """
        return cls(
            times=df["time"].to_numpy(dtype=np.float64),
            longitudes=df["longitude"].to_numpy(dtype=np.float64),
            latitudes=df["latitude"].to_numpy(dtype=np.float64),
            altitudes=df["altitude"].to_numpy(dtype=np.float64),
            speeds=df["speed"].to_numpy(dtype=np.float64),
        )

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """Returns the times of the route.

        Returns:
            npt.NDArray[np.float64]: Array of time values.
        """
        return self._times

    @property
    def longitudes(self) -> npt.NDArray[np.float64]:
        """Returns the longitudes of the route.

        Returns:
            npt.NDArray[np.float64]: Array of longitude values.
        """
        return self._longitudes

    @property
    def latitudes(self) -> npt.NDArray[np.float64]:
        """Returns the latitudes of the route.

        Returns:
            npt.NDArray[np.float64]: Array of latitude values.
        """
        return self._latitudes

    @property
    def altitudes(self) -> npt.NDArray[np.float64]:
        """Returns the altitudes of the route.

        Returns:
            npt.NDArray[np.float64]: Array of height values.
        """
        return self._altitudes

    @property
    def speeds(self) -> npt.NDArray[np.float64]:
        """Returns the speeds of the route.

        Returns:
            npt.NDArray[np.float64]: Array of velocity values.
        """
        return self._speeds

    @property
    def distances(self) -> npt.NDArray[np.float64]:
        """Calculates the distances between consecutive points along the route.

        Calculate the distances between consecutive points in a 3D space using the
        haversine formula.

        This method computes the distances between consecutive points defined by
        their latitude, longitude, and height. The calculation uses the haversine
        formula for horizontal distance and the Pythagorean theorem to combine
        horizontal and vertical distances.

        Returns:
            npt.NDArray[np.float64]: A NumPy array containing the distances
                between consecutive points in meters.
        """
        # Convert degrees to radians
        lat_rad = np.radians(self._latitudes)
        lon_rad = np.radians(self._longitudes)

        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)

        lat1 = lat_rad[:-1]
        lat2 = lat_rad[1:]

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        horizontal_distances = EARTH_RADIUS * c

        # Calculate differences in altitudes
        height_differences = np.diff(self._altitudes)

        # Calculate distances using Pythagorean theorem
        return np.sqrt(horizontal_distances**2 + height_differences**2)

    @property
    def avg_speeds(self) -> npt.NDArray[np.float64]:
        """Calculates the average speeds between consecutive points along the route.

        Returns:
            npt.NDArray[np.float64]: Array of average speeds between consecutive
                points.
        """
        return (self._speeds[:-1] + self._speeds[1:]) / 2

    @property
    def accelerations(self) -> npt.NDArray[np.float64]:
        """Calculates the accelerations between consecutive points along the route.

        Returns:
            npt.NDArray[np.float64]: Array of accelerations between consecutive points.
        """
        accelerations = np.diff(self._speeds) / np.diff(self._times)
        return np.clip(accelerations, -2, 2)
