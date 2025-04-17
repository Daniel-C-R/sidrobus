"""Module for representing a bus route as a polyline in 3D space.

The Route class stores geographical coordinates (longitude, latitude) and heights above
sea level for each point along the route.
"""

import numpy as np
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
    _heights: npt.NDArray[np.float64]
    _velocities: npt.NDArray[np.float64]

    def __init__(
        self,
        times: npt.NDArray[np.float64],
        longitudes: npt.NDArray[np.float64],
        latitudes: npt.NDArray[np.float64],
        heights: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
    ) -> None:
        """Initializes a Route object with geographical coordinates and heights.

        Args:
            times (npt.NDArray[np.float64]): Array of time values.
            longitudes (npt.NDArray[np.float64]): Array of longitude values.
            latitudes (npt.NDArray[np.float64]): Array of latitude values.
            heights (npt.NDArray[np.float64]): Array of height values.
            velocities (npt.NDArray[np.float64]): Array of velocity values.

        Returns:
            None

        """
        self._times = times
        self._longitudes = longitudes
        self._latitudes = latitudes
        self._heights = heights
        self._velocities = velocities

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
    def heights(self) -> npt.NDArray[np.float64]:
        """Returns the heights of the route.

        Returns:
            npt.NDArray[np.float64]: Array of height values.
        """
        return self._heights

    @property
    def velocities(self) -> npt.NDArray[np.float64]:
        """Returns the velocities of the route.

        Returns:
            npt.NDArray[np.float64]: Array of velocity values.
        """
        return self._velocities

    @property
    def distances(self) -> npt.NDArray[np.float64]:
        """Calculate the distances between consecutive points in a 3D space.

        This method computes the distances between consecutive points defined by
        their latitude, longitude, and height. The calculation takes into account
        the curvature of the Earth and uses the Pythagorean theorem to combine
        horizontal and vertical distances.

        Returns:
            npt.NDArray[np.float64]: A NumPy array containing the distances
                between consecutive points in meters.
        """
        # Convert latitude and longitude differences to meters
        lat_differences = np.radians(np.diff(self._latitudes)) * EARTH_RADIUS
        lon_differences = (
            np.radians(np.diff(self._longitudes))
            * EARTH_RADIUS
            * np.cos(np.radians(self._latitudes[:-1]))
        )

        # Calculate horizontal distances
        horizontal_distances = np.sqrt(lon_differences**2 + lat_differences**2)

        # Calculate differences in heights
        height_differences = np.diff(self._heights)

        # Calculate distances using Pythagorean theorem
        return np.sqrt(height_differences**2 + horizontal_distances**2)

    @property
    def avg_velocities(self) -> npt.NDArray[np.float64]:
        """Calculates the average velocities between consecutive points along the route.

        Returns:
            npt.NDArray[np.float64]: Array of average velocities between consecutive
                points.
        """
        return (self._velocities[:-1] + self._velocities[1:]) / 2

    @property
    def accelerations(self) -> npt.NDArray[np.float64]:
        """Calculates the accelerations between consecutive points along the route.

        Returns:
            npt.NDArray[np.float64]: Array of accelerations between consecutive points.
        """
        return np.diff(self._velocities) / np.diff(self._times)
