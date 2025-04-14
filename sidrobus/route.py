"""Module for representing a bus route as a polyline in 3D space.

The Route class stores geographical coordinates (longitude, latitude) and heights above
sea level for each point along the route.
"""

import numpy as np
from numpy import typing as npt


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
