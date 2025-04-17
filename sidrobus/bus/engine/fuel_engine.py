"""Fuel engine module for a bus."""

import numpy as np
from numpy.typing import NDArray

from sidrobus.bus.engine.abstract_engine import AbstractEngine
from sidrobus.route import Route


class FuelEngine(AbstractEngine):
    """Class representing a fuel engine for a bus.

    This class inherits from BaseEngine and implements the methods required for a fuel
    engine.
    """

    def __init__(self, efficiency: float, mass: float) -> None:
        """Initializes a FuelEngine object with efficiency, mass, energy, and capacity.

        Args:
            efficiency (float): Efficiency of the engine.
            mass (float): Mass of the engine.

        Returns:
            None
        """
        super().__init__(efficiency, mass)

    @property
    def mass(self) -> float:
        """Returns the mass of the engine."""
        return self._mass

    def calculate_route_consumptions(
        self, tractive_efforts: NDArray[np.float64], route: Route
    ) -> NDArray[np.float64]:
        """Calculate the fuel consumption for a given route based on tractive efforts.

        Args:
            tractive_efforts (NDArray[np.float64]): An array of tractive efforts (force)
                applied at different points along the route.
            route (Route): The route object containing information such as distances
                for each segment of the route.

        Returns:
            NDArray[np.float64]: An array of fuel consumption values corresponding
                to each segment of the route.
        """
        return (tractive_efforts * route.distances / self._efficiency).astype(
            np.float64
        )
