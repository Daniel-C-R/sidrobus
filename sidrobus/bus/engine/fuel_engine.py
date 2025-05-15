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

    def __init__(
        self,
        efficiency: float,
        capacity: float,
        mass: float,
        energy: float | None = None,
    ) -> None:
        """Initializes a FuelEngine object with efficiency, mass, energy, and capacity.

        Args:
            efficiency (float): Efficiency of the engine.
            capacity (float): Maximum energy capacity of the engine in Joules.
            mass (float): Mass of the engine.
            energy (float, optional): Current energy level of the engine in Joules.
                Defaults to None, which means the engine is fully charged.

        Returns:
            None
        """
        super().__init__(efficiency, capacity, mass, energy)

    @property
    def mass(self) -> float:
        """Returns the mass of the engine."""
        return self._mass

    def calculate_route_consumptions(
        self,
        tractive_efforts: NDArray[np.float64],
        hill_climb_resistances: NDArray[np.float64],  # noqa: ARG002
        linear_acceleration_forces: NDArray[np.float64],  # noqa: ARG002
        route: Route,
    ) -> NDArray[np.float64]:
        """Calculate the fuel consumption for a given route based on tractive efforts.

        Args:
            tractive_efforts (NDArray[np.float64]): An array of tractive efforts (force)
                applied at different points along the route.
            route (Route): The route object containing information such as distances
                for each segment of the route.
            bus_mass (float): The mass of the bus. For fuel engines, this is not used
                in the calculation.

        Returns:
            NDArray[np.float64]: An array of fuel consumption values corresponding
                to each segment of the route.
        """
        consumptions = (tractive_efforts * route.distances / self._efficiency).astype(
            np.float64
        )

        self._energy -= consumptions.sum()

        return consumptions
