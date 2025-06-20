"""Electrical engine model for a bus."""

import numpy as np
from numpy import typing as npt

from sidrobus.bus.emissions_standard import NULL_EMISSIONS_STANDARD
from sidrobus.bus.engine import AbstractEngine
from sidrobus.route import Route


class ElectricEngine(AbstractEngine):
    """Abstract class for electric engines.

    This class represents an electric engine model for a bus. It inherits from the
    abstract engine class and implements the methods to calculate energy consumption
    """

    _engine_type: str = "Electric"
    _regenerative_braking_efficiency: float

    def __init__(
        self,
        efficiency: float,
        capacity: float,
        mass: float,
        regenerative_braking_efficiency: float,
        energy: float | None = None,
    ) -> None:
        """Initialize an ElectricEngine instance.

        Args:
            efficiency (float, optional): The efficiency of the engine, represented as a
                value between 0 and 1.
            mass (float, optional): The mass of the engine in kilograms.
            capacity (float, optional): The maximum energy capacity of the engine in
                Joules.
            regenerative_braking_efficiency (float, optional): The efficiency of the
                regenerative braking system, represented as a value between 0 and 1.
            energy (float, optional): The current energy level of the engine in Joules.
                Defaults to None, which means the engine starts with full capacity.
        """
        super().__init__(efficiency, capacity, mass, energy, NULL_EMISSIONS_STANDARD)
        self._regenerative_braking_efficiency = regenerative_braking_efficiency

    @property
    def mass(self) -> float:
        """Returns the mass of the engine.

        TODO: Consider bettery mass in the future
        """
        return self._mass

    def compute_route_regeneration(
        self,
        route: Route,
        tractive_forces: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate the energy regeneration for a given route.

        This method computes the energy that can be regenerated during a route
        based on negative hill climb and linear acceleration forces.

        Args:
            route (Route): The route for which the regeneration is to be calculated.
            tractive_forces (npt.NDArray[np.float64]): The tractive forces acting on the
                bus during the route.

        Returns:
            npt.NDArray[np.float64]: Array of energy regeneration values for each
                segment.
        """
        mask = tractive_forces < 0

        return np.abs(
            tractive_forces
            * mask
            * route.distances
            * self._regenerative_braking_efficiency
        )
