"""Module containing the abstract class for engines."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from sidrobus.route import Route


class AbstractEngine(ABC):
    """Abstract class representing an engine.

    This class serves as a base for different types of engines, such as fuel engines and
    electric engines. It defines the basic properties and methods that any engine should
    implement.
    """

    _efficiency: float
    _mass: float
    _energy: float
    _capacity: float

    def __init__(
        self,
        efficiency: float,
        capacity: float,
        mass: float,
        energy: float | None = None,
    ) -> None:
        """Intializes the engine.

        Args:
            efficiency (float): Efficiency of the engine.
            capacity (float): Maximum energy capacity of the engine in Joules.
            mass (float): Mass of the engine.
            energy (float, optional): Current energy level of the engine in Joules.

        Returns:
            None
        """
        self._efficiency = efficiency
        self._capacity = capacity
        self._mass = mass
        self._energy = energy if energy is not None else capacity

    @property
    def capacity(self) -> float:
        """Returns the maximum energy capacity of the engine."""
        return self._capacity

    @property
    def energy(self) -> float:
        """Returns the current energy level of the engine."""
        return self._energy

    @property
    def energy_ratio(self) -> float:
        """Returns the ratio of current energy to maximum capacity."""
        return self._energy / self._capacity

    @property
    @abstractmethod
    def mass(self) -> float:
        """Returns the mass of the engine."""
        return self._mass

    @abstractmethod
    def calculate_route_consumptions(
        self, tractive_efforts: NDArray[np.float64], route: Route, bus_mass: float
    ) -> NDArray[np.float64]:
        """Calculate the energy consumption for a given route based on tractive efforts.

        Args:
            tractive_efforts (NDArray[np.float64]): An array of tractive effort values
                (force applied by the vehicle) at different points along the route.
            route (Route): An object representing the route, including details such as
                distance, elevation profile, and other relevant parameters.
            bus_mass (float): The mass of the bus, which may affect the energy
                consumption.

        Returns:
            None: This method is intended to be implemented by subclasses and does not
                return a value in its abstract form.
        """
        pass
