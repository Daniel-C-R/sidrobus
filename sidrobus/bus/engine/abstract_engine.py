"""Module containing the abstract class for engines."""

from abc import ABC, abstractmethod


class AbstractEngine(ABC):
    """Abstract class representing an engine.

    This class serves as a base for different types of engines, such as fuel engines and
    electric engines. It defines the basic properties and methods that any engine should
    implement.
    """

    _efficiency: float
    _mass: float

    def __init__(self, efficiency: float, mass: float) -> None:
        """Initializes an engine with efficiency and mass.

        Args:
            efficiency (float): Efficiency of the engine.
            mass (float): Mass of the engine.

        Returns:
            None
        """
        self._efficiency = efficiency
        self._mass = mass

    @property
    @abstractmethod
    def mass(self) -> float:
        """Returns the mass of the engine."""
        return self._mass
