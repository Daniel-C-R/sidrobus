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

    @property
    @abstractmethod
    def mass(self) -> float:
        """Returns the mass of the engine."""
        return self._mass
