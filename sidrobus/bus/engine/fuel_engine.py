"""Fuel engine module for a bus."""

from sidrobus.bus.engine.abstract_engine import AbstractEngine


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
