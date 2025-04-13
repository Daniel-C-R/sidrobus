"""Fuel engine module for a bus."""

from sidrobus.bus.engine.abstract_engine import BaseEngine


class FuelEngine(BaseEngine):
    """Class representing a fuel engine for a bus.

    This class inherits from BaseEngine and implements the methods required for a fuel
    engine.
    """

    def __init__(
        self, efficiency: float, mass: float, energy: float, capacity: float
    ) -> None:
        """Initializes a FuelEngine object with efficiency, mass, energy, and capacity.

        Args:
            efficiency (float): Efficiency of the engine.
            mass (float): Mass of the engine.
            energy (float): Energy of the engine.
            capacity (float): Capacity of the engine.

        Returns:
            None
        """
        super().__init__(efficiency, mass, energy, capacity)
