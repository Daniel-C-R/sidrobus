"""Fuel engine module for a bus."""

from sidrobus.bus.engine.abstract_engine import AbstractEngine


class FuelEngine(AbstractEngine):
    """Class representing a fuel engine for a bus.

    This class inherits from BaseEngine and implements the methods required for a fuel
    engine.
    """

    _engine_type: str = "Fuel"

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
        """Return the mass of the engine.

        Returns:
            float: Mass of the engine in kilograms.
        """
        return self._mass
