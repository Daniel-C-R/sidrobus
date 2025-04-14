"""Module for Bus class."""

from sidrobus.bus.engine.abstract_engine import AbstractEngine


class Bus:
    """Class representing a bus."""

    _engine: AbstractEngine
    _mass: float
    _frontal_area: float
    _aerodynamic_drag_coef: float
    _rolling_resistance_coef: float

    def __init__(
        self,
        engine: AbstractEngine,
        frontal_area: float,
        mass: float,
        aerodynamic_drag_coef: float,
        rolling_resistance_coef: float,
    ) -> None:
        """Initializes a Bus object with an engine, frontal area, and mass.

        Args:
            engine (AbstractEngine): The engine of the bus.
            frontal_area (float): The frontal area of the bus.
            mass (float): The mass of the bus.
            aerodynamic_drag_coef (float): The aerodynamic drag coefficient of the bus.
            rolling_resistance_coef (float): The rolling resistance coefficient of the
                bus.

        Returns:
            None
        """
        self._engine = engine
        self._frontal_area = frontal_area
        self._mass = mass
        self._aerodynamic_drag_coef = aerodynamic_drag_coef
        self._rolling_resistance_coef = rolling_resistance_coef
