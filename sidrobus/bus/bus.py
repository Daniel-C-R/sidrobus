"""Module for Bus class."""

import numpy as np
from numpy import typing as npt

from sidrobus.bus.engine import AbstractEngine
from sidrobus.constants import AIR_DENSITY, EARTH_GRAVITY
from sidrobus.route import Route


class Bus:
    """Class representing a bus."""

    _model_name: str
    _model_manufacturer: str
    _engine: AbstractEngine
    _mass: float
    _frontal_area: float
    _aerodynamic_drag_coef: float
    _rolling_resistance_coef: float

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        model_manufacturer: str,
        engine: AbstractEngine,
        frontal_area: float,
        mass: float,
        aerodynamic_drag_coef: float,
        rolling_resistance_coef: float,
    ) -> None:
        """Initializes a Bus object with an engine, frontal area, and mass.

        Args:
            model_name (str): The name of the bus model.
            model_manufacturer (str): The manufacturer of the bus model.
            engine (AbstractEngine): The engine of the bus.
            frontal_area (float): The frontal area of the bus.
            mass (float): The mass of the bus.
            aerodynamic_drag_coef (float): The aerodynamic drag coefficient of the bus.
            rolling_resistance_coef (float): The rolling resistance coefficient of the
                bus.

        Returns:
            None
        """
        self._model_name = model_name
        self._model_manufacturer = model_manufacturer
        self._engine = engine
        self._frontal_area = frontal_area
        self._mass = mass
        self._aerodynamic_drag_coef = aerodynamic_drag_coef
        self._rolling_resistance_coef = rolling_resistance_coef

    @property
    def mass(self) -> float:
        """Returns the mass of the bus."""
        return self._mass

    def compute_route_rolling_resistance(self) -> float:
        """Computes the rolling resistance of the bus.

        Returns:
            float: The rolling resistance of the bus.
        """
        return self._rolling_resistance_coef * self._mass * EARTH_GRAVITY

    def compute_route_aerodynamic_drag(self, route: Route) -> npt.NDArray[np.float64]:
        """Computes the aerodynamic drag for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            float: The aerodynamic drag of the bus.
        """
        return (
            0.5
            * AIR_DENSITY
            * self._aerodynamic_drag_coef
            * self._frontal_area
            * route.avg_velocities**2
        )

    def compute_route_hill_climb_resistance(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Computes the hill climb resistance for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: The hill climb resistance of the bus.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            sines = np.divide(
                np.diff(route.heights),
                route.distances,
                out=np.zeros_like(route.distances, dtype=float),
                where=route.distances != 0,
            )
        sines = np.clip(sines, np.sin(np.deg2rad(-10.2)), np.sin(np.deg2rad(10.2)))
        return self._mass * EARTH_GRAVITY * sines

    def compute_linear_acceleration_force(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Computes the linear acceleration force for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: The linear acceleration force of the bus.
        """
        return self._mass * route.accelerations

    def calculate_route_forces(self, route: Route) -> npt.NDArray[np.float64]:
        """Calculates the forces acting on the bus during a route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: The forces acting on the bus.
        """
        rolling_resistance = self.compute_route_rolling_resistance()
        aerodynamic_drag = self.compute_route_aerodynamic_drag(route)
        hill_climb_resistance = self.compute_route_hill_climb_resistance(route)
        linear_acceleration_force = self.compute_linear_acceleration_force(route)

        mask_hill_climb = hill_climb_resistance > 0
        mask_linear_acceleration = linear_acceleration_force > 0

        return (
            rolling_resistance
            + aerodynamic_drag
            + hill_climb_resistance * mask_hill_climb
            # +5% for rotational acceleration
            + 1.05 * linear_acceleration_force * mask_linear_acceleration
        )

    def calculate_route_energy_consumption(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Calculate the energy consumption for a given route.

        This method computes the forces acting on the bus throughout the route
        and uses the engine model to calculate the total energy consumption.

        Args:
            route (Route): The route for which the energy consumption is to be
                calculated.

        Returns:
            float: The total energy consumption for the route.
        """
        forces = self.calculate_route_forces(route)
        hill_climb_resistances = self.compute_route_hill_climb_resistance(route)
        linear_acceleration_forces = self.compute_linear_acceleration_force(route)
        return self._engine.calculate_route_consumptions(
            forces,
            hill_climb_resistances,
            linear_acceleration_forces,
            route,
        )
