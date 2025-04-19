"""Module for Bus class."""

import numpy as np
from numpy import typing as npt

from sidrobus.bus.engine import AbstractEngine
from sidrobus.constants import EARTH_GRAVITY
from sidrobus.route import Route


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

    def _compute_route_rolling_resistance(self) -> float:
        """Computes the rolling resistance of the bus.

        Returns:
            float: The rolling resistance of the bus.
        """
        return self._rolling_resistance_coef * self._mass * EARTH_GRAVITY

    def _compute_route_aerodynamic_drag(self, route: Route) -> npt.NDArray[np.float64]:
        """Computes the aerodynamic drag for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            float: The aerodynamic drag of the bus.
        """
        return (
            0.5
            * self._aerodynamic_drag_coef
            * self._frontal_area
            * route.avg_velocities**2
        )

    def _compute_route_hill_climb_resistance(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Computes the hill climb resistance for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: The hill climb resistance of the bus.
        """
        return self._mass * EARTH_GRAVITY * np.diff(route.heights) / route.distances

    def _compute_linear_acceleration_force(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Computes the linear acceleration force for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: The linear acceleration force of the bus.
        """
        return self._mass * route.accelerations

    def _calculate_route_forces(self, route: Route) -> npt.NDArray[np.float64]:
        """Calculates the forces acting on the bus during a route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: The forces acting on the bus.
        """
        rolling_resistance = self._compute_route_rolling_resistance()
        aerodynamic_drag = self._compute_route_aerodynamic_drag(route)
        hill_climb_resistance = self._compute_route_hill_climb_resistance(route)
        linear_acceleration_force = self._compute_linear_acceleration_force(route)

        return (
            rolling_resistance
            + aerodynamic_drag
            + hill_climb_resistance
            + 1.05 * linear_acceleration_force  # 5% for rotational acceleration
        )

    def _calculate_route_energy_consumption(
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
        forces = self._calculate_route_forces(route)
        return self._engine.calculate_route_consumptions(forces, route)
