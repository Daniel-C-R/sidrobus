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
        engine: AbstractEngine,
        frontal_area: float,
        mass: float,
        aerodynamic_drag_coef: float,
        rolling_resistance_coef: float,
        model_name: str = "Unknown",
        model_manufacturer: str = "Unknown",
    ) -> None:
        """Initializes a Bus object with an engine, frontal area, and mass.

        Args:
            engine (AbstractEngine): The engine of the bus.
            frontal_area (float): The frontal area of the bus.
            mass (float): The mass of the bus.
            aerodynamic_drag_coef (float): The aerodynamic drag coefficient of the bus.
            rolling_resistance_coef (float): The rolling resistance coefficient of the
                bus.
            model_name (str, optional): The name of the bus model. Defaults to
                "Unknown".
            model_manufacturer (str, optional): The manufacturer of the bus model.
                Defaults to "Unknown".

        Returns:
            None
        """
        self._engine = engine
        self._frontal_area = frontal_area
        self._mass = mass
        self._aerodynamic_drag_coef = aerodynamic_drag_coef
        self._rolling_resistance_coef = rolling_resistance_coef
        self._model_name = model_name
        self._model_manufacturer = model_manufacturer

    @property
    def mass(self) -> float:
        """Returns the mass of the bus."""
        return self._mass

    @property
    def energy_capacity(self) -> float:
        """Returns the energy capacity of the bus's engine."""
        return self._engine.capacity

    @property
    def actual_energy(self) -> float:
        """Returns the current energy level of the bus's engine."""
        return self._engine.energy

    @property
    def energy_ratio(self) -> float:
        """Returns the ratio of current energy to maximum capacity."""
        return self._engine.energy_ratio

    def compute_route_rolling_resistance_forces(self) -> npt.NDArray[np.float64]:
        """Compute the rolling resistance force of the bus.

        Returns:
            float: Rolling resistance force in Newtons.
        """
        value = self._rolling_resistance_coef * self._mass * EARTH_GRAVITY
        return np.array([value], dtype=np.float64)

    def compute_route_aerodynamic_drag_forces(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the aerodynamic drag forces for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Aerodynamic drag forces for each segment.
        """
        return (
            0.5
            * AIR_DENSITY
            * self._aerodynamic_drag_coef
            * self._frontal_area
            * route.avg_speeds**2
        )

    def compute_route_hill_climb_resistance_forces(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the hill climb resistance forces for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Hill climb resistance forces for each segment.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            sines = np.divide(
                np.diff(route.altitudes),
                route.distances,
                out=np.zeros_like(route.distances, dtype=float),
                where=route.distances != 0,
            )
        sines = np.clip(sines, np.sin(np.deg2rad(-10.2)), np.sin(np.deg2rad(10.2)))
        return self._mass * EARTH_GRAVITY * sines

    def compute_linear_acceleration_forces(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the linear acceleration forces for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Linear acceleration forces for each segment.
        """
        return 1.05 * self._mass * route.accelerations  # 5% for rotational acceleration

    def compute_route_tractive_forces(self, route: Route) -> npt.NDArray[np.float64]:
        """Calculate the total tractive forces acting on the bus during a route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Total tractive forces for each segment.
        """
        rolling_resistance = self.compute_route_rolling_resistance_forces()
        aerodynamic_drag = self.compute_route_aerodynamic_drag_forces(route)
        hill_climb_resistance = self.compute_route_hill_climb_resistance_forces(route)
        linear_acceleration_force = self.compute_linear_acceleration_forces(route)

        mask_hill_climb = hill_climb_resistance > 0
        mask_linear_acceleration = linear_acceleration_force > 0

        return (
            rolling_resistance
            + aerodynamic_drag
            + hill_climb_resistance * mask_hill_climb
            + linear_acceleration_force * mask_linear_acceleration
        )

    def compute_route_rolling_resistance_consumption(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the energy consumption due to rolling resistance along the route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Energy consumption for each segment.
        """
        return self._engine.compute_route_rolling_resistance_consuption(
            route, self.compute_route_rolling_resistance_forces()
        )

    def compute_route_aerodynamic_drag_consumption(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the energy consumption due to aerodynamic drag along the route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Energy consumption for each segment.
        """
        return self._engine.compute_route_aerodynamic_drag_consumption(
            route, self.compute_route_aerodynamic_drag_forces(route)
        )

    def compute_route_hill_climb_consumption(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the energy consumption due to hill climb resistance along the route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Energy consumption for each segment.
        """
        return self._engine.compute_route_hill_climb_consumption(
            route, self.compute_route_hill_climb_resistance_forces(route)
        )

    def compute_linear_acceleration_consumption(
        self, route: Route
    ) -> npt.NDArray[np.float64]:
        """Compute the energy consumption due to linear acceleration along the route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Energy consumption for each segment.
        """
        return self._engine.compute_route_linear_acceleration_consumption(
            route, self.compute_linear_acceleration_forces(route)
        )

    def compute_route_consumption(self, route: Route) -> npt.NDArray[np.float64]:
        """Compute the total energy consumption for a given route.

        Args:
            route (Route): The route of the bus.

        Returns:
            npt.NDArray[np.float64]: Total energy consumption for each segment.
        """
        tractive_forces = self.compute_route_tractive_forces(route)
        return self._engine.compute_route_consumption(tractive_forces, route)

    def compute_route_regeneration(self, route: Route) -> npt.NDArray[np.float64]:
        """Calculate the energy regeneration for a given route.

        This method computes the forces acting on the bus during the route and uses
        the engine model to calculate the total energy that can be regenerated.

        Args:
            route (Route): The route for which the regeneration is to be calculated.

        Returns:
            npt.NDArray[np.float64]: The total energy regeneration for the route.
        """
        hill_climb_resistances = self.compute_route_hill_climb_resistance_forces(route)
        linear_acceleration_forces = self.compute_linear_acceleration_forces(route)

        return self._engine.compute_route_regeneration(
            route, hill_climb_resistances, linear_acceleration_forces
        )

    def compute_route_final_consumption(
        self, route: Route, modify_bus: bool = False
    ) -> npt.NDArray[np.float64]:
        """Calculate the net energy consumption for a given route.

        This method computes the forces acting on the bus throughout the route
        and uses the engine model to calculate the total energy consumption,
        accounting for regeneration if available.

        Args:
            route (Route): The route for which the energy consumption is to be
                calculated.
            modify_bus (bool, optional): If True, modifies the bus's energy state
                after the calculation. Defaults to False.

        Returns:
            npt.NDArray[np.float64]: Net energy consumption for each segment.
        """
        return self._engine.compute_route_final_consumption(
            route,
            self.compute_route_tractive_forces(route),
            self.compute_route_hill_climb_resistance_forces(route),
            self.compute_linear_acceleration_forces(route),
            modify_engine=modify_bus,
        )
