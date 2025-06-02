"""Module containing the abstract class for engines."""

from abc import ABC

import numpy as np
from numpy.typing import NDArray

from sidrobus.route import Route


class AbstractEngine(ABC):
    """Abstract class representing an engine.

    This class serves as a base for different types of engines, such as fuel engines and
    electric engines. It defines the basic properties and methods that any engine should
    implement.
    """

    _engine_type: str
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
    def engine_type(self) -> str:
        """Return the type of the engine.

        Returns:
            str: The type of the engine.
        """
        return self._engine_type

    @property
    def capacity(self) -> float:
        """Return the maximum energy capacity of the engine.

        Returns:
            float: Maximum energy capacity in Joules.
        """
        return self._capacity

    @property
    def energy(self) -> float:
        """Return the current energy level of the engine.

        Returns:
            float: Current energy level in Joules.
        """
        return self._energy

    @energy.setter
    def energy(self, value: float) -> None:
        """Set the energy value for the engine.

        Args:
            value (float): The new energy value to assign.
        """
        self._energy = value

    @property
    def energy_ratio(self) -> float:
        """Return the ratio of current energy to maximum capacity.

        Returns:
            float: Ratio of current energy to capacity.
        """
        return self._energy / self._capacity

    @property
    def mass(self) -> float:
        """Return the mass of the engine.

        Returns:
            float: Mass of the engine in kilograms.
        """
        return self._mass

    def compute_route_rolling_resistance_consuption(
        self, route: Route, rolling_resistance_forces: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate the fuel consumption for rolling resistance forces along a route.

        Args:
            route (Route): The route object containing information such as distances
                for each segment of the route.
            rolling_resistance_forces (NDArray[np.float64]): An array of rolling
                resistance forces applied at different points along the route.

        Returns:
            NDArray[np.float64]: An array of fuel consumption values corresponding
                to each segment of the route.
        """
        return (rolling_resistance_forces * route.distances / self._efficiency).astype(
            np.float64
        )

    def compute_route_aerodynamic_drag_consumption(
        self,
        route: Route,
        aerodynamic_drag_forces: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the fuel consumption for aerodynamic drag forces along a route.

        Args:
            route (Route): The route object containing information such as distances
                for each segment of the route.
            aerodynamic_drag_forces (NDArray[np.float64]): An array of aerodynamic
                drag forces applied at different points along the route.

        Returns:
            NDArray[np.float64]: An array of fuel consumption values corresponding
                to each segment of the route.
        """
        return (aerodynamic_drag_forces * route.distances / self._efficiency).astype(
            np.float64
        )

    def compute_route_hill_climb_consumption(
        self,
        route: Route,
        hill_climb_resistances: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the fuel consumption for hill climb resistances along a route.

        Args:
            route (Route): The route object containing information such as distances
                for each segment of the route.
            hill_climb_resistances (NDArray[np.float64]): An array of hill climb
                resistances applied at different points along the route.

        Returns:
            NDArray[np.float64]: An array of fuel consumption values corresponding
                to each segment of the route.
        """
        mask = hill_climb_resistances > 0
        return (
            hill_climb_resistances * mask * route.distances / self._efficiency
        ).astype(np.float64)

    def compute_route_linear_acceleration_consumption(
        self,
        route: Route,
        linear_acceleration_forces: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the fuel consumption for linear acceleration forces along a route.

        Args:
            route (Route): The route object containing information such as distances
                for each segment of the route.
            linear_acceleration_forces (NDArray[np.float64]): An array of linear
                acceleration forces applied at different points along the route.

        Returns:
            NDArray[np.float64]: An array of fuel consumption values corresponding
                to each segment of the route.
        """
        mask = linear_acceleration_forces > 0
        return (
            linear_acceleration_forces * mask * route.distances / self._efficiency
        ).astype(np.float64)

    def compute_route_consumption(
        self,
        tractive_efforts: NDArray[np.float64],
        route: Route,
    ) -> NDArray[np.float64]:
        """Calculate the energy consumption for a given route based on tractive efforts.

        Args:
            tractive_efforts (NDArray[np.float64]): An array of tractive effort values
                (force applied by the vehicle) at different points along the route.
            route (Route): An object representing the route, including details such as
                distance, elevation profile, and other relevant parameters.

        Returns:
            NDArray[np.float64]: An array of energy consumption values corresponding to
                each segment of the route, calculated based on the tractive efforts and
                the engine's efficiency.
        """
        return (tractive_efforts * route.distances / self._efficiency).astype(
            np.float64
        )

    def compute_route_regeneration(
        self,
        route: Route,  # noqa: ARG002
        hill_climb_forces: NDArray[np.float64],  # noqa: ARG002
        linear_acceleration_forces: NDArray[np.float64],  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Calculate the energy regeneration for a given route.

        Calculate the energy that can be regenerated during a route based on
        hill climbing and linear acceleration forces, in case the vehicle type supports
        regenerative braking. If the vehicle does not support regenerative braking,
        this method should return zero.

        Args:
            route (Route): The route for which the regeneration is to be calculated.
            hill_climb_forces (NDArray[np.float64]): Forces due to hill climbing.
            linear_acceleration_forces (NDArray[np.float64]): Forces due to linear
                acceleration.

        Returns:
            NDArray[np.float64]: An array of energy regeneration values for each segment
                of the route.
        """
        return np.array([0], dtype=np.float64)

    def compute_route_final_consumption(
        self,
        route: Route,
        tractive_efforts: NDArray[np.float64],
        hill_climb_resistances: NDArray[np.float64],
        linear_acceleration_forces: NDArray[np.float64],
        modify_engine: bool = False,
    ) -> NDArray[np.float64]:
        """Calculate the net energy consumption for a given route.

        This method computes the total energy consumption minus regeneration for a
        route.  Optionally, it can update the engine's energy state.

        Args:
            route (Route): The route for which the energy consumption is to be
                calculated.
            tractive_efforts (NDArray[np.float64]): Tractive forces for each segment.
            hill_climb_resistances (NDArray[np.float64]): Hill climb resistances for
                each segment.
            linear_acceleration_forces (NDArray[np.float64]): Linear acceleration forces
                for each segment.
            modify_engine (bool, optional): If True, modifies the engine's energy state.

        Returns:
            NDArray[np.float64]: Net energy consumption for each segment.
        """
        consumption = self.compute_route_consumption(tractive_efforts, route)

        regeneration = self.compute_route_regeneration(
            route, hill_climb_resistances, linear_acceleration_forces
        )

        if modify_engine:
            self._energy -= np.sum(consumption - regeneration)

        return consumption - regeneration
