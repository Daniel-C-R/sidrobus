"""Module containing the abstract class for engines."""

from abc import ABC

import numpy as np
from numpy.typing import NDArray

from sidrobus.bus.emissions_standard import NULL_EMISSIONS_STANDARD, EmissionsStandard
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
    _emissions_standard: EmissionsStandard

    def __init__(
        self,
        efficiency: float,
        capacity: float,
        mass: float,
        energy: float | None = None,
        emissions_standard: EmissionsStandard = NULL_EMISSIONS_STANDARD,
    ) -> None:
        """Intializes the engine.

        Args:
            efficiency (float): Efficiency of the engine.
            capacity (float): Maximum energy capacity of the engine in Joules.
            mass (float): Mass of the engine.
            energy (float, optional): Current energy level of the engine in Joules.
            emissions_standard (EmissionsStandard, optional): Emissions standard for the
                engine.

        Returns:
            None
        """
        self._efficiency = efficiency
        self._capacity = capacity
        self._mass = mass
        self._energy = energy if energy is not None else capacity
        self._emissions_standard = emissions_standard

    @property
    def emissions_standard_name(self) -> str:
        """Return the name of the emissions standard.

        Returns:
            str: Name of the emissions standard.
        """
        return self._emissions_standard.name

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
        mask = tractive_efforts > 0
        return (tractive_efforts * mask * route.distances / self._efficiency).astype(
            np.float64
        )

    def compute_route_regeneration(
        self,
        route: Route,  # noqa: ARG002
        tractive_forces: NDArray[np.float64],  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Calculate the energy regeneration for a given route.

        Calculate the energy that can be regenerated during a route based on
        hill climbing and linear acceleration forces, in case the vehicle type supports
        regenerative braking. If the vehicle does not support regenerative braking,
        this method should return zero.

        Args:
            route (Route): The route for which the regeneration is to be calculated.
            tractive_forces (NDArray[np.float64]): An array of tractive forces for each

        Returns:
            NDArray[np.float64]: An array of energy regeneration values for each segment
                of the route.
        """
        return np.array([0], dtype=np.float64)

    def compute_route_net_consumption(
        self,
        route: Route,
        tractive_forces: NDArray[np.float64],
        modify_engine: bool = False,
    ) -> NDArray[np.float64]:
        """Calculate the net energy consumption for a given route.

        This method computes the total energy consumption minus regeneration for a
        route.  Optionally, it can update the engine's energy state.

        Args:
            route (Route): The route for which the energy consumption is to be
                calculated.
            tractive_forces (NDArray[np.float64]): Tractive forces for each segment.
            modify_engine (bool, optional): If True, modifies the engine's energy state.

        Returns:
            NDArray[np.float64]: Net energy consumption for each segment.
        """
        consumption = self.compute_route_consumption(tractive_forces, route)

        regeneration = self.compute_route_regeneration(route, tractive_forces)

        if modify_engine:
            self._energy -= np.sum(consumption - regeneration)

        return consumption - regeneration

    def compute_route_co_emissions(
        self,
        route: Route,
        tractive_forces: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the CO emissions for a given route.

        This method computes the CO emissions based on the tractive forces and the
        emissions standard of the engine.

        Args:
            route (Route): The route for which the CO emissions are to be calculated.
            tractive_forces (NDArray[np.float64]): The tractive forces acting on the bus
                during the route.

        Returns:
            NDArray[np.float64]: Array of CO emissions values for each segment.
        """
        consumption_joules = self.compute_route_net_consumption(route, tractive_forces)
        # Convert from Joules to kWh: 1 kWh = 3.6e6 J
        consumption_kwh = consumption_joules / 3.6e6
        return self._emissions_standard.co_per_kwh * consumption_kwh

    def compute_route_nox_emissions(
        self,
        route: Route,
        tractive_forces: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the NOx emissions for a given route.

        This method computes the NOx emissions based on the tractive forces and the
        emissions standard of the engine.

        Args:
            route (Route): The route for which the NOx emissions are to be calculated.
            tractive_forces (NDArray[np.float64]): The tractive forces acting on the bus
                during the route.

        Returns:
            NDArray[np.float64]: Array of NOx emissions values for each segment.
        """
        consumption_joules = self.compute_route_net_consumption(route, tractive_forces)
        # Convert from Joules to kWh: 1 kWh = 3.6e6 J
        consumption_kwh = consumption_joules / 3.6e6
        return self._emissions_standard.nox_per_kwh * consumption_kwh

    def compute_route_hc_emissions(
        self,
        route: Route,
        tractive_forces: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the HC emissions for a given route.

        This method computes the HC emissions based on the tractive forces and the
        emissions standard of the engine.

        Args:
            route (Route): The route for which the HC emissions are to be calculated.
            tractive_forces (NDArray[np.float64]): The tractive forces acting on the bus
                during the route.

        Returns:
            NDArray[np.float64]: Array of HC emissions values for each segment.
        """
        consumption_joules = self.compute_route_net_consumption(route, tractive_forces)
        # Convert from Joules to kWh: 1 kWh = 3.6e6 J
        consumption_kwh = consumption_joules / 3.6e6
        return self._emissions_standard.hc_per_kwh * consumption_kwh

    def compute_route_pm_emissions(
        self,
        route: Route,
        tractive_forces: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the PM emissions for a given route.

        This method computes the PM emissions based on the tractive forces and the
        emissions standard of the engine.

        Args:
            route (Route): The route for which the PM emissions are to be calculated.
            tractive_forces (NDArray[np.float64]): The tractive forces acting on the bus
                during the route.

        Returns:
            NDArray[np.float64]: Array of PM emissions values for each segment.
        """
        consumption_joules = self.compute_route_net_consumption(route, tractive_forces)
        # Convert from Joules to kWh: 1 kWh = 3.6e6 J
        consumption_kwh = consumption_joules / 3.6e6
        return self._emissions_standard.pm_per_kwh * consumption_kwh
