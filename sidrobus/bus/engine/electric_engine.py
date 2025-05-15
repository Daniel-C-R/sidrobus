"""Electrical engine model for a bus."""

import numpy as np
from numpy import typing as npt

from sidrobus.bus.engine import AbstractEngine
from sidrobus.route import Route


class ElectricEngine(AbstractEngine):
    """Abstract class for electric engines.

    This class represents an electric engine model for a bus. It inherits from the
    abstract engine class and implements the methods to calculate energy consumption
    """

    _regenerative_braking_efficiency: float

    def __init__(
        self,
        efficiency: float,
        capacity: float,
        mass: float,
        regenerative_braking_efficiency: float,
        energy: float | None = None,
    ) -> None:
        """Initialize an ElectricEngine instance.

        Args:
            efficiency (float, optional): The efficiency of the engine, represented as a
                value between 0 and 1.
            mass (float, optional): The mass of the engine in kilograms.
            capacity (float, optional): The maximum energy capacity of the engine in
                Joules.
            regenerative_braking_efficiency (float, optional): The efficiency of the
                regenerative braking system, represented as a value between 0 and 1.
            energy (float, optional): The current energy level of the engine in Joules.
                Defaults to None, which means the engine starts with full capacity.
        """
        super().__init__(efficiency, capacity, mass, energy)
        self._regenerative_braking_efficiency = regenerative_braking_efficiency

    @property
    def mass(self) -> float:
        """Returns the mass of the engine.

        TODO: Consider bettery mass in the future
        """
        return self._mass

    def compute_regenerative_braking(
        self,
        hill_clim_resistances: npt.NDArray[np.float64],
        linear_acceleration_forces: npt.NDArray[np.float64],
        route: Route,
    ) -> npt.NDArray[np.float64]:
        """Compute the regenerative braking force for a given route.

        This method calculates the regenerative braking force based on the accelerations
        of the route, the mass of the bus, and the efficiency of the regenerative
        braking system. Regenerative braking is applied only when the bus is
        decelerating (negative accelerations).

        Args:
            route (Route): The route object containing acceleration data.
            bus_mass (float): The mass of the bus in kilograms.

        Returns:
            npt.NDArray[np.float64]: An array representing the regenerative
            braking force applied at each point along the route.
        """
        hill_climb_mask = hill_clim_resistances < 0
        linear_acceleration_mask = linear_acceleration_forces < 0

        return (
            hill_clim_resistances * route.distances * hill_climb_mask
            + 1.05
            * linear_acceleration_forces
            * route.distances
            * linear_acceleration_mask
        ) * self._regenerative_braking_efficiency

    def calculate_route_consumptions(
        self,
        tractive_efforts: npt.NDArray[np.float64],
        hill_climb_resistances: npt.NDArray[np.float64],
        linear_acceleration_forces: npt.NDArray[np.float64],
        route: Route,
    ) -> npt.NDArray[np.float64]:
        """Calculate the energy consumption for a bus along a given route.

        This method computes the energy required to overcome tractive efforts along the
        route, taking into account the efficiency of the electric engine. It also
        incorporates energy recovered through regenerative braking.

        Args:
            tractive_efforts (npt.NDArray[np.float64]): An array of tractive efforts (in
                Newtons) applied at different segments of the route.
            route (Route): The route object containing information about the distances
                of each segment.
            bus_mass (float): The mass of the bus in Kilograms (used for calculating the
                regenerative braking).

        Returns:
            npt.NDArray[np.float64]: An array of energy consumptions (in Joules)
            for each segment of the route, adjusted for regenerative braking.
        """
        tractive_effort_energy = (
            tractive_efforts * route.distances / self._efficiency
        ).astype(np.float64)
        energy_consumption = (
            tractive_effort_energy
            + self.compute_regenerative_braking(
                hill_climb_resistances, linear_acceleration_forces, route
            )
        )

        self._energy -= np.sum(energy_consumption)

        return energy_consumption
