"""Test the calculations for a single trip."""

import numpy as np
import pytest

from sidrobus.bus.bus import Bus
from sidrobus.bus.engine.fuel_engine import FuelEngine
from sidrobus.route import Route


@pytest.fixture
def test_route() -> Route:
    """Fixture to generate an example route."""
    times = np.array([1, 2, 3, 4, 5])
    longitudes = np.array([0, 1, 2, 3, 2])
    latitudes = np.array([0, 2, 1, 2, 5])
    heights = np.array([0, 1, 3, 0, 2])
    velocities = np.array([0, 1, 3, 4, 2])

    return Route(times, longitudes, latitudes, heights, velocities)


@pytest.fixture
def test_fuel_bus() -> Bus:
    """Fixture to generate an example fuel bus."""
    engine = FuelEngine(
        efficiency=0.2,
        mass=0,  # For fuel buses the engine mass is not considered
    )
    return Bus(
        engine=engine,
        frontal_area=7.65,
        mass=19500,
        aerodynamic_drag_coef=0.25,
        rolling_resistance_coef=0.01,
    )


def test_route_distances_calculation(test_route: Route) -> None:
    """Test the route distances calculation."""
    route = test_route
    expected_distances = np.array([2.449489743, 2.449489743, 3.31662479, 3.741657387])
    np.testing.assert_array_almost_equal(route.distances, expected_distances)


def test_route_angles_calcuations(test_route: Route) -> None:
    """Test the route angles calculation."""
    route = test_route
    expected_angles = np.array([0.420534335, 0.955316618, -1.130285664, 0.563942641])
    np.testing.assert_array_almost_equal(route.angles, expected_angles)


def test_accelerations_calculation(test_route: Route) -> None:
    """Test the route accelerations calculation."""
    route = test_route
    expected_accelerations = np.array([1, 2, 1, -2], dtype=float)
    np.testing.assert_array_almost_equal(route.accelerations, expected_accelerations)


def test_avg_velocities_calculation(test_route: Route) -> None:
    """Test the route average velocities calculation."""
    route = test_route
    expected_avg_velocities = np.array([0.5, 2, 3.5, 3])
    np.testing.assert_array_almost_equal(route.avg_velocities, expected_avg_velocities)


def test_rolling_resistance_calculation(test_fuel_bus: Bus) -> None:
    """Test the rolling resistance calculation."""
    bus = test_fuel_bus
    expected_rolling_resistance = 1912.29675
    calculated_rolling_resistance = bus._compute_route_rolling_resistance()  # noqa: SLF001
    np.testing.assert_almost_equal(
        calculated_rolling_resistance, expected_rolling_resistance
    )


def test_aerodynamic_drag_calculation(test_fuel_bus: Bus, test_route: Route) -> None:
    """Test the aerodynamic drag calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_aerodynamic_drag = np.array([0.2390625, 3.825, 11.7140625, 8.60625])
    calculated_aerodynamic_drag = bus._compute_route_aerodynamic_drag(route)  # noqa: SLF001
    np.testing.assert_allclose(calculated_aerodynamic_drag, expected_aerodynamic_drag)


def test_hill_climb_resistance_calculation(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the hill climb resistance calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_hill_climb_resistance = np.array(
        [78069.1879, 156138.3758, -172973.7493, 102216.5609]
    )
    calculated_hill_climb_resistance = bus._compute_route_hill_climb_resistance(route)  # noqa: SLF001
    np.testing.assert_allclose(
        calculated_hill_climb_resistance, expected_hill_climb_resistance
    )


def test_linear_acceleration_force_calculations(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the linear acceleration force calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_linear_acceleration_force = np.array([19500, 39000, 19500, -39000])
    calculated_linear_acceleration_force = bus._compute_linear_aceleration_force(route)  # noqa: SLF001
    np.testing.assert_allclose(
        calculated_linear_acceleration_force, expected_linear_acceleration_force
    )


def test_tractive_effort_calculations(test_fuel_bus: Bus, test_route: Route) -> None:
    """Test the tractive effort calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_tractive_effort = np.array(
        [100456.7237, 197092.2008, -152487.0352, 61275.16711]
    )
    calculated_tractive_effort = bus._calculate_route_forces(route)  # noqa: SLF001
    np.testing.assert_allclose(
        calculated_tractive_effort, expected_tractive_effort, rtol=1e-1
    )
