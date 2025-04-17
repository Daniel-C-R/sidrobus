"""Test the calculations for a single trip."""

import numpy as np
import pytest

from sidrobus.bus.bus import Bus
from sidrobus.bus.engine.fuel_engine import FuelEngine
from sidrobus.route import Route


@pytest.fixture
def test_route() -> Route:
    """Fixture to generate an example route.

    This example route are five points taken with the MatLab mobile app, near the
    Polytechnical School of Gijón, on board of a Solaris Urbino 12 electric bus.
    """
    times = np.array([1, 2, 3, 4, 5])
    longitudes = np.array([-5.6339, -5.634, -5.6341, -5.6343, -5.6344])
    latitudes = np.array([43.5237, 43.5236, 43.5236, 43.5235, 43.5235])
    heights = np.array([66.044, 65.564, 65.349, 65.266, 65.343])
    velocities = np.array([8.235, 8.663, 9.295, 8.946, 9.309])

    return Route(times, longitudes, latitudes, heights, velocities)


@pytest.fixture
def test_fuel_bus() -> Bus:
    """Fixture to generate an example fuel bus.

    The parameters are taken from a Mercedes-Benz Citaro fuel bus.
    """
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
    """Test the route distances calculation.

    The expected distances are calculated using the haversine formula, which is a bit
    more precise, but the difference is minimal.
    """
    route = test_route
    expected_distances = np.array([13.74334743, 8.06550769, 19.58762861, 8.06302262])
    np.testing.assert_array_almost_equal(route.distances, expected_distances, decimal=5)


def test_accelerations_calculation(test_route: Route) -> None:
    """Test the route accelerations calculation."""
    route = test_route
    expected_accelerations = np.array([0.428, 0.632, -0.349, 0.363])
    np.testing.assert_array_almost_equal(route.accelerations, expected_accelerations)


def test_avg_velocities_calculation(test_route: Route) -> None:
    """Test the route average velocities calculation."""
    route = test_route
    expected_avg_velocities = np.array([8.449, 8.979, 9.1205, 9.1275])
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
    expected_aerodynamic_drag = np.array(
        [68.26248096, 77.09520921, 79.54424124, 79.66638879]
    )
    calculated_aerodynamic_drag = bus._compute_route_aerodynamic_drag(route)  # noqa: SLF001
    np.testing.assert_allclose(calculated_aerodynamic_drag, expected_aerodynamic_drag)


def test_hill_climb_resistance_calculation(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the hill climb resistance calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_hill_climb_resistance = np.array(
        [-6678.885509, -5097.556373, -810.3105966, 1826.19914]
    )
    calculated_hill_climb_resistance = bus._compute_route_hill_climb_resistance(route)  # noqa: SLF001
    np.testing.assert_allclose(
        calculated_hill_climb_resistance, expected_hill_climb_resistance, rtol=1e-1
    )


def test_linear_acceleration_force_calculations(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the linear acceleration force calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_linear_acceleration_force = np.array([8346, 12324, -6805.5, 7078.5])
    calculated_linear_acceleration_force = bus._compute_linear_acceleration_force(route)  # noqa: SLF001
    np.testing.assert_allclose(
        calculated_linear_acceleration_force, expected_linear_acceleration_force
    )


def test_tractive_effort_calculations(test_fuel_bus: Bus, test_route: Route) -> None:
    """Test the tractive effort calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_tractive_effort = np.array(
        [4064.973722, 9832.035586, -5964.244605, 11250.58728]
    )
    calculated_tractive_effort = bus._calculate_route_forces(route)  # noqa: SLF001
    np.testing.assert_allclose(
        calculated_tractive_effort, expected_tractive_effort, rtol=1e-1
    )
