"""Test the calculations for a single trip."""

import numpy as np
import pytest

from sidrobus.bus import Bus
from sidrobus.bus.engine import ElectricEngine, FuelEngine
from sidrobus.route import Route
from sidrobus.unit_conversions import kwh_to_joules


@pytest.fixture
def test_route() -> Route:
    """Fixture to generate an example route."""
    times = np.array([1323.017, 1356.017, 1359.017, 1362.017, 1372.017])
    longitudes = np.array(
        [-5.8440438, -5.84410455, -5.84418669, -5.84425666, -5.84433499]
    )
    latitudes = np.array(
        [43.36878569, 43.36867881, 43.36857426, 43.36847559, 43.3683948]
    )
    altitudes = np.array([261.74, 267.84, 265.21, 262.04, 261.29])
    speeds = np.array([2.25, 3.59, 4.5, 4, 0.52])

    return Route(times, longitudes, latitudes, altitudes, speeds)


@pytest.fixture
def test_fuel_bus() -> Bus:
    """Fixture to generate an example fuel bus."""
    engine = FuelEngine(
        efficiency=0.35,
        mass=0,  # For fuel buses the engine mass is not considered
        capacity=9.3236e9,
    )
    return Bus(
        model_name="Test Fuel Bus",
        model_manufacturer="None",
        engine=engine,
        frontal_area=8.67,
        mass=13500,
        aerodynamic_drag_coef=0.6,
        rolling_resistance_coef=0.01,
    )


@pytest.fixture
def test_electric_bus() -> Bus:
    """Fixture to generate an example electric bus."""
    engine = ElectricEngine(
        efficiency=0.9,
        mass=0,  # TODO: Add mass for the electric engine in the future
        regenerative_braking_efficiency=0.5,
        capacity=kwh_to_joules(686),
    )
    return Bus(
        model_name="Test Electric Bus",
        model_manufacturer="None",
        engine=engine,
        frontal_area=8.67,
        mass=13500,
        aerodynamic_drag_coef=0.6,
        rolling_resistance_coef=0.01,
    )


def test_route_distances_calculation(test_route: Route) -> None:
    """Test the route distances calculation.

    The expected distances are calculated using the haversine formula, which is a bit
    more precise, but the difference is minimal.
    """
    route = test_route
    expected_distances = np.array([14.233, 13.644, 12.744, 11.016])
    np.testing.assert_allclose(route.distances, expected_distances, rtol=1e-4)


def test_accelerations_calculation(test_route: Route) -> None:
    """Test the route accelerations calculation."""
    route = test_route
    expected_accelerations = np.array([0.040606061, 0.303333333, -0.166666667, -0.348])
    np.testing.assert_array_almost_equal(route.accelerations, expected_accelerations)


def test_avg_speeds_calculation(test_route: Route) -> None:
    """Test the route average speeds calculation."""
    route = test_route
    expected_avg_speeds = np.array([2.92, 4.045, 4.25, 2.26])
    np.testing.assert_array_almost_equal(route.avg_speeds, expected_avg_speeds)


def test_rolling_resistance_calculation(test_fuel_bus: Bus) -> None:
    """Test the rolling resistance calculation."""
    bus = test_fuel_bus
    expected_rolling_resistance = 1324.35
    calculated_rolling_resistance = bus.compute_route_rolling_resistance_forces()
    np.testing.assert_almost_equal(
        calculated_rolling_resistance, expected_rolling_resistance
    )


def test_aerodynamic_drag_calculation(test_fuel_bus: Bus, test_route: Route) -> None:
    """Test the aerodynamic drag calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_aerodynamic_drag = np.array(
        [27.721458, 53.19703378, 58.72570313, 16.6060845]
    )
    calculated_aerodynamic_drag = bus.compute_route_aerodynamic_drag_forces(route)
    np.testing.assert_allclose(calculated_aerodynamic_drag, expected_aerodynamic_drag)


def test_hill_climb_resistance_calculation(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the hill climb resistance calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_hill_climb_resistance = np.array(
        [40922.415, -25528.43829, -32941.99428, -9016.444815]
    )
    calculated_hill_climb_resistance = bus.compute_route_hill_climb_resistance_forces(
        route
    )
    np.testing.assert_allclose(
        calculated_hill_climb_resistance, expected_hill_climb_resistance, rtol=1e-1
    )


def test_linear_acceleration_force_calculations(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the linear acceleration force calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_linear_acceleration_force = np.array([548.1818182, 4095, -2250, -4698])
    calculated_linear_acceleration_force = bus.compute_linear_acceleration_forces(route)
    np.testing.assert_allclose(
        calculated_linear_acceleration_force, expected_linear_acceleration_force
    )


def test_tractive_effort_calculations(test_fuel_bus: Bus, test_route: Route) -> None:
    """Test the tractive effort calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_tractive_effort = np.array([42850.07737, 0, 0, 0])
    calculated_tractive_effort = bus.compute_route_tractive_efforts(route)
    np.testing.assert_allclose(
        calculated_tractive_effort, expected_tractive_effort, rtol=1e-1
    )


def test_fuel_bus_consumption_calculation(
    test_fuel_bus: Bus, test_route: Route
) -> None:
    """Test the consumption calculation."""
    route = test_route
    bus = test_fuel_bus
    expected_consumption = np.array([1.18223e08, 0, 0, 0])
    calculated_consumption = bus.compute_route_consumption(route)
    np.testing.assert_allclose(calculated_consumption, expected_consumption, rtol=1e-1)


def test_regenerative_braking_calculation(
    test_electric_bus: Bus, test_route: Route
) -> None:
    """Test the regenerative braking calculation."""
    route = test_route
    bus = test_electric_bus
    expected_regenerative_braking = np.array(
        [0, -174152.025, -224963.5464, -76833.83744]
    )
    if not isinstance(bus._engine, ElectricEngine):  # noqa: SLF001
        pytest.skip("This test is only applicable for electric buses.")

    calculated_regenerative_braking = bus._engine.compute_regenerative_braking(  # noqa: SLF001
        bus.compute_route_hill_climb_resistance_forces(route),
        bus.compute_linear_acceleration_forces(route),
        route,
    )
    np.testing.assert_allclose(
        calculated_regenerative_braking, expected_regenerative_braking
    )


def test_electric_bus_consumption_calculation(
    test_electric_bus: Bus, test_route: Route
) -> None:
    """Test the energy consumption calculation for an electric bus on a given route."""
    route = test_route
    bus = test_electric_bus
    expected_consumption = np.array([6.77629e05, 88111.50959, -3402.75, 100793.0441])
    calculated_consumption = bus.compute_route_consumption(route)
    np.testing.assert_allclose(calculated_consumption, expected_consumption, rtol=1e-1)
