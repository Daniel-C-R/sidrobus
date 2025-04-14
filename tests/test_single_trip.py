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
        mass=19500,
    )
    return Bus(
        engine=engine,
        frontal_area=10,
        mass=5000,
        aerodynamic_drag_coef=0.3,
        rolling_resistance_coef=0.01,
    )


def test_route_distances_calculation(test_route: Route) -> None:
    """Test the route distances calculation."""
    route = test_route
    expected_distances = np.array([2.449489743, 2.449489743, 3.31662479, 3.741657387])
    np.testing.assert_array_almost_equal(route.distances, expected_distances)
