"""Module with for generating folium interactive maps."""

import folium
import numpy as np
import pandas as pd
from numpy import typing as npt

from sidrobus.route import Route


def plot_route_map(route: Route) -> folium.Map:
    """Plots a route as a polyline on a map using Folium.

    Args:
        route (Route): Route object containing coordinates, altitude, speed, and time.

    Returns:
        folium.Map: A Folium map object with the route plotted as a polyline.
    """
    m = folium.Map(location=[route.latitudes[0], route.longitudes[0]], zoom_start=13)

    # Base layer: simple route (always visible)
    fg_base = folium.FeatureGroup(name="Base route", show=True)
    folium.PolyLine(
        locations=list(zip(route.latitudes, route.longitudes, strict=False)),
        color="blue",
        weight=2.5,
        opacity=1,
        tooltip="Base route",
    ).add_to(fg_base)
    fg_base.add_to(m)

    # Helper for coloring gradients using a color list
    def color_gradient(values: npt.NDArray, color_list: list[str]) -> list[str]:
        norm = (values - np.min(values)) / (np.ptp(values) + 1e-9)
        # Interpolate color index
        idx = (norm * (len(color_list) - 1)).astype(int)
        return [color_list[i] for i in idx]

    # Define a single color list (viridis) for all gradients
    viridis_colors = [
        "#440154",
        "#482878",
        "#3E4989",
        "#31688E",
        "#26828E",
        "#1F9E89",
        "#35B779",
        "#6DCD59",
        "#B4DE2C",
        "#FDE725",
    ]

    # Altitude (hidden by default)
    if hasattr(route, "altitudes"):
        altitudes = route.altitudes
        colors = color_gradient(altitudes, viridis_colors)
        fg_alt = folium.FeatureGroup(name="Altitude Gradient", show=False)
        for i in range(len(altitudes) - 1):
            folium.PolyLine(
                locations=[
                    (route.latitudes[i], route.longitudes[i]),
                    (route.latitudes[i + 1], route.longitudes[i + 1]),
                ],
                color=colors[i],
                weight=4,
                opacity=0.8,
            ).add_to(fg_alt)
        fg_alt.add_to(m)

    # Speed (hidden by default)
    if hasattr(route, "speeds"):
        speeds = route.speeds
        colors = color_gradient(speeds, viridis_colors)
        fg_speed = folium.FeatureGroup(name="Speed Gradient", show=False)
        for i in range(len(speeds) - 1):
            folium.PolyLine(
                locations=[
                    (route.latitudes[i], route.longitudes[i]),
                    (route.latitudes[i + 1], route.longitudes[i + 1]),
                ],
                color=colors[i],
                weight=4,
                opacity=0.8,
            ).add_to(fg_speed)
        fg_speed.add_to(m)

    # Acceleration (hidden by default)
    if hasattr(route, "speeds") and hasattr(route, "times"):
        acc = np.diff(route.speeds) / np.diff(route.times)
        acc = np.clip(acc, -2, 2)
        colors = color_gradient(acc, viridis_colors)
        fg_acc = folium.FeatureGroup(name="Acceleration Gradient", show=False)
        for i in range(len(acc)):
            folium.PolyLine(
                locations=[
                    (route.latitudes[i], route.longitudes[i]),
                    (route.latitudes[i + 1], route.longitudes[i + 1]),
                ],
                color=colors[i],
                weight=4,
                opacity=0.8,
            ).add_to(fg_acc)
        fg_acc.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def plot_simulation_results_map(route: Route, results: pd.DataFrame) -> folium.Map:  # noqa: C901
    """Plots simulation results on an interactive map using Folium.

    Args:
        route (Route): Route object containing coordinates for the map.
        results (pd.DataFrame): DataFrame containing simulation results with consumption
            data.

    Returns:
        folium.Map: A Folium map object with simulation results visualized.
    """
    # Create base map centered at the first point of the route
    m = folium.Map(location=[route.latitudes[0], route.longitudes[0]], zoom_start=13)

    # Base layer: simple route (always visible)
    fg_base = folium.FeatureGroup(name="Base route", show=True)
    folium.PolyLine(
        locations=list(zip(route.latitudes, route.longitudes, strict=False)),
        color="blue",
        weight=2.5,
        opacity=1,
        tooltip="Base route",
    ).add_to(fg_base)
    fg_base.add_to(m)

    # Helper for coloring gradients using a color list
    def color_gradient(values: npt.NDArray, color_list: list[str]) -> list[str]:
        norm = (values - np.min(values)) / (np.ptp(values) + 1e-9)
        # Interpolate color index
        idx = (norm * (len(color_list) - 1)).astype(int)
        return [color_list[i] for i in idx]

    # Define a single color list (viridis) for all gradients
    viridis_colors = [
        "#440154",
        "#482878",
        "#3E4989",
        "#31688E",
        "#26828E",
        "#1F9E89",
        "#35B779",
        "#6DCD59",
        "#B4DE2C",
        "#FDE725",
    ]

    # Net consumption (shown by default)
    if "net_consumption" in results:
        net_consumption = results["net_consumption"]
        colors = color_gradient(net_consumption.to_numpy(), viridis_colors)
        fg_net = folium.FeatureGroup(name="Net Consumption", show=True)

        for i in range(len(net_consumption) - 1):
            folium.PolyLine(
                locations=[
                    (route.latitudes[i], route.longitudes[i]),
                    (route.latitudes[i + 1], route.longitudes[i + 1]),
                ],
                color=colors[i],
                weight=4,
                opacity=0.8,
                tooltip=f"Net Consumption: {net_consumption[i]:.2f} kWh",
            ).add_to(fg_net)
        fg_net.add_to(m)

    # Total consumption (hidden by default)
    if "consumption" in results:
        consumption = results["consumption"]
        colors = color_gradient(consumption.to_numpy(), viridis_colors)
        fg_cons = folium.FeatureGroup(name="Total Consumption", show=False)

        for i in range(len(consumption) - 1):
            folium.PolyLine(
                locations=[
                    (route.latitudes[i], route.longitudes[i]),
                    (route.latitudes[i + 1], route.longitudes[i + 1]),
                ],
                color=colors[i],
                weight=4,
                opacity=0.8,
                tooltip=f"Consumption: {consumption[i]:.2f} kWh",
            ).add_to(fg_cons)
        fg_cons.add_to(m)

    # Regeneration (hidden by default)
    if "regeneration" in results:
        regeneration = results["regeneration"]
        colors = color_gradient(regeneration.to_numpy(), viridis_colors)
        fg_regen = folium.FeatureGroup(name="Regeneration", show=False)

        for i in range(len(regeneration) - 1):
            folium.PolyLine(
                locations=[
                    (route.latitudes[i], route.longitudes[i]),
                    (route.latitudes[i + 1], route.longitudes[i + 1]),
                ],
                color=colors[i],
                weight=4,
                opacity=0.8,
                tooltip=f"Regeneration: {regeneration[i]:.2f} kWh",
            ).add_to(fg_regen)
        fg_regen.add_to(m)

    # Forces visualization (hidden by default)
    force_components = [
        "rolling_resistance",
        "aerodynamic_drag_resistance",
        "hill_climb_resistance",
        "linear_acceleration_force",
        "tractive_force",
    ]

    for component in force_components:
        if component in results:
            values = results[component]
            colors = color_gradient(values.to_numpy(), viridis_colors)
            # Format component name for display
            display_name = component.replace("_", " ").title()
            fg_force = folium.FeatureGroup(name=f"{display_name} Force", show=False)

            for i in range(len(values) - 1):
                folium.PolyLine(
                    locations=[
                        (route.latitudes[i], route.longitudes[i]),
                        (route.latitudes[i + 1], route.longitudes[i + 1]),
                    ],
                    color=colors[i],
                    weight=4,
                    opacity=0.8,
                    tooltip=f"{display_name}: {values[i]:.2f} N",
                ).add_to(fg_force)
            fg_force.add_to(m)

    # Emissions visualization (hidden by default)
    emission_components = [
        "co_emissions",
        "nox_emissions",
        "hc_emissions",
        "pm_emissions",
    ]

    for component in emission_components:
        if component in results:
            values = results[component]
            colors = color_gradient(values.to_numpy(), viridis_colors)
            # Format component name for display
            display_name = (
                component.replace("_", " ").replace("emissions", "Emissions").upper()
            )
            fg_emission = folium.FeatureGroup(name=f"{display_name}", show=False)

            for i in range(len(values) - 1):
                folium.PolyLine(
                    locations=[
                        (route.latitudes[i], route.longitudes[i]),
                        (route.latitudes[i + 1], route.longitudes[i + 1]),
                    ],
                    color=colors[i],
                    weight=4,
                    opacity=0.8,
                    tooltip=f"{display_name}: {values[i]:.4f} g",
                ).add_to(fg_emission)
            fg_emission.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m
