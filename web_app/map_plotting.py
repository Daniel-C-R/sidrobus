"""Module with for generating folium interactive maps."""

import folium
import numpy as np
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
