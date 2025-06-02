"""Module for interactive plots."""

import plotly.graph_objects as go

from sidrobus.route import Route


def plot_route_data(route: Route) -> go.Figure:
    """Creates an interactive Plotly graph to visualize route data over time.

    The graph displays time on the x-axis and allows the user to switch between
    altitude, speed, or acceleration on the y-axis using a dropdown menu.
    Hovering over points shows all relevant data properties.

    Args:
        route (Route): A Route object containing the data to visualize

    Returns:
        go.Figure: A Plotly figure object with the interactive visualization
    """
    # Create figure
    fig = go.Figure()

    # Add traces for each metric
    # Altitude trace
    fig.add_trace(
        go.Scatter(
            x=route.times,
            y=route.altitudes,
            mode="lines+markers",
            name="Altitude",
            hovertemplate="<b>Time</b>: %{x:.2f}s<br>"
            + "<b>Altitude</b>: %{y:.2f}m<br>"
            + "<b>Speed</b>: %{customdata[0]:.2f}m/s<br>"
            + "<b>Acceleration</b>: %{customdata[1]:.2f}m/s²<br>"
            + "<extra></extra>",
            visible=True,
            customdata=[
                [speed, 0] for speed in route.speeds
            ],  # Placeholder for acceleration at endpoints
        )
    )

    # Speed trace
    fig.add_trace(
        go.Scatter(
            x=route.times,
            y=route.speeds,
            mode="lines+markers",
            name="Speed",
            hovertemplate="<b>Time</b>: %{x:.2f}s<br>"
            + "<b>Speed</b>: %{y:.2f}m/s<br>"
            + "<b>Altitude</b>: %{customdata[0]:.2f}m<br>"
            + "<b>Acceleration</b>: %{customdata[1]:.2f}m/s²<br>"
            + "<extra></extra>",
            visible=False,
            customdata=[
                [alt, 0] for alt in route.altitudes
            ],  # Placeholder for acceleration at endpoints
        )
    )

    # Acceleration trace - accelerations array is one element shorter than the others
    accel_times = route.times[:-1]  # Using time points except the last one
    accel_values = route.accelerations
    # For acceleration trace, we need matching altitude and speed data
    altitude_accel = route.altitudes[:-1]
    speed_accel = route.speeds[:-1]

    fig.add_trace(
        go.Scatter(
            x=accel_times,
            y=accel_values,
            mode="lines+markers",
            name="Acceleration",
            hovertemplate="<b>Time</b>: %{x:.2f}s<br>"
            + "<b>Acceleration</b>: %{y:.2f}m/s²<br>"
            + "<b>Altitude</b>: %{customdata[0]:.2f}m<br>"
            + "<b>Speed</b>: %{customdata[1]:.2f}m/s<br>"
            + "<extra></extra>",
            visible=False,
            customdata=list(zip(altitude_accel, speed_accel, strict=False)),
        )
    )

    # Create dropdown menu for selecting which data to display
    dropdown_buttons = [
        {
            "label": "Altitude",
            "method": "update",
            "args": [
                {"visible": [True, False, False]},
                {"title": "Altitude over Time", "yaxis": {"title": "Altitude (m)"}},
            ],
        },
        {
            "label": "Speed",
            "method": "update",
            "args": [
                {"visible": [False, True, False]},
                {"title": "Speed over Time", "yaxis": {"title": "Speed (m/s)"}},
            ],
        },
        {
            "label": "Acceleration",
            "method": "update",
            "args": [
                {"visible": [False, False, True]},
                {
                    "title": "Acceleration over Time",
                    "yaxis": {"title": "Acceleration (m/s²)"},
                },
            ],
        },
    ]

    # Update layout with dropdown menu and initial settings
    fig.update_layout(
        title="Altitude over Time",
        xaxis_title="Time (s)",
        yaxis_title="Altitude (m)",
        hovermode="closest",
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.15,
            }
        ],
    )

    return fig
