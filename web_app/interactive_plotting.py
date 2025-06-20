"""Module for interactive plots."""

import altair as alt
import numpy as np
import pandas as pd
from numpy import typing as npt

from sidrobus.route import Route

# Theme colors
THEME_COLORS = {
    "primary": "#2c4f2d",
    "background": "#f9f3e5",
    "secondary_bg": "#cddac4",
    "text": "#21392b",
}


def plot_route_data(route: Route) -> alt.VConcatChart:
    """Creates an interactive Altair chart to visualize route data over time.

    The graph displays time on the x-axis and allows the user to switch between
    altitude, speed, or acceleration on the y-axis using a selection.
    Hovering over points shows all relevant data properties.
    A time selector allows viewing specific time intervals.

    Args:
        route (Route): A Route object containing the data to visualize

    Returns:
        alt.VConcatChart: An Altair chart with time selection capabilities
    """
    # Prepare data
    # Create a main DataFrame with altitude and speed data
    main_data = pd.DataFrame(
        {
            "time": route.times,
            "altitude": route.altitudes,
            "speed": route.speeds,
            "measure": ["Altitude" for _ in route.times],  # Default measure
        }
    )

    # Create a separate DataFrame for acceleration data (which is one element shorter)
    accel_data = pd.DataFrame(
        {
            "time": route.times[:-1],
            "acceleration": route.accelerations,
            "altitude": route.altitudes[:-1],  # Add altitude for tooltip
            "speed": route.speeds[:-1],  # Add speed for tooltip
            "measure": ["Acceleration" for _ in route.times[:-1]],
        }
    )

    # Create a selection for choosing the measure to display
    measure_selection = alt.param(
        name="measure_select",
        value="Altitude",  # Default selection
        bind=alt.binding_select(
            options=["Altitude", "Speed", "Acceleration"], name="Select Measure: "
        ),
    )

    # Create a brush selection for the time interval
    time_brush = alt.selection_interval(
        encodings=["x"],
    )

    # Create conditional time selectors for each measure
    altitude_selector = (
        alt.Chart(main_data)
        .mark_area(color=THEME_COLORS["primary"], opacity=0.3)
        .encode(
            x=alt.X("time:Q", title="Time (s)"),
            y=alt.Y("altitude:Q", title="", axis=None),
        )
        .add_params(time_brush)
        .transform_filter(measure_selection == "Altitude")
    )

    speed_selector = (
        alt.Chart(main_data)
        .mark_area(color=THEME_COLORS["primary"], opacity=0.3)
        .encode(
            x=alt.X("time:Q", title="Time (s)"), y=alt.Y("speed:Q", title="", axis=None)
        )
        .add_params(time_brush)
        .transform_filter(measure_selection == "Speed")
    )

    acceleration_selector = (
        alt.Chart(accel_data)
        .mark_area(color=THEME_COLORS["primary"], opacity=0.3)
        .encode(
            x=alt.X("time:Q", title="Time (s)"),
            y=alt.Y("acceleration:Q", title="", axis=None),
        )
        .add_params(time_brush)
        .transform_filter(measure_selection == "Acceleration")
    )

    # Combine selectors into a single layer
    time_selector = alt.layer(
        altitude_selector, speed_selector, acceleration_selector
    ).properties(width=700, height=60, title="Time Selector")

    # Create a chart for altitude and speed data (they share the same time points)
    base_chart = (
        alt.Chart(main_data)
        .encode(
            x=alt.X(
                "time:Q",
                title="Time (s)",
                scale=alt.Scale(
                    domain=time_brush
                ),  # Use the brush selection to filter time domain
            )
        )
        .properties(
            width=700,
            height=400,
            title="Route Data Visualization",
        )
    )

    # Create the conditional visualization with three possible views
    main_chart = alt.layer(
        # Altitude visualization
        base_chart.mark_line(
            color=THEME_COLORS["primary"],
            point=alt.OverlayMarkDef(color=THEME_COLORS["primary"]),
        )
        .encode(
            y=alt.Y("altitude:Q", title="Altitude (m)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("time:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("altitude:Q", title="Altitude (m)", format=".2f"),
                alt.Tooltip("speed:Q", title="Speed (m/s)", format=".2f"),
            ],
        )
        .transform_filter(measure_selection == "Altitude"),
        # Speed visualization
        base_chart.mark_line(
            color=THEME_COLORS["primary"],
            point=alt.OverlayMarkDef(color=THEME_COLORS["primary"]),
        )
        .encode(
            y=alt.Y("speed:Q", title="Speed (m/s)"),
            tooltip=[
                alt.Tooltip("time:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("speed:Q", title="Speed (m/s)", format=".2f"),
                alt.Tooltip("altitude:Q", title="Altitude (m)", format=".2f"),
            ],
        )
        .transform_filter(measure_selection == "Speed"),
        # Acceleration visualization (on a different dataset)
        alt.Chart(accel_data)
        .encode(
            x=alt.X(
                "time:Q",
                title="Time (s)",
                scale=alt.Scale(domain=time_brush),  # Use the brush selection here too
            )
        )
        .mark_line(
            color=THEME_COLORS["primary"],
            point=alt.OverlayMarkDef(color=THEME_COLORS["primary"]),
        )
        .encode(
            y=alt.Y("acceleration:Q", title="Acceleration (m/s²)"),
            tooltip=[
                alt.Tooltip("time:Q", title="Time (s)", format=".2f"),
                alt.Tooltip(
                    "acceleration:Q", title="Acceleration (m/s²)", format=".2f"
                ),
                alt.Tooltip("altitude:Q", title="Altitude (m)", format=".2f"),
                alt.Tooltip("speed:Q", title="Speed (m/s)", format=".2f"),
            ],
        )
        .transform_filter(measure_selection == "Acceleration"),
    ).add_params(measure_selection)

    # Vertically concatenate the time selector and the main chart
    return alt.vconcat(main_chart, time_selector).configure_axis(
        labelPadding=10,  # Add padding between axis and labels
        titlePadding=10,  # Add padding between axis and title
    )


def plot_simulation_results(
    times: npt.NDArray[np.float64], results: pd.DataFrame
) -> alt.VConcatChart:
    """Creates an interactive Altair chart to visualize simulation results over time.

    The graph displays time on the x-axis and allows the user to switch between
    different simulation metrics on the y-axis using a selection.
    Hovering over points shows all relevant data properties.
    Includes a time interval selector for zooming into specific periods.

    Args:
        times (npt.NDArray[np.float64]): Array of time points
        results (pd.DataFrame): DataFrame containing simulation results per segment

    Returns:
        alt.VConcatChart: An Altair chart with interactive selection capabilities
    """
    # Skip the first time point to match segment results (N-1 elements)
    segment_times = times[1:]

    # Prepare data for plotting
    data = pd.DataFrame()

    # Add all the columns from results with time as a common axis
    for column in results.columns:
        if len(results[column]) == len(segment_times):
            data_slice = pd.DataFrame(
                {"time": segment_times, "value": results[column], "metric": column}
            )
            data = pd.concat([data, data_slice], ignore_index=True)

    # Create a selection for choosing the metric to display
    default_metric = (
        "net_consumption"
        if "net_consumption" in results.columns
        else data["metric"].unique()[0]
    )

    metric_selection = alt.param(
        name="metric_select",
        value=default_metric,
        bind=alt.binding_select(
            options=sorted(data["metric"].unique().tolist()), name="Select Metric: "
        ),
    )

    # Time interval selector (brush)
    time_brush = alt.selection_interval(encodings=["x"])

    # Time selector area chart (shows the selected metric)
    time_selector = (
        alt.Chart(data)
        .mark_area(opacity=0.3, color=THEME_COLORS["secondary_bg"])
        .encode(
            x=alt.X("time:Q", title="Time (s)"),
            y=alt.Y("value:Q", title="", axis=None),
        )
        .transform_filter(alt.datum.metric == metric_selection)
        .add_params(time_brush)
        .properties(width=700, height=60, title="Time Selector")
    )

    # Main chart with time brush and metric selection
    main_chart = (
        alt.Chart(data)
        .encode(
            x=alt.X(
                "time:Q",
                title="Time (s)",
                scale=alt.Scale(domain=time_brush),
            ),
        )
        .mark_line(point=True)
        .encode(
            y=alt.Y("value:Q", title="Value"),
            color=alt.value(THEME_COLORS["primary"]),
            tooltip=[
                alt.Tooltip("time:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
                alt.Tooltip("metric:N", title="Metric"),
            ],
        )
        .transform_filter(alt.datum.metric == metric_selection)
        .add_params(metric_selection)
        .properties(width=700, height=400, title="Simulation Results Over Time")
    )

    return alt.vconcat(main_chart, time_selector).configure_axis(
        labelPadding=10,
        titlePadding=10,
    )
