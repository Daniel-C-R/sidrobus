"""Web UI for the simulator."""

import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from sidrobus.bus import Bus
from sidrobus.bus.engine import ElectricEngine, FuelEngine
from sidrobus.constants import DIESEL_LHV
from sidrobus.route import Route
from sidrobus.unit_conversions import joules_to_kwh, kwh_to_joules
from web_app.interactive_plotting import plot_route_data
from web_app.map_plotting import plot_route_map, plot_simulation_results_map


def snake_to_title(snake_str: str) -> str:
    """Convert snake_case string to Title Case."""
    return snake_str.replace("_", " ").title()


st.title("Bus Simulator")

with st.sidebar:
    st.header("Route Loader")
    route_file = st.file_uploader("Upload Route", type=["csv"])

    st.header("Bus Parameters")

    mass = st.number_input("Bus Mass (kg)", min_value=1000, value=12000)
    rolling_resistance_coef = st.number_input(
        "Rolling Resistance Coefficient", min_value=0.0, value=0.01, step=0.001
    )
    frontal_area = st.number_input(
        "Frontal Area (m^2)", min_value=0.0, value=5.0, step=0.01
    )
    aerodynamic_drag_coef = st.number_input(
        "Aerodynamic Drag Coefficient", min_value=0.0, value=0.6, step=0.01
    )
    energy_efficiency = st.number_input(
        "Energy efficiency [0, 1]", min_value=0.0, max_value=1.0, value=0.9, step=0.01
    )

    engine_type = st.selectbox(
        "Engine Type",
        options=["Electric", "Diesel"],
        index=0,
    )

    energy_capacity = 0.0
    regenerative_braking_efficiency = 0.0

    if engine_type == "Electric":
        energy_capacity = st.number_input(
            "Battery Capacity (kWh)", min_value=0.0, value=100.0
        )
        regenerative_braking_efficiency = st.number_input(
            "Regenerative Braking Efficiency [0, 1]",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
    elif engine_type == "Diesel":
        energy_capacity = (
            st.number_input("Fuel Tank Capacity (liters)", min_value=1.0, value=100.0)
            * DIESEL_LHV
        )

    run = st.button(
        "Run Simulation",
        disabled=route_file is None,
    )

if route_file is not None:
    st.header("Route details")

    route_data = pd.read_csv(route_file)
    route = Route.from_dataframe(route_data)

    with st.expander("Route Data"):
        st.dataframe(route_data)

    st.subheader("Plots")

    route_map = plot_route_map(route)
    folium_static(
        route_map,
        width=700,
        height=500,
    )

    route_plot = plot_route_data(route)
    st.plotly_chart(
        route_plot,
        use_container_width=True,
        responsive=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "sendDataToCloud",
                "editInChartStudio",
                "zoom2d",
                "select2d",
                "pan2d",
                "lasso2d",
                "autoScale2d",
                "resetScale2d",
            ],
        },
    )

    route_summary = route.summary

    st.subheader("Route Summary")

    st.write("Total distance (m): ", f"{route_summary['total_distance']:.2f}")
    st.write("Number of points: ", route_summary["number_of_points"])
    st.write("Start time: ", route_summary["start_time"])
    st.write("End time: ", route_summary["end_time"])
    st.write("Duration (s): ", f"{route_summary['duration']:.2f}")
    st.write("Min altitude (m): ", f"{route_summary['min_altitude']:.2f}")
    st.write("Max altitude (m): ", f"{route_summary['max_altitude']:.2f}")
    st.write("Avg altitude (m): ", f"{route_summary['avg_altitude']:.2f}")
    st.write("Min speed (m/s): ", f"{route_summary['min_speed']:.2f}")
    st.write("Max speed (m/s): ", f"{route_summary['max_speed']:.2f}")
    st.write("Avg speed (m/s): ", f"{route_summary['avg_speed']:.2f}")
    st.write("Min acceleration (m/s²): ", f"{route_summary['min_acceleration']:.2f}")
    st.write("Max acceleration (m/s²): ", f"{route_summary['max_acceleration']:.2f}")
    st.write("Avg acceleration (m/s²): ", f"{route_summary['avg_acceleration']:.2f}")

    if run:
        # --- Prepare bus and engine ---
        if engine_type == "Electric":
            engine = ElectricEngine(
                efficiency=energy_efficiency,
                capacity=kwh_to_joules(energy_capacity),
                mass=0,
                regenerative_braking_efficiency=regenerative_braking_efficiency,
            )
        else:
            engine = FuelEngine(
                efficiency=energy_efficiency,
                capacity=energy_capacity,
                mass=0,
            )
        bus = Bus(
            model_name="",
            model_manufacturer="",
            mass=mass,
            engine=engine,
            frontal_area=frontal_area,
            aerodynamic_drag_coef=aerodynamic_drag_coef,
            rolling_resistance_coef=rolling_resistance_coef,
        )

        st.header("Simulation Results")

        simulation_results = bus.simulate_trip(route, modify_bus=False)
        results_per_segment = pd.DataFrame(simulation_results["results_per_segment"])

        st.subheader("Simulation summary")

        st.write("Simulation type: ", simulation_results["simulation_type"])
        st.write(
            "Total Rolling Resistance Force (N): ",
            f"{simulation_results['total_rolling_resistance_force']:.2f}",
        )
        st.write(
            "Total Aerodynamic Drag Force (N): ",
            f"{simulation_results['total_aerodynamic_drag_force']:.2f}",
        )
        st.write(
            "Total Hill Climb Resistance Force (N): ",
            f"{simulation_results['total_hill_climb_resistance_force']:.2f}",
        )
        st.write(
            "Total Linear Acceleration Force (N): ",
            f"{simulation_results['total_linear_acceleration_force']:.2f}",
        )
        st.write(
            "Total Tractive Force (N): ",
            f"{simulation_results['total_tractive_force']:.2f}",
        )
        st.write(
            "Total Rolling Resistance Consumption (kWh): ",
            f"{joules_to_kwh(simulation_results['total_rolling_resistance_consumption']):.2f}",
        )
        st.write(
            "Total Aerodynamic Drag Consumption (kWh): ",
            f"{joules_to_kwh(simulation_results['total_aerodynamic_drag_consumption']):.2f}",
        )
        st.write(
            "Total Hill Climb Consumption (kWh): ",
            f"{joules_to_kwh(simulation_results['total_hill_climb_consumption']):.2f}",
        )
        st.write(
            "Total Linear Acceleration Consumption (kWh): ",
            f"{joules_to_kwh(simulation_results['total_linear_acceleration_consumption']):.2f}",
        )
        st.write(
            "Total Consumption (kWh): ",
            f"{joules_to_kwh(simulation_results['total_consumption']):.2f}",
        )
        st.write(
            "Total Regeneration (kWh): ",
            f"{joules_to_kwh(simulation_results['total_regeneration']):.2f}",
        )
        st.write(
            "Total Net Consumption (kWh): ",
            f"{joules_to_kwh(simulation_results['total_net_consumption']):.2f}",
        )
        st.write(
            "Percentage Consumption (%): ",
            f"{simulation_results['percentage_consumption']:.2f}",
        )
        st.write(
            "Net Consumption per km (kWh/km): ",
            f"{joules_to_kwh(simulation_results['net_consumption_per_km']):.2f}",
        )
        st.write(
            "Net Consumption per 100km (kWh/100km): ",
            f"{joules_to_kwh(simulation_results['net_consumption_per_100km']):.2f}",
        )

        st.subheader("Simulation results per segment")
        st.dataframe(results_per_segment)

        st.subheader("Simulation results map")

        results_map = plot_simulation_results_map(route, results_per_segment)
        folium_static(
            results_map,
            width=700,
            height=500,
        )
