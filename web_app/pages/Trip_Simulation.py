"""Web UI for the simulator."""

import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from sidrobus.bus import Bus
from sidrobus.bus.emissions_standard import EmissionsStandard
from sidrobus.bus.emissions_standard.euro_standards import EURO_VI
from sidrobus.bus.engine import ElectricEngine, FuelEngine
from sidrobus.constants import DIESEL_LHV
from sidrobus.route import Route
from sidrobus.unit_conversions import joules_to_kwh, kwh_to_joules
from web_app.interactive_plotting import plot_route_data, plot_simulation_results
from web_app.map_plotting import plot_route_map, plot_simulation_results_map


def joules_to_diesel_liters(joules: float) -> float:
    """Convert joules to diesel liters.

    Parameters:
    joules (float): Energy in joules.

    Returns:
    float: Volume in diesel liters.
    """
    return joules / DIESEL_LHV


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
    # Default values from EURO VI standard
    co_per_kwh = EURO_VI.co_per_kwh
    nox_per_kwh = EURO_VI.nox_per_kwh
    hc_per_kwh = EURO_VI.hc_per_kwh
    pm_per_kwh = EURO_VI.pm_per_kwh

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

        # Custom emissions parameters
        st.write("**Emissions Parameters (g/kWh)**")
        st.caption("Default values correspond to EURO VI standard")
        co_per_kwh = st.number_input(
            "CO (g/kWh)",
            min_value=0.0,
            value=EURO_VI.co_per_kwh,
            step=0.01,
            help="Carbon monoxide emissions in grams per kilowatt-hour",
        )
        nox_per_kwh = st.number_input(
            "NOx (g/kWh)",
            min_value=0.0,
            value=EURO_VI.nox_per_kwh,
            step=0.01,
            help="Nitrogen oxides emissions in grams per kilowatt-hour",
        )
        hc_per_kwh = st.number_input(
            "HC (g/kWh)",
            min_value=0.0,
            value=EURO_VI.hc_per_kwh,
            step=0.01,
            help="Hydrocarbon emissions in grams per kilowatt-hour",
        )
        pm_per_kwh = st.number_input(
            "PM (g/kWh)",
            min_value=0.0,
            value=EURO_VI.pm_per_kwh,
            step=0.01,
            help="Particulate matter emissions in grams per kilowatt-hour",
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
    st.altair_chart(
        route_plot,
        use_container_width=True,
    )

    route_summary = route.summary

    st.subheader("Route Summary")

    # Row 1: Basic route info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Distance (m)", f"{route_summary['total_distance']:.2f}")
    with col2:
        st.metric("Number of Points", route_summary["number_of_points"])
    with col3:
        st.metric("Duration (s)", f"{route_summary['duration']:.2f}")

    # Row 2: Time info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Start Time", route_summary["start_time"])
    with col2:
        st.metric("End Time", route_summary["end_time"])
    with col3:
        # Empty placeholder or we could add another relevant metric
        st.empty()

    # Row 3: Altitude metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Altitude (m)", f"{route_summary['min_altitude']:.2f}")
    with col2:
        st.metric("Max Altitude (m)", f"{route_summary['max_altitude']:.2f}")
    with col3:
        st.metric("Avg Altitude (m)", f"{route_summary['avg_altitude']:.2f}")

    # Row 4: Speed metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Speed (m/s)", f"{route_summary['min_speed']:.2f}")
    with col2:
        st.metric("Max Speed (m/s)", f"{route_summary['max_speed']:.2f}")
    with col3:
        st.metric("Avg Speed (m/s)", f"{route_summary['avg_speed']:.2f}")

    # Row 5: Acceleration metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Acceleration (m/s²)", f"{route_summary['min_acceleration']:.2f}")
    with col2:
        st.metric("Max Acceleration (m/s²)", f"{route_summary['max_acceleration']:.2f}")
    with col3:
        st.metric("Avg Acceleration (m/s²)", f"{route_summary['avg_acceleration']:.2f}")

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
            # Create a custom emissions standard with the input values
            custom_emissions_standard = EmissionsStandard(
                co_per_kwh=co_per_kwh,
                nox_per_kwh=nox_per_kwh,
                hc_per_kwh=hc_per_kwh,
                pm_per_kwh=pm_per_kwh,
                name="Custom Emissions Standard",
            )

            engine = FuelEngine(
                efficiency=energy_efficiency,
                capacity=energy_capacity,
                mass=0,
                emissions_standard=custom_emissions_standard,
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

        # Simulation type row
        st.metric("Simulation Type", simulation_results["simulation_type"])

        # Force metrics - Row 1
        st.write("**Forces (N)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Rolling Resistance",
                f"{simulation_results['total_rolling_resistance_force']:.2f}",
            )
        with col2:
            st.metric(
                "Aerodynamic Drag",
                f"{simulation_results['total_aerodynamic_drag_force']:.2f}",
            )
        with col3:
            st.metric(
                "Hill Climb Resistance",
                f"{simulation_results['total_hill_climb_resistance_force']:.2f}",
            )

        # Force metrics - Row 2
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Linear Acceleration",
                f"{simulation_results['total_linear_acceleration_force']:.2f}",
            )
        with col2:
            st.metric(
                "Total Tractive Force",
                f"{simulation_results['total_tractive_force']:.2f}",
            )
        with col3:
            # Empty placeholder for visual balance
            st.empty()

        # Consumption metrics - Row 1
        st.write("**Energy Summary**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Energy Consumed (kWh)",
                f"{joules_to_kwh(simulation_results['total_consumption']):.2f}",
            )
        with col2:
            st.metric(
                "Total Regeneration (kWh)",
                f"{joules_to_kwh(simulation_results['total_regeneration']):.2f}",
            )
        with col3:
            st.metric(
                "Total Net Consumption (kWh)",
                f"{joules_to_kwh(simulation_results['total_net_consumption']):.2f}",
            )  # Consumption metrics - Row 2
        col1, col2, col3 = st.columns(3)
        with col1:
            if simulation_results["simulation_type"] == "Fuel":
                st.metric(
                    "Total Diesel Consumed (L)",
                    f"{joules_to_diesel_liters(simulation_results['total_consumption']):.2f}",
                )
            else:
                st.metric(
                    "Percentage Consumption (%)",
                    f"{simulation_results['percentage_consumption']:.2f}",
                )
        with col2:
            if simulation_results["simulation_type"] == "Electric":
                st.metric(
                    "Energy for 1 km (kWh)",
                    f"{joules_to_kwh(simulation_results['energy_for_1km']):.2f}",
                )
            else:  # Fuel engine
                st.metric(
                    "Diesel for 1 km (L)",
                    f"{joules_to_diesel_liters(simulation_results['energy_for_1km']):.4f}",
                )
        with col3:
            if simulation_results["simulation_type"] == "Electric":
                st.metric(
                    "Energy for 100 km (kWh)",
                    f"{joules_to_kwh(simulation_results['energy_for_100km']):.2f}",
                )
            else:  # Fuel engine
                st.metric(
                    "Diesel for 100 km (L)",
                    f"{joules_to_diesel_liters(simulation_results['energy_for_100km']):.2f}",
                )

        # Emissions metrics (for Fuel engines only)
        st.write("**Emissions (g)**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total CO",
                f"{simulation_results['total_co_emissions']:.2f}",
            )
        with col2:
            st.metric(
                "Total NOx",
                f"{simulation_results['total_nox_emissions']:.2f}",
            )
        with col3:
            st.metric(
                "Total HC",
                f"{simulation_results['total_hc_emissions']:.2f}",
            )
        with col4:
            st.metric(
                "Total PM",
                f"{simulation_results['total_pm_emissions']:.2f}",
            )

        if engine_type == "Electric":
            st.markdown("""
            **Important:** Although electric buses do not emit pollutants directly, the
            energy they use may come from sources that do generate emissions. Therefore,
            it is necessary to investigate the origin of this energy to assess its real
            environmental impact.
            """)

        st.subheader("Simulation results per segment")

        # Create a user-friendly copy of the results with converted units
        display_results = results_per_segment.copy()

        # Convert energy values from Joules to kWh for display
        energy_columns = ["consumption", "regeneration", "net_consumption"]
        for col in energy_columns:
            if col in display_results.columns:
                display_results[col] = display_results[col].apply(joules_to_kwh)

        # Rename columns for better readability
        column_mapping = {
            "consumption": "Consumption (kWh)",
            "regeneration": "Regeneration (kWh)",
            "net_consumption": "Net Consumption (kWh)",
            "co_emissions": "CO Emissions (g)",
            "nox_emissions": "NOx Emissions (g)",
            "hc_emissions": "HC Emissions (g)",
            "pm_emissions": "PM Emissions (g)",
            "rolling_resistance": "Rolling Resistance (N)",
            "aerodynamic_drag_resistance": "Aerodynamic Drag (N)",
            "hill_climb_resistance": "Hill Climb Resistance (N)",
            "linear_acceleration_force": "Linear Acceleration Force (N)",
            "tractive_force": "Tractive Force (N)",
        }

        display_results = display_results.rename(columns=column_mapping)
        st.dataframe(display_results)

        st.subheader("Simulation results map")

        # Create map-friendly data with original column names but converted energy units
        map_results = results_per_segment.copy()
        energy_columns = ["consumption", "regeneration", "net_consumption"]
        for col in energy_columns:
            if col in map_results.columns:
                map_results[col] = map_results[col].apply(joules_to_kwh)

        results_map = plot_simulation_results_map(route, map_results)
        folium_static(
            results_map,
            width=700,
            height=500,
        )

        st.altair_chart(
            plot_simulation_results(
                times=route.times,
                results=display_results,  # Use the display version with converted units
            ),
            use_container_width=True,
        )
