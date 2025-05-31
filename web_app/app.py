"""Web UI for the simulator."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from sidrobus.bus import Bus
from sidrobus.bus.engine import ElectricEngine, FuelEngine
from sidrobus.constants import DIESEL_LHV
from sidrobus.route import Route
from sidrobus.unit_conversions import joules_to_kwh, kwh_to_joules
from web_app.map_plotting import plot_route_map

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

    route_map = plot_route_map(route)
    st_folium(
        route_map,
        key="map",
        returned_objects=[],
        width=700,
        height=500,
    )

    if run and engine_type == "Electric":
        engine = ElectricEngine(
            efficiency=energy_efficiency,
            capacity=kwh_to_joules(energy_capacity),
            mass=0,
            regenerative_braking_efficiency=regenerative_braking_efficiency,
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

        energy_consumption = bus.calculate_route_energy_consumption(route)

        st.header("Simulation Results")

        st.write("Energy Consumption (kWh):", joules_to_kwh(energy_consumption.sum()))
        st.write("Battery percentage remaining:", bus._engine.energy_ratio)  # noqa: SLF001

    if run and engine_type == "Diesel":
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

        energy_consumption = bus.calculate_route_energy_consumption(route)

        st.header("Simulation Results")

        st.write("Energy Consumption (liters):", energy_consumption.sum() / DIESEL_LHV)
        st.write(
            "Fuel percentage remaining:",
            bus._engine.energy_ratio,  # noqa: SLF001
        )
