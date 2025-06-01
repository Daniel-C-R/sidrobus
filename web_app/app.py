"""Web UI for the simulator."""

import numpy as np
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

        # --- Force and consumption calculations ---
        rolling_resistance = bus.compute_route_rolling_resistance_forces()
        aerodynamic_drag = bus.compute_route_aerodynamic_drag_forces(route)
        hill_climb = bus.compute_route_hill_climb_resistance_forces(route)
        linear_acc = bus.compute_linear_acceleration_forces(route)

        rolling_resistance_energy = bus.compute_route_rolling_resistance_consumption(
            route
        )
        aerodynamic_drag_energy = bus.compute_route_aerodynamic_drag_consumption(route)
        hill_climb_energy = bus.compute_route_hill_climb_consumption(route)
        linear_acc_energy = bus.compute_linear_acceleration_consumption(route)

        regeneration = bus.compute_route_regeneration(route)
        total_consumption = bus.compute_route_consumption(route)
        net_consumption = bus.compute_route_final_consumption(route)

        # --- Show results ---
        st.header("Simulation Results")

        if engine_type == "Electric":
            st.write("Energy Consumption (kWh):", joules_to_kwh(net_consumption.sum()))
            st.write("---")
            st.write("**Detailed Energy Breakdown (kWh):**")
            st.write(
                "Rolling Resistance:", joules_to_kwh(rolling_resistance_energy.sum())
            )
            st.write("Aerodynamic Drag:", joules_to_kwh(aerodynamic_drag_energy.sum()))
            st.write("Hill Climb:", joules_to_kwh(hill_climb_energy.sum()))
            st.write("Linear Acceleration:", joules_to_kwh(linear_acc_energy.sum()))
            st.write("Regeneration:", joules_to_kwh(regeneration.sum()))
            st.write(
                "Total Consumption (no regen):", joules_to_kwh(total_consumption.sum())
            )
            st.write(
                "Net Consumption (with regen):", joules_to_kwh(net_consumption.sum())
            )
        else:
            st.write(
                "Energy Consumption (liters):", total_consumption.sum() / DIESEL_LHV
            )
            st.write("---")
            st.write("**Detailed Energy Breakdown (liters):**")
            st.write(
                "Rolling Resistance:", rolling_resistance_energy.sum() / DIESEL_LHV
            )
            st.write("Aerodynamic Drag:", aerodynamic_drag_energy.sum() / DIESEL_LHV)
            st.write("Hill Climb:", hill_climb_energy.sum() / DIESEL_LHV)
            st.write("Linear Acceleration:", linear_acc_energy.sum() / DIESEL_LHV)
            st.write("Regeneration (not available):", 0.0)
            st.write(
                "Total Consumption (no regen):", total_consumption.sum() / DIESEL_LHV
            )
            st.write(
                "Net Consumption (with regen):", net_consumption.sum() / DIESEL_LHV
            )

        # --- DataFrame with per-segment results ---

        # Ensure regeneration is zeros for Diesel (length matches route.distances)
        if engine_type == "Diesel":
            regeneration = np.zeros_like(route.distances)

        # Ensure rolling_resistance has the correct length for the number of route
        # segments
        if np.array(rolling_resistance).size == 1:
            rolling_resistance = np.full_like(
                route.distances, rolling_resistance.item(), dtype=np.float64
            )

        # Ensure all columns have the same length
        results_dict = {
            "distance_m": route.distances,
            "avg_speed_m_s": route.avg_speeds,
            "acceleration_m_s2": route.accelerations,
            "rolling_resistance_N": rolling_resistance,
            "aero_drag_N": aerodynamic_drag,
            "hill_climb_N": hill_climb,
            "linear_acc_N": linear_acc,
            "rolling_resistance_energy": rolling_resistance_energy,
            "aero_drag_energy": aerodynamic_drag_energy,
            "hill_climb_energy": hill_climb_energy,
            "linear_acc_energy": linear_acc_energy,
            "regeneration": regeneration,
            "total_consumption": total_consumption,
            "net_consumption": net_consumption,
        }

        # Normalize lengths of all arrays to match the number of route segments
        n = len(route.distances)
        for k, v in results_dict.items():
            arr = np.array(v)
            if arr.shape[0] > n:
                results_dict[k] = arr[:n]
            elif arr.shape[0] < n:
                results_dict[k] = np.pad(
                    arr, (0, n - arr.shape[0]), constant_values=np.nan
                )
            # else: already correct

        df_results = pd.DataFrame(results_dict)
        st.write("**Per-segment Results:**")
        st.dataframe(df_results)
