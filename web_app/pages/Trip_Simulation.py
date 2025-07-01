"""Web UI for the simulator."""

from typing import Any

import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from sidrobus.bus import Bus
from sidrobus.bus.bus_models import BusFactory
from sidrobus.bus.emissions_standard import EmissionsStandard
from sidrobus.bus.emissions_standard.euro_standards import EURO_VI
from sidrobus.bus.engine import ElectricEngine, FuelEngine
from sidrobus.constants import DIESEL_LHV
from sidrobus.route import Route
from sidrobus.unit_conversions import joules_to_kwh, kwh_to_joules
from web_app.interactive_plotting import plot_route_data, plot_simulation_results
from web_app.map_plotting import plot_route_map, plot_simulation_results_map


def joules_to_diesel_liters(joules: float) -> float:
    """Convert joules to diesel liters."""
    return joules / DIESEL_LHV


def snake_to_title(snake_str: str) -> str:
    """Convert snake_case string to Title Case."""
    return snake_str.replace("_", " ").title()


def create_bus_from_config(
    config_mode: str,
    **kwargs,  # type: ignore[misc]  # noqa: ANN003
) -> tuple[Bus, dict[str, Any]]:
    """Create a bus based on configuration mode and parameters.

    Args:
        config_mode: Either "Use Predefined Model" or "Manual Configuration"
        **kwargs: Configuration parameters specific to the chosen mode

    Returns:
        Tuple of (bus instance, configuration info dict)
    """
    if config_mode == "Use Predefined Model":
        # Extract predefined model parameters
        selected_model_id: str = kwargs["selected_model_id"]
        selected_capacity_id: str = kwargs["selected_capacity_id"]
        custom_energy: float | None = kwargs.get("custom_energy")
        custom_mass: float | None = kwargs.get("custom_mass")

        bus = BusFactory.create_bus(
            model_id=selected_model_id,
            capacity_option_id=selected_capacity_id,
            energy=custom_energy,
            mass=custom_mass,
        )

        selected_model = BusFactory.get_model(selected_model_id)
        config_info = {
            "type": "Predefined Model",
            "model_name": bus.model_name,
            "manufacturer": bus.model_manufacturer,
            "engine_type": selected_model.engine_type,
            "mass": bus.mass,
            "capacity_kwh": joules_to_kwh(bus.engine.capacity),
            "initial_energy_kwh": joules_to_kwh(bus.engine.energy),
            "energy_ratio": bus.engine.energy_ratio,
            "efficiency": bus.engine.efficiency,
        }

    else:  # Manual Configuration
        # Extract manual parameters
        mass: float = kwargs["mass"]
        frontal_area: float = kwargs["frontal_area"]
        aerodynamic_drag_coef: float = kwargs["aerodynamic_drag_coef"]
        rolling_resistance_coef: float = kwargs["rolling_resistance_coef"]
        energy_efficiency: float = kwargs["energy_efficiency"]
        engine_type: str = kwargs["engine_type"]
        energy_capacity: float = kwargs["energy_capacity"]

        if engine_type == "Electric":
            regenerative_braking_efficiency = kwargs["regenerative_braking_efficiency"]
            engine = ElectricEngine(
                efficiency=energy_efficiency,
                capacity=kwh_to_joules(energy_capacity),
                mass=0,
                regenerative_braking_efficiency=regenerative_braking_efficiency,
            )
        else:  # Diesel
            co_per_kwh = kwargs["co_per_kwh"]
            nox_per_kwh = kwargs["nox_per_kwh"]
            hc_per_kwh = kwargs["hc_per_kwh"]
            pm_per_kwh = kwargs["pm_per_kwh"]

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
            engine,
            frontal_area,
            mass,
            aerodynamic_drag_coef,
            rolling_resistance_coef,
            model_name="Custom Bus",
            model_manufacturer="Manual Configuration",
        )

        config_info = {
            "type": "Manual Configuration",
            "model_name": bus.model_name,
            "manufacturer": bus.model_manufacturer,
            "engine_type": engine_type,
            "mass": bus.mass,
            "capacity_kwh": joules_to_kwh(bus.engine.capacity),
            "initial_energy_kwh": joules_to_kwh(bus.engine.energy),
            "energy_ratio": bus.engine.energy_ratio,
            "efficiency": bus.engine.efficiency,
        }

    return bus, config_info


def render_sidebar() -> tuple[Any, bool]:
    """Render the sidebar with route loader and bus configuration."""
    with st.sidebar:
        st.header("Route Loader")
        route_file = st.file_uploader("Upload Route", type=["csv"])

        st.header("Bus Configuration")

        # Bus configuration mode selection
        config_mode = st.radio(
            "Configuration Mode",
            options=["Use Predefined Model", "Manual Configuration"],
            index=0,
            help="Choose between using predefined bus models or "
            "configuring parameters manually",
        )

        if config_mode == "Use Predefined Model":
            render_predefined_model_config()
        else:
            render_manual_config()

        run = st.button(
            "Run Simulation",
            disabled=route_file is None,
        )

    return route_file, run


def render_predefined_model_config() -> None:
    """Render the predefined model configuration section."""
    # Predefined model configuration
    available_models = BusFactory.get_available_models()

    if not available_models:
        st.error("No bus models available. Please check the bus models configuration.")
        st.stop()

    # Model selection
    model_options = list(available_models.keys())
    model_display_names = [
        f"{available_models[model_id].name} ({model_id})" for model_id in model_options
    ]

    selected_model_index = st.selectbox(
        "Select Bus Model",
        options=range(len(model_options)),
        format_func=lambda x: model_display_names[x],
        help="Choose from available predefined bus models",
    )

    selected_model_id = model_options[selected_model_index]
    selected_model = available_models[selected_model_id]

    # Capacity option selection
    capacity_options = selected_model.capacity_options
    capacity_option_ids = list(capacity_options.keys())

    capacity_display_names = []
    for option_id in capacity_option_ids:
        if selected_model.engine_type == "Fuel":
            # For fuel engines, show liters instead of kWh
            capacity_liters = capacity_options[option_id].capacity_kwh / DIESEL_LHV
            capacity_display_names.append(f"{option_id}: {capacity_liters:.0f} L")
        else:
            # For electric engines, show kWh
            capacity_kwh = joules_to_kwh(capacity_options[option_id].capacity_kwh)
            capacity_display_names.append(f"{option_id}: {capacity_kwh:.0f} kWh")

    selected_capacity_index = st.selectbox(
        "Select Capacity Option",
        options=range(len(capacity_option_ids)),
        format_func=lambda x: capacity_display_names[x],
        help="Choose the capacity configuration for the selected model",
    )

    selected_capacity_id = capacity_option_ids[selected_capacity_index]

    # Optional customizations
    st.subheader("Optional Customizations")

    # Mass override
    use_custom_mass = st.checkbox(
        "Override Mass",
        value=False,
        help="Check to specify a custom mass (e.g., for different passenger loads)",
    )

    custom_mass = None
    if use_custom_mass:
        custom_mass = st.number_input(
            "Custom Mass (kg)",
            min_value=1000,
            value=int(selected_model.mass),
            step=100,
            help="Specify custom mass to account for passengers, luggage, etc.",
        )

    # Energy level override
    use_custom_energy = st.checkbox(
        "Override Initial Energy Level",
        value=False,
        help="Check to specify a custom initial energy level",
    )

    custom_energy = None
    custom_energy_kwh = None
    energy_percentage = 100

    if use_custom_energy:
        max_capacity_kwh = joules_to_kwh(
            capacity_options[selected_capacity_id].capacity_kwh
        )
        energy_percentage = st.slider(
            "Initial Energy Level (%)",
            min_value=10,
            max_value=100,
            value=100,
            step=5,
            help="Set the initial energy level as percentage of capacity",
        )
        custom_energy_kwh = max_capacity_kwh * (energy_percentage / 100)
        custom_energy = kwh_to_joules(custom_energy_kwh)

        st.write(f"Initial Energy: {custom_energy_kwh:.1f} kWh")

    # Store configuration in session state
    st.session_state.bus_config = {
        "mode": "predefined",
        "selected_model_id": selected_model_id,
        "selected_capacity_id": selected_capacity_id,
        "custom_energy": custom_energy,
        "custom_mass": custom_mass,
        "use_custom_energy": use_custom_energy,
        "use_custom_mass": use_custom_mass,
        "energy_percentage": energy_percentage,
        "custom_energy_kwh": custom_energy_kwh,
    }


def render_manual_config() -> None:
    """Render the manual configuration section."""
    st.subheader("Bus Parameters")

    mass = st.number_input("Bus Mass (kg)", min_value=1000, value=12000)
    rolling_resistance_coef = st.number_input(
        "Rolling Resistance Coefficient [0, 1]",
        min_value=0.0,
        value=0.01,
        step=0.001,
    )
    frontal_area = st.number_input(
        "Frontal Area (m²)", min_value=0.0, value=5.0, step=0.01
    )
    aerodynamic_drag_coef = st.number_input(
        "Aerodynamic Drag Coefficient [0, 1]", min_value=0.0, value=0.6, step=0.01
    )
    energy_efficiency = st.number_input(
        "Energy efficiency [0, 1]",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.01,
    )

    engine_type = st.selectbox(
        "Engine Type",
        options=["Electric", "Diesel"],
        index=0,
    )

    # Initialize variables
    energy_capacity = 0.0
    regenerative_braking_efficiency = 0.0
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
            st.number_input("Fuel Tank Capacity (L)", min_value=1.0, value=100.0)
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

    # Store configuration in session state
    st.session_state.bus_config = {
        "mode": "manual",
        "mass": mass,
        "frontal_area": frontal_area,
        "aerodynamic_drag_coef": aerodynamic_drag_coef,
        "rolling_resistance_coef": rolling_resistance_coef,
        "energy_efficiency": energy_efficiency,
        "engine_type": engine_type,
        "energy_capacity": energy_capacity,
        "regenerative_braking_efficiency": regenerative_braking_efficiency
        if engine_type == "Electric"
        else None,
        "co_per_kwh": co_per_kwh if engine_type == "Diesel" else None,
        "nox_per_kwh": nox_per_kwh if engine_type == "Diesel" else None,
        "hc_per_kwh": hc_per_kwh if engine_type == "Diesel" else None,
        "pm_per_kwh": pm_per_kwh if engine_type == "Diesel" else None,
    }


def render_bus_details() -> None:
    """Render the bus details section."""
    if not (hasattr(st.session_state, "bus_config") and st.session_state.bus_config):
        return

    st.header("Bus Details")
    config = st.session_state.bus_config

    if config["mode"] == "predefined":
        render_predefined_model_details(config)
    else:
        render_manual_config_details(config)


def render_predefined_model_details(config: dict[str, Any]) -> None:
    """Render details for predefined model configuration."""
    available_models = BusFactory.get_available_models()
    selected_model = available_models[config["selected_model_id"]]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Information")
        st.write(f"**Name:** {selected_model.name}")
        st.write(f"**Manufacturer:** {selected_model.manufacturer}")
        st.write(f"**Engine Type:** {selected_model.engine_type}")
        st.write(
            f"**Default Mass:** {selected_model.mass:,} kg "
            f"({selected_model.mass / 1000:.1f} tons)"
        )
        if config.get("use_custom_mass", False) and config.get("custom_mass"):
            st.write(
                f"**Custom Mass:** {config['custom_mass']:,} kg "
                f"({config['custom_mass'] / 1000:.1f} tons)"
            )

    with col2:
        st.subheader("Technical Specifications")
        st.write(f"**Frontal Area:** {selected_model.frontal_area} m²")
        st.write(
            f"**Aerodynamic Drag Coefficient:** {selected_model.aerodynamic_drag_coef}"
        )
        st.write(
            f"**Rolling Resistance Coefficient:** "
            f"{selected_model.rolling_resistance_coef}"
        )
        st.write(f"**Efficiency:** {selected_model.engine_options.efficiency:.1%}")

        # Show capacity information
        capacity_info = selected_model.capacity_options[config["selected_capacity_id"]]
        if selected_model.engine_type == "Fuel":
            capacity_liters = capacity_info.capacity_kwh / DIESEL_LHV
            st.write(f"**Tank Capacity:** {capacity_liters:.0f} L")
            if config.get("use_custom_energy", False) and config.get("custom_energy"):
                initial_liters = joules_to_diesel_liters(config["custom_energy"])
                energy_pct = config.get("energy_percentage", 100)
                st.write(f"**Initial Fuel:** {initial_liters:.1f} L ({energy_pct}%)")
            else:
                st.write(f"**Initial Fuel:** {capacity_liters:.0f} L (100%)")
        else:
            capacity_kwh = joules_to_kwh(capacity_info.capacity_kwh)
            st.write(f"**Battery Capacity:** {capacity_kwh:.0f} kWh")
            use_custom_energy = config.get("use_custom_energy", False)
            custom_energy_kwh = config.get("custom_energy_kwh")
            if use_custom_energy and custom_energy_kwh:
                energy_pct = config.get("energy_percentage", 100)
                st.write(
                    f"**Initial Charge:** {config['custom_energy_kwh']:.1f} kWh "
                    f"({energy_pct}%)"
                )
            else:
                st.write(f"**Initial Charge:** {capacity_kwh:.0f} kWh (100%)")


def render_manual_config_details(config: dict[str, Any]) -> None:
    """Render details for manual configuration."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bus Configuration")
        st.write("**Configuration Type:** Manual")
        st.write(f"**Mass:** {config['mass']:,} kg ({config['mass'] / 1000:.1f} tons)")
        st.write(f"**Frontal Area:** {config['frontal_area']} m²")
        st.write(f"**Engine Type:** {config['engine_type']}")

    with col2:
        st.subheader("Technical Parameters")
        st.write(f"**Aerodynamic Drag Coefficient:** {config['aerodynamic_drag_coef']}")
        st.write(
            f"**Rolling Resistance Coefficient:** {config['rolling_resistance_coef']}"
        )
        st.write(f"**Energy Efficiency:** {config['energy_efficiency']:.1%}")

        if config["engine_type"] == "Electric":
            st.write(f"**Battery Capacity:** {config['energy_capacity']:.0f} kWh")
            if config.get("regenerative_braking_efficiency"):
                st.write(
                    f"**Regenerative Braking Efficiency:** "
                    f"{config['regenerative_braking_efficiency']:.1%}"
                )
        else:  # Diesel
            fuel_capacity_liters = config["energy_capacity"] / DIESEL_LHV
            st.write(f"**Tank Capacity:** {fuel_capacity_liters:.0f} L")


def render_route_details(route_file: object) -> Route | None:
    """Render the route details section."""
    if route_file is None:
        return None

    st.header("Route details")
    route_data = pd.read_csv(route_file)  # type: ignore
    route = Route.from_dataframe(route_data)

    with st.expander("Route Data"):
        st.dataframe(route_data)

    st.subheader("Plots")

    route_map = plot_route_map(route)
    folium_static(route_map, width=700, height=500)

    route_plot = plot_route_data(route)
    st.altair_chart(route_plot, use_container_width=True)

    render_route_summary(route.summary)

    return route


def render_route_summary(route_summary: dict[str, Any]) -> None:
    """Render the route summary metrics."""
    st.subheader("Route Summary")

    # Row 1: Basic route info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Distance (km)", f"{route_summary['total_distance'] / 1000:.2f}"
        )
    with col2:
        st.metric("Number of Points", route_summary["number_of_points"])
    with col3:
        st.metric("Duration (min)", f"{route_summary['duration'] / 60:.2f}")

    # Row 2: Time info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Start Time (s)", route_summary["start_time"])
    with col2:
        st.metric("End Time (s)", route_summary["end_time"])
    with col3:
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


def run_simulation(route: Route) -> None:
    """Run the bus simulation and display results."""
    try:
        if (
            not hasattr(st.session_state, "bus_config")
            or not st.session_state.bus_config
        ):
            st.error("No bus configuration found. Please configure the bus parameters.")
            st.stop()

        config = st.session_state.bus_config

        if config["mode"] == "predefined":
            bus, config_info = create_bus_from_config(
                "Use Predefined Model",
                selected_model_id=config["selected_model_id"],
                selected_capacity_id=config["selected_capacity_id"],
                custom_energy=config.get("custom_energy"),
                custom_mass=config.get("custom_mass"),
            )
        else:  # Manual Configuration
            bus, config_info = create_bus_from_config(
                "Manual Configuration",
                mass=config["mass"],
                frontal_area=config["frontal_area"],
                aerodynamic_drag_coef=config["aerodynamic_drag_coef"],
                rolling_resistance_coef=config["rolling_resistance_coef"],
                energy_efficiency=config["energy_efficiency"],
                engine_type=config["engine_type"],
                energy_capacity=config["energy_capacity"],
                regenerative_braking_efficiency=config.get(
                    "regenerative_braking_efficiency"
                ),
                co_per_kwh=config.get("co_per_kwh"),
                nox_per_kwh=config.get("nox_per_kwh"),
                hc_per_kwh=config.get("hc_per_kwh"),
                pm_per_kwh=config.get("pm_per_kwh"),
            )

    except Exception as e:
        st.error(f"Error creating bus: {e}")
        st.stop()

    st.header("Simulation Results")
    simulation_results = bus.simulate_trip(route, modify_bus=False)

    render_simulation_summary(simulation_results)
    render_simulation_results_table(simulation_results)
    render_simulation_map_and_charts(simulation_results, route)


def render_simulation_summary(simulation_results: dict[str, Any]) -> None:
    """Render the simulation summary metrics."""
    st.subheader("Simulation summary")
    st.metric("Simulation Type", simulation_results["simulation_type"])

    _render_force_metrics(simulation_results)
    _render_energy_metrics(simulation_results)
    _render_consumption_metrics(simulation_results)
    _render_emissions_metrics(simulation_results)

    # Note for electric buses
    if simulation_results["simulation_type"] == "Electric":
        st.markdown("""
        **Important:** Although electric buses do not emit pollutants directly, the
        energy they use may come from sources that do generate emissions. Therefore,
        it is necessary to investigate the origin of this energy to assess its real
        environmental impact.
        """)


def _render_force_metrics(simulation_results: dict[str, Any]) -> None:
    """Render force metrics."""
    st.write("**Forces (N)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Rolling Resistance",
            f"{simulation_results['total_rolling_resistance_force']:.2e}",
        )
    with col2:
        st.metric(
            "Aerodynamic Drag",
            f"{simulation_results['total_aerodynamic_drag_force']:.2e}",
        )
    with col3:
        st.metric(
            "Hill Climb Resistance",
            f"{simulation_results['total_hill_climb_resistance_force']:.2e}",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Linear Acceleration",
            f"{simulation_results['total_linear_acceleration_force']:.2e}",
        )
    with col2:
        st.metric(
            "Total Tractive Force",
            f"{simulation_results['total_tractive_force']:.2e}",
        )
    with col3:
        st.empty()


def _render_energy_metrics(simulation_results: dict[str, Any]) -> None:
    """Render energy metrics."""
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
        )


def _render_consumption_metrics(simulation_results: dict[str, Any]) -> None:
    """Render consumption metrics."""
    # Consumption metrics by type
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
        if simulation_results["simulation_type"] == "Fuel":
            st.metric(
                "Percentage Consumption (%)",
                f"{simulation_results['percentage_consumption']:.2f}",
            )
        else:
            st.metric(
                "Energy for 1 km (kWh)",
                f"{joules_to_kwh(simulation_results['energy_for_1km']):.2f}",
            )
    with col3:
        if simulation_results["simulation_type"] == "Fuel":
            st.metric(
                "Diesel for 1 km (L)",
                f"{joules_to_diesel_liters(simulation_results['energy_for_1km']):.4f}",
            )
        else:
            st.metric(
                "Energy for 100 km (kWh)",
                f"{joules_to_kwh(simulation_results['energy_for_100km']):.2f}",
            )

    # Additional consumption metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if simulation_results["simulation_type"] == "Fuel":
            st.metric(
                "Diesel for 100 km (L)",
                f"{joules_to_diesel_liters(simulation_results['energy_for_100km']):.2f}",
            )
        else:
            st.empty()
    with col2:
        st.empty()
    with col3:
        st.empty()


def _render_emissions_metrics(simulation_results: dict[str, Any]) -> None:
    """Render emissions metrics."""
    st.write("**Emissions (g)**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CO", f"{simulation_results['total_co_emissions']:.2f}")
    with col2:
        st.metric("Total NOx", f"{simulation_results['total_nox_emissions']:.2f}")
    with col3:
        st.metric("Total HC", f"{simulation_results['total_hc_emissions']:.2f}")
    with col4:
        st.metric("Total PM", f"{simulation_results['total_pm_emissions']:.2f}")


def render_simulation_results_table(simulation_results: dict[str, Any]) -> None:
    """Render the simulation results table."""
    st.subheader("Simulation results per segment")

    results_per_segment = pd.DataFrame(simulation_results["results_per_segment"])
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
        "rolling_resistance": "Rolling Resistance (N)",
        "aerodynamic_drag_resistance": "Aerodynamic Drag (N)",
        "hill_climb_resistance": "Hill Climb Resistance (N)",
        "linear_acceleration_force": "Linear Acceleration Force (N)",
        "tractive_force": "Tractive Force (N)",
    }

    # Add emissions columns only for fuel engines
    if simulation_results["simulation_type"] == "Fuel":
        emissions_mapping = {
            "co_emissions": "CO Emissions (g)",
            "nox_emissions": "NOx Emissions (g)",
            "hc_emissions": "HC Emissions (g)",
            "pm_emissions": "PM Emissions (g)",
        }
        column_mapping.update(emissions_mapping)
    else:
        # Remove emissions columns for electric engines
        emissions_columns = [
            "co_emissions",
            "nox_emissions",
            "hc_emissions",
            "pm_emissions",
        ]
        for col in emissions_columns:
            if col in display_results.columns:
                display_results = display_results.drop(columns=[col])

    display_results = display_results.rename(columns=column_mapping)
    st.dataframe(display_results)


def render_simulation_map_and_charts(
    simulation_results: dict[str, Any], route: Route
) -> None:
    """Render the simulation results map and charts."""
    st.subheader("Simulation results map")

    results_per_segment = pd.DataFrame(simulation_results["results_per_segment"])
    map_results = results_per_segment.copy()

    # Convert energy values for map display
    energy_columns = ["consumption", "regeneration", "net_consumption"]
    for col in energy_columns:
        if col in map_results.columns:
            map_results[col] = map_results[col].apply(joules_to_kwh)

    # Remove emissions columns for electric engines
    if simulation_results["simulation_type"] == "Electric":
        emissions_columns = [
            "co_emissions",
            "nox_emissions",
            "hc_emissions",
            "pm_emissions",
        ]
        for col in emissions_columns:
            if col in map_results.columns:
                map_results = map_results.drop(columns=[col])

    results_map = plot_simulation_results_map(route, map_results)
    folium_static(results_map, width=700, height=500)

    # Display charts
    display_results = results_per_segment.copy()
    energy_columns = ["consumption", "regeneration", "net_consumption"]
    for col in energy_columns:
        if col in display_results.columns:
            display_results[col] = display_results[col].apply(joules_to_kwh)

    st.altair_chart(
        plot_simulation_results(times=route.times, results=display_results),
        use_container_width=True,
    )


def main() -> None:
    """Main application function."""
    st.title("Bus Simulator")

    # Render sidebar and get inputs
    route_file, run = render_sidebar()

    # Render bus details (always visible when configured)
    render_bus_details()

    # Render route details (when route is uploaded)
    route = render_route_details(route_file)

    # Run simulation (when button is clicked and route is available)
    if run and route is not None:
        run_simulation(route)


if __name__ == "__main__":
    main()
