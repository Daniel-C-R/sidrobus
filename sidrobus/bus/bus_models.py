"""Bus models for the simulation."""

from sidrobus.unit_conversions import kwh_to_joules

ELECTRIC_BUSES = {
    "eCitaro": {
        "name": "eCitaro",
        "manufacturer": "Mercedes-Benz",
        "mass": 13_500,
        "rolling_resistance_coef": 0.005,
        "frontal_area": 8.67,
        "aerodynamic_drag_coef": 0.25,
        "engine_params": {
            "efficiency": 0.9,
            "capacity": kwh_to_joules(3 * 98),  # 3 modules of 98 kWh
            "mass": 0,
            "regenerative_braking_efficiency": 0.5,
        },
    },
    "eCitaroG": {
        "name": "eCitaro G",
        "manufacturer": "Mercedes-Benz",
        "mass": 20_000,
        "rolling_resistance_coef": 0.005,
        "aerodynamic_drag_coef": 0.25,
        "frontal_area": 8.67,
        "engine_params": {
            "efficiency": 0.35,
            "capacity": kwh_to_joules(4 * 98),  # 4 modules of 98 kWh
            "mass": 0,
            "regenerative_braking_efficiency": 0.5,
        },
    },
}

FUEL_BUSES = {
    "Citaro": {
        "name": "Citaro",
        "manufacturer": "Mercedes-Benz",
        "mass": 13_500,
        "rolling_resistance_coef": 0.005,
        "frontal_area": 8.67,
        "aerodynamic_drag_coef": 0.25,
        "engine_params": {
            "efficiency": 0.35,
            "capacity": 260 * 35860000,  # 260L of diesel
            "mass": 0,
        },
    },
    "CitaroG": {
        "name": "Citaro G",
        "manufacturer": "Mercedes-Benz",
        "mass": 20_000,
        "rolling_resistance_coef": 0.005,
        "frontal_area": 8.67,
        "aerodynamic_drag_coef": 0.25,
        "engine_params": {
            "efficiency": 0.35,
            "capacity": 300 * 35860000,  # 300L of diesel
            "mass": 0,
        },
    },
}
