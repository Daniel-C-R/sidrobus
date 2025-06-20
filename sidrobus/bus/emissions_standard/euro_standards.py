"""This module defines the Euro emissions standards for buses."""

from sidrobus.bus.emissions_standard import EmissionsStandard

EURO_IV = EmissionsStandard(
    co_per_kwh=1.5,
    nox_per_kwh=0.46,
    hc_per_kwh=3.5,
    pm_per_kwh=0.02,
    name="EURO IV",
)

EURO_V = EmissionsStandard(
    co_per_kwh=1.5,
    nox_per_kwh=0.46,
    hc_per_kwh=2,
    pm_per_kwh=0.02,
    name="EURO V",
)

EURO_VI = EmissionsStandard(
    co_per_kwh=0.13,
    nox_per_kwh=0.46,
    hc_per_kwh=0.4,
    pm_per_kwh=0.01,
    name="EURO VI",
)
