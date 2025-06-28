"""Emissions Standard Module."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmissionsStandard:
    """Represents emissions standards for various pollutants per kilowatt-hour.

    This dataclass is frozen to ensure immutability of emissions standards.

    Attributes:
        co_per_kwh: Carbon monoxide (CO) emissions in grams per kilowatt-hour.
        nox_per_kwh: Nitrogen oxides (NOx) emissions in grams per kilowatt-hour.
        hc_per_kwh: Hydrocarbon (HC) emissions in grams per kilowatt-hour.
        pm_per_kwh: Particulate matter (PM) emissions in grams per kilowatt-hour.
        name: Name of the emissions standard.
    """

    co_per_kwh: float
    nox_per_kwh: float
    hc_per_kwh: float
    pm_per_kwh: float
    name: str = "Unknown"

    def __str__(self) -> str:
        """Return a string representation of the emissions standard."""
        return f"{self.name} (CO: {self.co_per_kwh}g/kWh, NOx: {self.nox_per_kwh}g/kWh)"


# Null emissions standard for cases where no emissions are applicable
# (e.g., electric buses)
NULL_EMISSIONS_STANDARD = EmissionsStandard(
    co_per_kwh=0.0,
    nox_per_kwh=0.0,
    hc_per_kwh=0.0,
    pm_per_kwh=0.0,
    name="Null Emissions Standard",
)
