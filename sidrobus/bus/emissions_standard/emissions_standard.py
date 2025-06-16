"""Emissions Standard Module."""


class EmissionsStandard:
    """Represents emissions standards for various pollutants per kilowatt-hour.

    Attributes:
        _co_per_kwh (float): Carbon monoxide (CO) emissions in grams per kilowatt-hour.
        _nox_per_kwh (float): Nitrogen oxides (NOx) emissions in grams per
            kilowatt-hour.
        _hc_per_kwh (float): Hydrocarbon (HC) emissions in grams per kilowatt-hour.
        _pm_per_kwh (float): Particulate matter (PM) emissions in grams per
            kilowatt-hour.
        _name (str): Name of the emissions standard.

        co_per_kwh (float): CO emissions per kilowatt-hour in grams.
        nox_per_kwh (float): NOx emissions per kilowatt-hour in grams.
        hc_per_kwh (float): HC emissions per kilowatt-hour in grams.
        pm_per_kwh (float): PM emissions per kilowatt-hour in grams.

    Properties:
        co_per_kwh (float): Returns CO emissions in grams per kilowatt-hour.
        nox_per_kwh (float): Returns NOx emissions in grams per kilowatt-hour.
        hc_per_kwh (float): Returns HC emissions in grams per kilowatt-hour.
        pm_per_kwh (float): Returns PM emissions in grams per kilowatt-hour.
    """

    _co_per_kwh: float
    _nox_per_kwh: float
    _hc_per_kwh: float
    _pm_per_kwh: float
    _name: str

    def __init__(
        self,
        co_per_kwh: float,
        nox_per_kwh: float,
        hc_per_kwh: float,
        pm_per_kwh: float,
        name: str = "Unknown",
    ) -> None:
        """Initializes the emissions standard.

        Args:
            co_per_kwh (float): CO emissions per kilometer in grams.
            nox_per_kwh (float): NOx emissions per kilometer in grams.
            hc_per_kwh (float): HC emissions per kilometer in grams.
            pm_per_kwh (float): PM emissions per kilometer in grams.
            name (str, optional): Name of the emissions standard. Defaults to "Unknown".

        Returns:
            None
        """
        self._co_per_kwh = co_per_kwh
        self._nox_per_kwh = nox_per_kwh
        self._hc_per_kwh = hc_per_kwh
        self._pm_per_kwh = pm_per_kwh
        self._name = name

    @property
    def co_per_kwh(self) -> float:
        """Return the CO emissions per kilometer.

        Returns:
            float: CO emissions in grams per kilometer.
        """
        return self._co_per_kwh

    @property
    def nox_per_kwh(self) -> float:
        """Return the NOx emissions per kilometer.

        Returns:
            float: NOx emissions in grams per kilometer.
        """
        return self._nox_per_kwh

    @property
    def hc_per_kwh(self) -> float:
        """Return the HC emissions per kilometer.

        Returns:
            float: HC emissions in grams per kilometer.
        """
        return self._hc_per_kwh

    @property
    def pm_per_kwh(self) -> float:
        """Return the PM emissions per kilometer.

        Returns:
            float: PM emissions in grams per kilometer.
        """
        return self._pm_per_kwh

    @property
    def name(self) -> str:
        """Return the name of the emissions standard.

        Returns:
            str: Name of the emissions standard.
        """
        return self._name


# Null emissions standard for cases where no emissions are applicable, e.g., electric
# buses
NULL_EMISSIONS_STANDARD = EmissionsStandard(
    co_per_kwh=0.0,
    nox_per_kwh=0.0,
    hc_per_kwh=0.0,
    pm_per_kwh=0.0,
    name="Null Emissions Standard",
)
