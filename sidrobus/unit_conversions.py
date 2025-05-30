"""Unit conversion functions."""


def kwh_to_joules(kwh: float) -> float:
    """Convert kilowatt-hours to joules.

    Parameters:
    kwh (float): Energy in kilowatt-hours.

    Returns:
    float: Energy in joules.
    """
    return kwh * 3.6e6


def joules_to_kwh(joules: float) -> float:
    """Convert joules to kilowatt-hours.

    Parameters:
    joules (float): Energy in joules.

    Returns:
    float: Energy in kilowatt-hours.
    """
    return joules / 3.6e6
