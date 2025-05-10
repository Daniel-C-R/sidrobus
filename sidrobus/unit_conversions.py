"""Unit conversion functions."""


def kwh_to_joules(kwh: float) -> float:
    """Convert kilowatt-hours to joules.

    Parameters:
    kwh (float): Energy in kilowatt-hours.

    Returns:
    float: Energy in joules.
    """
    return kwh * 3.6e6
