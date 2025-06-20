"""Converts KML GPS track data to a pandas DataFrame."""

import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd


def kml_to_dataframe(kml_str: str) -> pd.DataFrame:
    """Converts a KML string containing GPS track data into a pandas DataFrame.

    Extracts times, coordinates (longitude, latitude, altitude), and speed from a KML
    track, and returns them in a DataFrame with the columns:
    ["time", "longitude", "latitude", "altitude", "speed"]

    Args:
        kml_str (str): Content of the KML file as a string.

    Returns:
        pd.DataFrame: DataFrame with the mentioned columns, where 'time' is the relative
            time in seconds from the first point.
    """
    ns = {
        "gx": "http://www.google.com/kml/ext/2.2",
        "kml": "http://www.opengis.net/kml/2.2",
    }
    root = ET.fromstring(kml_str)  # noqa: S314
    track = root.find(".//gx:Track", ns)
    whens = [w.text for w in track.findall("kml:when", ns)]  # type: ignore
    coords = [c.text for c in track.findall("gx:coord", ns)]  # type: ignore

    speed_data = track.find('.//gx:SimpleArrayData[@name="speed"]', ns)  # type: ignore
    speeds = (
        [float(s.text) for s in speed_data.findall("gx:value", ns)]  # type: ignore
        if speed_data is not None
        else [None] * len(whens)
    )

    base_time = datetime.fromisoformat(whens[0].replace("Z", "+00:00"))  # type: ignore
    times_sec = [
        (datetime.fromisoformat(w.replace("Z", "+00:00")) - base_time).total_seconds()  # type: ignore
        for w in whens
    ]

    data = []
    for t, coord, speed in zip(times_sec, coords, speeds, strict=False):
        lon, lat, alt = map(float, coord.split())  # type: ignore
        data.append([t, lon, lat, alt, speed])

    return pd.DataFrame(
        data, columns=["time", "longitude", "latitude", "altitude", "speed"]
    )
