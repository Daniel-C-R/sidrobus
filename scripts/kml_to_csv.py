"""Script to convert KML track data to CSV format.

This script reads a KML file containing GPS track data, extracts the time, coordinates,
and speed, and saves the data into a CSV file that can be easily imported into the
application.
"""

import argparse

from sidrobus.preprocessing.kml_to_dataframe import kml_to_dataframe


def main() -> None:  # noqa: D103
    # Argument parser for input/output files
    parser = argparse.ArgumentParser(
        description="Convert KML track data to CSV format."
    )
    parser.add_argument("input_kml", help="Path to the input KML file")
    parser.add_argument("output_csv", help="Path to the output CSV file")
    args = parser.parse_args()

    input_kml = args.input_kml
    output_csv = args.output_csv

    with open(input_kml, encoding="utf-8") as f:
        kml_str = f.read()
    df = kml_to_dataframe(kml_str)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


if __name__ == "__main__":
    main()
