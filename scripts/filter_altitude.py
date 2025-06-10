"""Filter altitude data in a CSV file using Gaussian or Savitzky-Golay filter."""

import argparse

import pandas as pd

from sidrobus.preprocessing.filter_altitude import gaussian_filter, savgol_filter_route


def main() -> None:  # noqa: D103
    # Argument parser for input/output files
    parser = argparse.ArgumentParser(description="Filter altitude data in a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument(
        "--filter-type",
        choices=["gaussian", "savgol"],
        default="savgol",
        help="Type of filter to apply to the altitude data (default: savgol)",
    )
    parser.add_argument("output_csv", help="Path to the output CSV file")

    # New arguments for filter parameters
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Sigma for Gaussian filter (default: 2.0)",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="""
        Window length for Savitzky-Golay filter (default: to 1/10 of data length)
        """,
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay filter (default: 2)",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    filter_type = args.filter_type

    df = pd.read_csv(input_csv)
    if args.window_length <= 0:
        args.window_length = max(11, df.shape[0] // 10)

    if filter_type == "gaussian":
        filtered_df = gaussian_filter(df, sigma=args.sigma)
    elif filter_type == "savgol":
        filtered_df = savgol_filter_route(
            df,
            window_length=args.window_length,
            polyorder=args.polyorder,
        )
    else:
        error_msg = f"Unknown filter type: {filter_type}"
        raise ValueError(error_msg)

    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered CSV file saved to {output_csv}")


if __name__ == "__main__":
    main()
