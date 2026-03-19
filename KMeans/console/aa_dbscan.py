#!/usr/bin/env python3
"""
Console tool for performing DBSCAN clustering on echosounder NetCDF files.

Accepts an Sv dataset (or other variable), clusters pixels across frequencies
using either the direct or absolute-differences model, and writes a new
NetCDF containing integer cluster labels in the same spatial grid as the
original echogram.  Unlike KMeans, DBSCAN discovers the number of clusters
automatically and labels noise points as -1.

Follows the aa-* console-tool architecture:
    - Accepts a file path from STDIN or as a positional argument
    - Performs a single, well-defined operation
    - Prints the output file path to STDOUT for piping
"""

import argparse
import io
import pprint
import sys
from contextlib import redirect_stdout
from pathlib import Path

import xarray as xr
from loguru import logger

from KMeans.dbscan_core import cluster_dataset
from KMeans.kmeans_core import list_channels


def print_help():
    help_text = """
    Usage: aa-dbscan [OPTIONS] [INPUT_PATH]

    Arguments:
    INPUT_PATH                  Path to a NetCDF file containing Sv data.
                                Optional. Defaults to stdin if not provided.

    Options:
    -o, --output_path           Path to save the cluster-map NetCDF.
                                Default: <stem>_dbscan.nc

    --model                     Clustering model to use.
                                Choices: abd, dir
                                  abd = absolute differences (default)
                                        Pairwise |Sv(A)-Sv(B)| across channels.
                                        Identical channels produce a blank result,
                                        so 100%% of the information is meaningful.
                                  dir = direct
                                        Raw Sv values across channels.
                                Default: abd

    --eps                       Maximum distance between two samples for them
                                to be considered neighbours.
                                Default: 0.5

    --min_samples               Minimum number of points required to form a
                                dense region (core point threshold).
                                Default: 5

    --metric                    Distance metric for neighbourhood computation.
                                Default: euclidean

    --channels                  Space-separated 0-based channel indices to use.
                                Default: all channels in the dataset.
                                Example: --channels 0 1 2

    --var                       Data variable to cluster on.
                                Default: Sv

    --n_jobs                    Number of parallel jobs for distance computation.
                                -1 uses all available cores.
                                Default: None (single-threaded)

    --list_channels             List available channels and exit.

    --quiet                     Suppress logger info, only print output path.

    Description:
    Performs DBSCAN clustering on multi-frequency echosounder data.
    The output NetCDF has the same spatial dimensions (ping_time ×
    range_sample) as the input echogram, but contains integer cluster
    labels instead of Sv values.  This "cluster map" can be visualised
    in the same way an echogram is plotted.

    Unlike KMeans, DBSCAN does not require the number of clusters to
    be specified.  It discovers clusters based on density and labels
    points that do not belong to any cluster as noise (-1).

    Two feature-matrix models are available:

      abd (absolute differences) — default
        For each pair of selected channels, compute |Sv_A - Sv_B|.
        These pairwise differences form the feature matrix.  If two
        identical channels are selected, the result is blank — meaning
        all visual information is meaningful.

      dir (direct)
        Each pixel is a vector of raw Sv values across selected
        channels.  Straightforward, but allows identical frequencies
        to contribute redundant information.

    Examples:
      echo file.nc | aa-dbscan --model abd --eps 1.0 --min_samples 10
      echo file.nc | aa-dbscan --model dir --eps 0.5 --channels 0 1 3
      aa-dbscan /path/to/input_Sv.nc --eps 2.0 --min_samples 20 -o clustered.nc
    """
    print(help_text)


def main():

    if len(sys.argv) == 1:
        if not sys.stdin.isatty():
            stdin_data = sys.stdin.readline().strip()
            if stdin_data:
                sys.argv.append(stdin_data)
        else:
            print_help()
            sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Perform DBSCAN clustering on multi-frequency echosounder Sv data."
    )

    # ---------------------------
    # Required file arguments
    # ---------------------------
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the NetCDF file containing Sv data.",
        nargs="?",
    )

    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        help="Path to save cluster-map NetCDF (default: <stem>_dbscan.nc).",
    )

    # ---------------------------
    # Clustering model
    # ---------------------------
    parser.add_argument(
        "--model",
        type=str,
        choices=["abd", "dir"],
        default="abd",
        help="Feature-matrix model: abd (absolute differences, default) or dir (direct).",
    )

    # ---------------------------
    # DBSCAN parameters
    # ---------------------------
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Maximum neighbourhood radius (default: 0.5).",
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="Minimum points to form a dense region (default: 5).",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric (default: euclidean).",
    )

    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="0-based channel indices to use (default: all). Example: --channels 0 1 2",
    )

    parser.add_argument(
        "--var",
        type=str,
        default="Sv",
        help="Data variable to cluster on (default: Sv).",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Parallel jobs for distance computation. -1 = all cores (default: None).",
    )

    # ---------------------------
    # Utility flags
    # ---------------------------
    parser.add_argument(
        "--list_channels",
        action="store_true",
        help="List available channels in the input file and exit.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logging; only print output path.",
    )

    args = parser.parse_args()

    # ---------------------------
    # Validate input
    # ---------------------------
    if args.input_path is None:
        args.input_path = Path(sys.stdin.readline().strip())
        if not args.quiet:
            logger.info(f"Read input path from stdin: {args.input_path}")

    if not args.input_path.exists():
        logger.error(f"File '{args.input_path}' does not exist.")
        sys.exit(1)

    allowed_extensions = {".netcdf4": "netcdf", ".nc": "netcdf"}
    ext = args.input_path.suffix.lower()
    if ext not in allowed_extensions:
        logger.error(
            f"'{args.input_path.name}' is not a supported file type. "
            f"Allowed: {', '.join(allowed_extensions.keys())}"
        )
        sys.exit(1)

    # ---------------------------
    # Load dataset (suppress echopype chatter)
    # ---------------------------
    f = io.StringIO()
    with redirect_stdout(f):
        ds = xr.open_dataset(args.input_path)

    # ---------------------------
    # --list_channels: inspect and exit
    # ---------------------------
    if args.list_channels:
        try:
            chans = list_channels(ds, var=args.var)
            for i, c in enumerate(chans):
                print(f"  [{i}] {c}")
        except Exception as e:
            logger.error(f"Could not list channels: {e}")
            sys.exit(1)
        sys.exit(0)

    # ---------------------------
    # Set default output path
    # ---------------------------
    if args.output_path is None:
        args.output_path = args.input_path.with_stem(
            args.input_path.stem + "_dbscan"
        ).with_suffix(".nc")

    # ---------------------------
    # Perform clustering
    # ---------------------------
    try:
        if not args.quiet:
            args_dict = vars(args)
            pretty_args = pprint.pformat(args_dict)
            logger.debug(
                f"Executing aa-dbscan configured with [OPTIONS]:\n{pretty_args}\n"
                f"* ( Each aa-dbscan associated option_name may be "
                f"overridden using --option_name value )"
            )

        cluster_ds = cluster_dataset(
            ds,
            model=args.model,
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
            channels=args.channels,
            var=args.var,
            n_jobs=args.n_jobs,
        )

        # ---------------------------
        # Save output
        # ---------------------------
        logger.info(f"Saving cluster map to {args.output_path} ...")
        cluster_ds.to_netcdf(args.output_path, mode="w", format="NETCDF4")

        if not args.quiet:
            logger.info(
                f"Generating {args.output_path.resolve()} with aa-dbscan. "
                f"Passing nc path to stdout..."
            )

        # Print output path to stdout for piping
        print(args.output_path.resolve())

    except Exception as e:
        logger.exception(f"Error during DBSCAN clustering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()