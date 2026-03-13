#!/usr/bin/env python3
"""
Console tool for performing KMeans clustering on echosounder NetCDF files.

Accepts an Sv dataset (or other variable), clusters pixels across frequencies
using either the direct or absolute-differences model, and writes a new
NetCDF containing integer cluster labels in the same spatial grid as the
original echogram.

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

from KMeans.kmeans_core import cluster_dataset, list_channels

def print_help():
    help_text = """
    Usage: aa-kmeans [OPTIONS] [INPUT_PATH]

    Arguments:
    INPUT_PATH                  Path to a NetCDF file containing Sv data.
                                Optional. Defaults to stdin if not provided.

    Options:
    -o, --output_path           Path to save the cluster-map NetCDF.
                                Default: <stem>_kmeans.nc

    --model                     Clustering model to use.
                                Choices: abd, dir
                                  abd = absolute differences (default)
                                        Pairwise |Sv(A)-Sv(B)| across channels.
                                        Identical channels produce a blank result,
                                        so 100%% of the information is meaningful.
                                  dir = direct
                                        Raw Sv values across channels.
                                Default: abd

    -k, --n_clusters            Number of KMeans clusters.
                                Default: 3

    --channels                  Space-separated 0-based channel indices to use.
                                Default: all channels in the dataset.
                                Example: --channels 0 1 2

    --var                       Data variable to cluster on.
                                Default: Sv

    --n_init                    Number of KMeans initialisations.
                                Default: 10

    --max_iter                  Maximum iterations per KMeans run.
                                Default: 300

    --random_state              Random seed for reproducibility.
                                Default: None

    --list_channels             List available channels and exit.

    --quiet                     Suppress logger info, only print output path.

    Description:
    Performs KMeans clustering on multi-frequency echosounder data.
    The output NetCDF has the same spatial dimensions (ping_time ×
    range_sample) as the input echogram, but contains integer cluster
    labels instead of Sv values.  This "cluster map" can be visualised
    in the same way an echogram is plotted.

    Two models are available:

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
      echo file.nc | aa-kmeans --model abd
      echo file.nc | aa-kmeans --model dir -k 5 --channels 0 1 3
      aa-kmeans /path/to/input_Sv.nc --model abd -k 4 -o clustered.nc
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
        description="Perform KMeans clustering on multi-frequency echosounder Sv data."
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
        help="Path to save cluster-map NetCDF (default: <stem>_kmeans.nc).",
    )

    # ---------------------------
    # Clustering model
    # ---------------------------
    parser.add_argument(
        "--model",
        type=str,
        choices=["abd", "dir"],
        default="abd",
        help="Clustering model: abd (absolute differences, default) or dir (direct).",
    )

    # ---------------------------
    # KMeans parameters
    # ---------------------------
    parser.add_argument(
        "-k", "--n_clusters",
        type=int,
        default=3,
        help="Number of KMeans clusters (default: 3).",
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
        "--n_init",
        type=int,
        default=10,
        help="Number of KMeans initialisations (default: 10).",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=300,
        help="Maximum iterations per KMeans run (default: 300).",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
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
            args.input_path.stem + "_kmeans"
        ).with_suffix(".nc")

    # ---------------------------
    # Perform clustering
    # ---------------------------
    try:
        if not args.quiet:
            args_dict = vars(args)
            pretty_args = pprint.pformat(args_dict)
            logger.debug(
                f"Executing aa-kmeans configured with [OPTIONS]:\n{pretty_args}\n"
                f"* ( Each aa-kmeans associated option_name may be "
                f"overridden using --option_name value )"
            )

        cluster_ds = cluster_dataset(
            ds,
            model=args.model,
            n_clusters=args.n_clusters,
            channels=args.channels,
            var=args.var,
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )

        # ---------------------------
        # Save output
        # ---------------------------
        logger.info(f"Saving cluster map to {args.output_path} ...")
        cluster_ds.to_netcdf(args.output_path, mode="w", format="NETCDF4")

        if not args.quiet:
            logger.info(
                f"Generating {args.output_path.resolve()} with aa-kmeans. "
                f"Passing nc path to stdout..."
            )

        # Print output path to stdout for piping
        print(args.output_path.resolve())

    except Exception as e:
        logger.exception(f"Error during KMeans clustering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()