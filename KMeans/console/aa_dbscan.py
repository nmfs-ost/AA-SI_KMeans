#!/usr/bin/env python3
"""
Console tool for performing DBSCAN clustering on echosounder NetCDF files
using the unified (alpha, beta) feature construction shared with
aa-kmeans and aa-hdbscan.

For each pixel x = (Sv_1, ..., Sv_N), the record fed to DBSCAN is

    phi(x) = ( alpha * c_1, ..., alpha * c_N,  beta * Sv_mean )

with c_i = Sv_i - Sv_mean.  The named (alpha, beta) presets are
identical to aa-kmeans and aa-hdbscan:

    --preset direct      ->  (alpha, beta) = (1, 1)
    --preset contrast    ->  (alpha, beta) = (1, 0)
    --preset loudness    ->  (alpha, beta) = (0, 1)

Unlike KMeans, DBSCAN discovers the number of clusters automatically
from the data's density structure and labels points outside any dense
region as noise (-1).  Unlike HDBSCAN, DBSCAN does not produce
per-cluster persistence or per-pixel membership scores.

Follows the aa-* console-tool architecture: accepts an input path from
STDIN or as a positional argument, performs a single well-defined
operation, and prints the output file path to STDOUT for piping.
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
from KMeans.kmeans_core import PRESETS, list_channels, resolve_preset


def print_help():
    help_text = """
    Usage: aa-dbscan [OPTIONS] [INPUT_PATH]

    Arguments:
    INPUT_PATH                  Path to a NetCDF file containing Sv data.
                                Optional. Defaults to stdin if not provided.

    Options:
    -o, --output_path           Path to save the cluster-map NetCDF.
                                Default: <stem>_dbscan.nc

    --preset                    Named (alpha, beta) recipe.
                                Choices: direct, contrast, loudness
                                  direct   = (1, 1)  raw-Sv equivalent
                                  contrast = (1, 0)  colour-only
                                  loudness = (0, 1)  mean-only
                                Aliases for back-compat: dir, abd, mean.
                                If both --preset and --alpha/--beta are
                                supplied, --preset wins.

    --alpha                     Weight on the colour (centered) component.
                                Must be >= 0.  Default: 1.0

    --beta                      Weight on the loudness (mean) component.
                                Must be >= 0.  Default: 1.0

    --eps                       Maximum neighbourhood radius in phi-space.
                                NOTE: eps is in phi-space units, so it must
                                be rescaled when --alpha or --beta change.
                                Default: 0.5

    --min_samples               Minimum population for a point to be a
                                core point (forms a dense region).
                                Default: 5

    --metric                    Distance metric for neighbourhood
                                computation.
                                Default: euclidean

    --channels                  Space-separated 0-based channel indices.
                                Default: all channels in the dataset.
                                Example: --channels 0 1 2

    --var                       Data variable to cluster on.
                                Default: Sv

    --n_jobs                    Parallel jobs for distance computation.
                                -1 = all available cores.
                                Default: None (single-threaded)

    --list_channels             List available channels and exit.

    --quiet                     Suppress logger info, only print output path.

    Description:
    Performs DBSCAN clustering on multi-frequency echosounder data using
    the same (alpha, beta) feature space as aa-kmeans and aa-hdbscan.
    The output NetCDF has the same spatial dimensions (ping_time x
    range_sample) as the input echogram and contains the variable:
        cluster_map      integer cluster labels (-1 = noise)

    aa-report consumes this output and produces a sorted-spectrum
    fingerprint identical in shape to the HDBSCAN fingerprint (without
    the persistence-weighted variant, which DBSCAN does not provide).

    Feature construction:
        For an N-channel pixel x = (Sv_1, ..., Sv_N), define
            Sv_mean = (1/N) sum_i Sv_i           (loudness)
            c_i     = Sv_i - Sv_mean             (colour)
        Then DBSCAN is run on the record
            phi(x) = (alpha * c_1, ..., alpha * c_N, beta * Sv_mean).
        Columns whose weight is exactly zero are dropped before clustering.

    Examples:
      # Use a named preset
      echo file.nc | aa-dbscan --preset contrast --eps 1.0 --min_samples 10
      aa-dbscan file.nc --preset direct --eps 0.5 --channels 0 1 3

      # Specify alpha/beta directly
      aa-dbscan file.nc --alpha 1 --beta 0.5 --eps 0.7 -o out.nc

      # Pipe straight into aa-report
      aa-dbscan file.nc --preset contrast --eps 1.0 | aa-report --tag krill
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
        description=(
            "Perform DBSCAN clustering on multi-frequency echosounder Sv "
            "data using the unified (alpha, beta) feature construction."
        )
    )

    # ---------------------------
    # File arguments
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
    # Feature construction
    # ---------------------------
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default=None,
        help=(
            "Named (alpha, beta) preset.  Overrides --alpha/--beta if given. "
            "direct=(1,1), contrast=(1,0), loudness=(0,1)."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight on colour (centered) component (default: 1.0).",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on loudness (mean) component (default: 1.0).",
    )

    # ---------------------------
    # DBSCAN parameters
    # ---------------------------
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Maximum neighbourhood radius in phi-space (default: 0.5).",
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
        help="Parallel jobs for distance computation; -1 = all cores (default: None).",
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
    # Resolve preset -> (alpha, beta) (preset wins over explicit values)
    # ---------------------------
    if args.preset is not None:
        args.alpha, args.beta = resolve_preset(args.preset)
        if not args.quiet:
            logger.info(
                f"Preset '{args.preset}' -> alpha={args.alpha}, beta={args.beta}"
            )

    if args.alpha < 0 or args.beta < 0:
        logger.error(
            f"alpha and beta must be non-negative; got alpha={args.alpha}, "
            f"beta={args.beta}."
        )
        sys.exit(1)
    if args.alpha == 0 and args.beta == 0:
        logger.error("alpha and beta cannot both be zero.")
        sys.exit(1)

    if args.eps <= 0:
        logger.error("--eps must be > 0.")
        sys.exit(1)
    if args.min_samples < 1:
        logger.error("--min_samples must be >= 1.")
        sys.exit(1)

    # ---------------------------
    # Validate input path
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
    # Default output path
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
            alpha=args.alpha,
            beta=args.beta,
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
            channels=args.channels,
            var=args.var,
            n_jobs=args.n_jobs,
            preset=None,  # already resolved above
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