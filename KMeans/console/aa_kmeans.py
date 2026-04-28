#!/usr/bin/env python3
"""
Console tool for performing KMeans clustering on echosounder NetCDF files
using the unified (alpha, beta) feature construction.

For each pixel x = (Sv_1, ..., Sv_N), the record fed to KMeans is

    phi(x) = ( alpha * c_1, ..., alpha * c_N,  beta * Sv_mean )

with c_i = Sv_i - Sv_mean.  The three classical recipes are corners of
this construction:

    --preset direct      ->  (alpha, beta) = (1, 1)   info-equivalent to raw Sv
    --preset contrast    ->  (alpha, beta) = (1, 0)   colour-only (was: abd)
    --preset loudness    ->  (alpha, beta) = (0, 1)   mean-only

Anything in between is reachable by passing --alpha and --beta directly.
The recommended workflow (per the unified-framework memo) is to fix
alpha = 1 and sweep beta on a small grid, since only the ratio
beta/alpha matters to KMeans.

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

from KMeans.kmeans_core import (
    PRESETS,
    cluster_dataset,
    list_channels,
    resolve_preset,
)


def print_help():
    help_text = """
    Usage: aa-kmeans [OPTIONS] [INPUT_PATH]

    Arguments:
    INPUT_PATH                  Path to a NetCDF file containing Sv data.
                                Optional. Defaults to stdin if not provided.

    Options:
    -o, --output_path           Path to save the cluster-map NetCDF.
                                Default: <stem>_kmeans.nc

    --preset                    Named (alpha, beta) recipe.
                                Choices: direct, contrast, loudness
                                  direct   = (1, 1)  raw-Sv equivalent
                                  contrast = (1, 0)  colour-only (replaces 'abd')
                                  loudness = (0, 1)  mean-only
                                Aliases for back-compat: dir, abd, mean.
                                If both --preset and --alpha/--beta are
                                supplied, --preset wins.

    --alpha                     Weight on the colour (centered) component.
                                Must be >= 0.  Default: 1.0

    --beta                      Weight on the loudness (mean) component.
                                Must be >= 0.  Default: 1.0

    -k, --n_clusters            Number of KMeans clusters.
                                Default: 3

    --channels                  Space-separated 0-based channel indices.
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
    The output NetCDF has the same spatial dimensions (ping_time x
    range_sample) as the input echogram, but contains integer cluster
    labels instead of Sv values.  This "cluster map" can be visualised
    in the same way an echogram is plotted.

    Feature construction:
        For an N-channel pixel x = (Sv_1, ..., Sv_N), define
            Sv_mean = (1/N) sum_i Sv_i           (loudness)
            c_i     = Sv_i - Sv_mean             (colour)
        Then KMeans is run on the record
            phi(x) = (alpha * c_1, ..., alpha * c_N, beta * Sv_mean).
        Columns whose weight is exactly zero are dropped before clustering.

    Examples:
      # Use a named preset
      echo file.nc | aa-kmeans --preset contrast
      aa-kmeans file.nc --preset direct -k 5 --channels 0 1 3

      # Specify alpha/beta directly (any non-negative ratio works)
      aa-kmeans file.nc --alpha 1 --beta 0.5 -k 4 -o out.nc

      # Sweep beta with alpha fixed (recommended; only ratio matters)
      for b in 0 0.25 0.5 1 2 4; do
          aa-kmeans file.nc --alpha 1 --beta $b -o "out_b${b}.nc"
      done
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
            "Perform KMeans clustering on multi-frequency echosounder Sv "
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
        help="Path to save cluster-map NetCDF (default: <stem>_kmeans.nc).",
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
            "Named (alpha, beta) preset.  Overrides --alpha/--beta if given.  "
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
            alpha=args.alpha,
            beta=args.beta,
            n_clusters=args.n_clusters,
            channels=args.channels,
            var=args.var,
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.random_state,
            preset=None,  # already resolved above
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