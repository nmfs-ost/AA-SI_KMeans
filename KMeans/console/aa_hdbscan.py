#!/usr/bin/env python3
"""
Console tool for performing HDBSCAN clustering on echosounder NetCDF files
using the unified (alpha, beta) feature construction shared with aa-kmeans.

For each pixel x = (Sv_1, ..., Sv_N), the record fed to HDBSCAN is

    phi(x) = ( alpha * c_1, ..., alpha * c_N,  beta * Sv_mean )

with c_i = Sv_i - Sv_mean.  The named (alpha, beta) presets are
identical to aa-kmeans:

    --preset direct      ->  (alpha, beta) = (1, 1)
    --preset contrast    ->  (alpha, beta) = (1, 0)
    --preset loudness    ->  (alpha, beta) = (0, 1)

Unlike KMeans, HDBSCAN discovers the number of clusters automatically
from the data's density structure, exposes a per-cluster persistence
(stability) score, and provides per-pixel membership probabilities and
GLOSH outlier scores.  All four are written to the output NetCDF so
aa-report can build persistence-weighted fingerprints.

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

from KMeans.hdbscan_core import cluster_dataset
from KMeans.kmeans_core import PRESETS, list_channels, resolve_preset


def print_help():
    help_text = """
    Usage: aa-hdbscan [OPTIONS] [INPUT_PATH]

    Arguments:
    INPUT_PATH                  Path to a NetCDF file containing Sv data.
                                Optional. Defaults to stdin if not provided.

    Options:
    -o, --output_path           Path to save the cluster-map NetCDF.
                                Default: <stem>_hdbscan.nc

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

    --min_cluster_size          Smallest size a final cluster may have.
                                Larger values produce fewer, broader clusters.
                                Default: 5

    --min_samples               Conservativeness of clustering.  Larger
                                values label more points as noise.
                                Default: equal to --min_cluster_size

    --cluster_selection_method  How clusters are extracted from the
                                condensed tree.
                                Choices: eom, leaf
                                  eom  = excess-of-mass (default; picks
                                         the most stable clusters)
                                  leaf = leaves of the condensed tree
                                         (more, smaller clusters)

    --cluster_selection_epsilon Distance threshold for merging tiny
                                micro-clusters.  Useful when EOM picks
                                clusters that are too fine-grained.
                                Default: 0.0

    --metric                    Distance metric for core-distance and
                                neighbourhood computation.
                                Default: euclidean

    --channels                  Space-separated 0-based channel indices.
                                Default: all channels in the dataset.
                                Example: --channels 0 1 2

    --var                       Data variable to cluster on.
                                Default: Sv

    --core_dist_n_jobs          Parallel jobs for core-distance computation.
                                -1 = all available cores.
                                Default: -1

    --list_channels             List available channels and exit.

    --quiet                     Suppress logger info, only print output path.

    Description:
    The output NetCDF has the same spatial dimensions (ping_time x
    range_sample) as the input echogram and contains four variables:
        cluster_map              integer cluster labels (-1 = noise)
        membership_probability   per-pixel cluster strength in [0, 1]
        outlier_score            GLOSH outlier score per pixel
        cluster_persistence      stability score per discovered cluster
    aa-report consumes all four when building HDBSCAN fingerprints.

    Feature construction:
        For an N-channel pixel x = (Sv_1, ..., Sv_N), define
            Sv_mean = (1/N) sum_i Sv_i           (loudness)
            c_i     = Sv_i - Sv_mean             (colour)
        Then HDBSCAN is run on the record
            phi(x) = (alpha * c_1, ..., alpha * c_N, beta * Sv_mean).
        Columns whose weight is exactly zero are dropped before clustering.

    Examples:
      # Use a named preset
      echo file.nc | aa-hdbscan --preset contrast --min_cluster_size 50
      aa-hdbscan file.nc --preset direct --min_cluster_size 30 --channels 0 1 3

      # Specify alpha/beta directly and tighten micro-cluster merging
      aa-hdbscan file.nc --alpha 1 --beta 0.5 \\
          --min_cluster_size 100 --cluster_selection_epsilon 0.25 -o out.nc

      # Pipe straight into aa-report
      aa-hdbscan file.nc --preset contrast | aa-report --tag krill_swarm
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
            "Perform HDBSCAN clustering on multi-frequency echosounder Sv "
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
        help="Path to save cluster-map NetCDF (default: <stem>_hdbscan.nc).",
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
    # HDBSCAN parameters
    # ---------------------------
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=5,
        help="Smallest size a final cluster may have (default: 5).",
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help=(
            "Conservativeness of clustering. Defaults to --min_cluster_size."
        ),
    )

    parser.add_argument(
        "--cluster_selection_method",
        type=str,
        choices=["eom", "leaf"],
        default="eom",
        help="Cluster selection method (default: eom).",
    )

    parser.add_argument(
        "--cluster_selection_epsilon",
        type=float,
        default=0.0,
        help="Distance threshold for merging micro-clusters (default: 0.0).",
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
        "--core_dist_n_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for core-distance computation; -1 = all cores (default: -1).",
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

    if args.min_cluster_size < 2:
        logger.error("--min_cluster_size must be >= 2.")
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
            args.input_path.stem + "_hdbscan"
        ).with_suffix(".nc")

    # ---------------------------
    # Perform clustering
    # ---------------------------
    try:
        if not args.quiet:
            args_dict = vars(args)
            pretty_args = pprint.pformat(args_dict)
            logger.debug(
                f"Executing aa-hdbscan configured with [OPTIONS]:\n{pretty_args}\n"
                f"* ( Each aa-hdbscan associated option_name may be "
                f"overridden using --option_name value )"
            )

        cluster_ds = cluster_dataset(
            ds,
            alpha=args.alpha,
            beta=args.beta,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            metric=args.metric,
            channels=args.channels,
            var=args.var,
            core_dist_n_jobs=args.core_dist_n_jobs,
            preset=None,  # already resolved above
        )

        # ---------------------------
        # Save output
        # ---------------------------
        logger.info(f"Saving cluster map to {args.output_path} ...")
        cluster_ds.to_netcdf(args.output_path, mode="w", format="NETCDF4")

        if not args.quiet:
            logger.info(
                f"Generating {args.output_path.resolve()} with aa-hdbscan. "
                f"Passing nc path to stdout..."
            )

        # Print output path to stdout for piping
        print(args.output_path.resolve())

    except Exception as e:
        logger.exception(f"Error during HDBSCAN clustering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()