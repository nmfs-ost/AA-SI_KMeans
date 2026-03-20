#!/usr/bin/env python3
"""
Console tool for generating a cluster-ratio fingerprint report from
a cluster_map NetCDF file produced by aa-kmeans or aa-dbscan.

The report is a lightweight YAML file containing:
    - source metadata (tool, model, eps/k, channels, etc.)
    - cluster ratio profile (percentage of each cluster label)
    - a user-assignable echoclassification tag for linking the
      fingerprint to species, material, or biomass categories

Follows the aa-* console-tool architecture:
    - Accepts a file path from STDIN or as a positional argument
    - Performs a single, well-defined operation
    - Prints the output file path to STDOUT for piping
"""

import argparse
import io
import pprint
import sys
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from loguru import logger


def print_help():
    help_text = """
    Usage: aa-report [OPTIONS] [INPUT_PATH]

    Arguments:
    INPUT_PATH                  Path to a NetCDF file containing a cluster_map
                                variable (output of aa-kmeans or aa-dbscan).
                                Optional. Defaults to stdin if not provided.

    Options:
    -o, --output_path           Path to save the YAML report.
                                Default: <stem>_report.yaml

    --tag                       Echoclassification tag to associate with this
                                fingerprint.  Free-form string meant to link
                                this cluster profile to a species, material,
                                or biomass category.
                                Default: unclassified
                                Examples:
                                  --tag "euphausiid"
                                  --tag "myctophidae"
                                  --tag "krill_swarm"
                                  --tag "mixed_layer_01"

    --var                       Name of the cluster-map variable in the NetCDF.
                                Default: cluster_map

    --note                      Free-form note to embed in the report.
                                Default: None

    --quiet                     Suppress logger info, only print output path.

    Description:
    Reads a cluster_map NetCDF (from aa-kmeans or aa-dbscan) and produces
    a small YAML file capturing the cluster-ratio fingerprint — the
    relative proportion of each cluster label across the entire grid.

    This fingerprint is designed to be a low-footprint, disk-friendly
    summary of the biomass composition under the transducer.  Over time
    these profiles can be collected, compared, and linked to an
    echoclassification system via the --tag field.

    The report includes:
      - source file and tool metadata
      - clustering parameters (model, k or eps, channels, etc.)
      - per-cluster pixel counts, ratios, and percentages
      - noise fraction (for DBSCAN, label -1)
      - an echoclassification tag for species/material association

    Examples:
      echo clustered.nc | aa-report --tag "euphausiid"
      aa-report kmeans_out.nc --tag "krill_swarm" -o profile.yaml
      aa-report dbscan_out.nc --tag "myctophidae" --note "survey leg 3"
    """
    print(help_text)


def compute_fingerprint(cluster_data: np.ndarray) -> dict:
    """Compute cluster-ratio fingerprint from a 2-D cluster label array.

    Parameters
    ----------
    cluster_data : np.ndarray
        2-D array of integer cluster labels.  -1 = noise/NaN.

    Returns
    -------
    dict
        Fingerprint with per-cluster counts, ratios, percentages,
        plus summary statistics.
    """
    flat = cluster_data.ravel()
    total_pixels = len(flat)
    nan_mask = np.isnan(flat.astype(float))
    valid_pixels = int((~nan_mask).sum())

    # Treat NaN as -1 for counting purposes
    labels = flat.copy()
    labels[nan_mask] = -1
    labels = labels.astype(int)

    unique, counts = np.unique(labels, return_counts=True)

    clusters = OrderedDict()
    noise_count = 0

    for label, count in sorted(zip(unique, counts)):
        count = int(count)
        label = int(label)
        if label == -1:
            noise_count = count
            continue
        clusters[label] = {
            "pixel_count": count,
            "ratio": round(count / valid_pixels, 6) if valid_pixels > 0 else 0.0,
            "percent": round(100.0 * count / valid_pixels, 2) if valid_pixels > 0 else 0.0,
        }

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "noise_pixels": noise_count,
        "noise_percent": round(100.0 * noise_count / total_pixels, 2) if total_pixels > 0 else 0.0,
        "n_clusters": len(clusters),
        "clusters": dict(clusters),
    }


def build_report(
    ds: xr.Dataset,
    input_path: Path,
    tag: str = "unclassified",
    var: str = "cluster_map",
    note: str = None,
) -> dict:
    """Build the full YAML report dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        Cluster-map dataset.
    input_path : Path
        Path to the source .nc file.
    tag : str
        Echoclassification tag.
    var : str
        Cluster-map variable name.
    note : str or None
        Optional free-form note.

    Returns
    -------
    dict
        Complete report structure.
    """
    cluster_data = ds[var].values

    # Pull metadata baked in by aa-kmeans / aa-dbscan
    attrs = ds.attrs

    source_info = {
        "file": str(input_path.resolve()),
        "tool": attrs.get("source_tool", "unknown"),
        "clustering_variable": attrs.get("clustering_variable", "Sv"),
        "clustering_model": attrs.get("clustering_model", "unknown"),
        "channels_used": attrs.get("channels_used", "unknown"),
    }

    # Algorithm-specific metadata
    algorithm = attrs.get("algorithm", None)
    if algorithm == "DBSCAN":
        source_info["algorithm"] = "DBSCAN"
        source_info["eps"] = float(attrs.get("eps", -1))
        source_info["min_samples"] = int(attrs.get("min_samples", -1))
        source_info["metric"] = attrs.get("metric", "unknown")
    else:
        source_info["algorithm"] = "KMeans"
        source_info["n_clusters"] = int(attrs.get("n_clusters", -1))

    # Spatial extent
    dims_info = {}
    for dim in ds[var].dims:
        coord = ds[var].coords[dim]
        vals = coord.values
        dims_info[dim] = {
            "size": int(len(vals)),
        }
        # Add time range for ping_time
        if "time" in dim.lower() and np.issubdtype(vals.dtype, np.datetime64):
            dims_info[dim]["start"] = str(vals[0])
            dims_info[dim]["end"] = str(vals[-1])

    fingerprint = compute_fingerprint(cluster_data)

    report = {
        "aa_report": {
            "version": "1.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "source": source_info,
            "dimensions": dims_info,
            "echoclassification": {
                "tag": tag,
                "description": (
                    "User-assigned label linking this cluster profile "
                    "to a species, material, or biomass category."
                ),
            },
            "fingerprint": fingerprint,
        },
    }

    if note:
        report["aa_report"]["note"] = note

    return report


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
        description="Generate a cluster-ratio fingerprint report from a cluster_map NetCDF."
    )

    # ---------------------------
    # Required file arguments
    # ---------------------------
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the NetCDF file containing a cluster_map variable.",
        nargs="?",
    )

    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        help="Path to save the YAML report (default: <stem>_report.yaml).",
    )

    # ---------------------------
    # Report options
    # ---------------------------
    parser.add_argument(
        "--tag",
        type=str,
        default="unclassified",
        help="Echoclassification tag (species, material, biomass). Default: unclassified.",
    )

    parser.add_argument(
        "--var",
        type=str,
        default="cluster_map",
        help="Cluster-map variable name in the NetCDF (default: cluster_map).",
    )

    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-form note to embed in the report.",
    )

    # ---------------------------
    # Utility flags
    # ---------------------------
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
    # Load dataset
    # ---------------------------
    f = io.StringIO()
    with redirect_stdout(f):
        ds = xr.open_dataset(args.input_path)

    if args.var not in ds:
        logger.error(
            f"Variable '{args.var}' not found in {args.input_path.name}. "
            f"Available variables: {list(ds.data_vars)}"
        )
        sys.exit(1)

    # ---------------------------
    # Set default output path
    # ---------------------------
    if args.output_path is None:
        args.output_path = args.input_path.with_stem(
            args.input_path.stem + "_report"
        ).with_suffix(".yaml")

    # ---------------------------
    # Build and write report
    # ---------------------------
    try:
        if not args.quiet:
            args_dict = vars(args)
            pretty_args = pprint.pformat(args_dict)
            logger.debug(
                f"Executing aa-report configured with [OPTIONS]:\n{pretty_args}\n"
                f"* ( Each aa-report associated option_name may be "
                f"overridden using --option_name value )"
            )

        report = build_report(
            ds,
            input_path=args.input_path,
            tag=args.tag,
            var=args.var,
            note=args.note,
        )

        # Write YAML
        with open(args.output_path, "w") as fh:
            yaml.dump(
                report,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        if not args.quiet:
            logger.info(
                f"Generating {args.output_path.resolve()} with aa-report. "
                f"Passing yaml path to stdout..."
            )

        # Print output path to stdout for piping
        print(args.output_path.resolve())

    except Exception as e:
        logger.exception(f"Error during report generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()