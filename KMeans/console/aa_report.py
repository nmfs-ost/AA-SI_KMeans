#!/usr/bin/env python3
"""
Console tool for generating a cluster-ratio fingerprint report from
a cluster_map NetCDF file produced by aa-kmeans (or aa-dbscan).

The report is a structured YAML file capturing:
    - provenance (source file, tool, timestamp)
    - clustering configuration ((alpha, beta), preset, k, channels)
    - the cluster fingerprint p(R) = (p_1, ..., p_k) on the simplex
    - per-cluster centroids in raw Sv, colour (c_i), and loudness (Sv_mean)
      coordinates, so labels are physically interpretable across runs
    - region-of-interest scoping (whole grid, or a sub-region by index)
    - a user-assignable echoclassification tag

The fingerprint is Hellinger-ready: written as ordered probabilities on
the (k-1)-simplex, so two fingerprints from the *same* clustering run
can be compared directly via dH(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||.

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
from typing import Optional, Tuple, Dict, Any

import numpy as np
import xarray as xr
import yaml
from loguru import logger


REPORT_VERSION = "2.1"


# ===========================================================================
# Acoustic variable descriptor (read back from cluster_map attrs)
# ===========================================================================

def read_variable_descriptor(attrs: dict) -> dict:
    """Reconstruct the AcousticVariable descriptor from cluster_map attrs.

    aa-kmeans writes the descriptor as a flat set of `acoustic_variable_*`
    attributes.  We collect them back into a dict here.  Falls back to a
    minimal Sv-shaped descriptor for legacy cluster maps that predate the
    descriptor system, so old files still report sensibly.
    """
    desc = {}
    prefix = "acoustic_variable_"
    for k, v in attrs.items():
        if k.startswith(prefix):
            desc[k[len(prefix):]] = v

    if not desc:
        # Legacy cluster_map without descriptor — infer minimum from clustering_variable
        var_name = attrs.get("clustering_variable", "Sv")
        desc = {
            "name": var_name,
            "long_name": f"{var_name} (legacy, no descriptor)",
            "units": "dB (assumed)",
            "pixel_population": "Unspecified.",
            "loudness_meaning": "Per-pixel mean across selected channels.",
            "colour_meaning": "Per-channel deviation from the per-pixel mean.",
            "detection_filtered": "False",
            "notes": (
                "Cluster map predates the AcousticVariable descriptor "
                "system; interpret with care."
            ),
        }

    # Normalise the boolean
    df = str(desc.get("detection_filtered", "False")).lower() == "true"
    desc["detection_filtered_bool"] = df
    return desc


# ===========================================================================
# YAML formatting: keep dicts in insertion order, emit clean numbers
# ===========================================================================

def _represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())


def _represent_float(dumper, data):
    # Avoid scientific notation, excessive precision, and the noisy !!float tag
    if np.isnan(data):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "")
    if np.isinf(data):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf" if data > 0 else "-.inf")
    # Force a decimal point so YAML doesn't re-tag whole-number floats as ints
    text = f"{data:.6g}"
    if "." not in text and "e" not in text and "n" not in text:
        text += ".0"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


yaml.add_representer(OrderedDict, _represent_ordereddict)
yaml.add_representer(float, _represent_float)
yaml.add_representer(np.float32, _represent_float)
yaml.add_representer(np.float64, _represent_float)


# ===========================================================================
# Help
# ===========================================================================

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

    --tag                       Echoclassification tag for this fingerprint.
                                Free-form string linking the cluster profile
                                to a species, material, or biomass category.
                                Default: unclassified

    --source_sv                 Path to the original Sv NetCDF used to build
                                the cluster map.  If supplied, per-cluster
                                centroids in raw Sv, colour, and loudness
                                coordinates are computed and embedded in the
                                report.  Without this, the report still
                                generates but without centroids (label
                                interpretation must be inferred elsewhere).

    --roi_ping                  Restrict fingerprint to a ping_time range,
                                given as 'start,end' integer indices into
                                the ping_time dimension.  Example:
                                  --roi_ping 100,500
                                Default: full range.

    --roi_range                 Restrict fingerprint to a range_sample range,
                                given as 'start,end' integer indices into
                                the range/depth dimension.  Example:
                                  --roi_range 0,200
                                Default: full range.

    --var                       Cluster-map variable name in the NetCDF.
                                Default: cluster_map

    --sv_var                    Sv variable name in the source file (only
                                used with --source_sv).
                                Default: Sv

    --note                      Free-form note to embed in the report.

    --quiet                     Suppress logger info, only print output path.

    Description:
    Reads a cluster_map NetCDF and produces a YAML fingerprint p(R).  The
    report carries enough metadata that two fingerprints from the same
    clustering run can be compared by Hellinger distance on the simplex,
    and enough physical context (centroids in Sv space) that cluster
    labels remain interpretable across runs.

    Examples:
      # Whole-grid fingerprint, no centroids
      echo clustered.nc | aa-report --tag euphausiid

      # ROI-scoped fingerprint with centroids in Sv coordinates
      aa-report clustered.nc \\
          --source_sv original_Sv.nc \\
          --roi_ping 200,800 --roi_range 50,300 \\
          --tag krill_swarm -o krill_fp.yaml
    """
    print(help_text)


# ===========================================================================
# Fingerprint computation
# ===========================================================================

def _parse_roi(spec: Optional[str], dim_size: int, dim_name: str) -> Tuple[int, int]:
    """Parse a 'start,end' ROI spec into validated integer indices."""
    if spec is None:
        return 0, dim_size
    try:
        start, end = (int(s.strip()) for s in spec.split(","))
    except (ValueError, AttributeError):
        raise ValueError(f"--roi_{dim_name} must be 'start,end' integers; got {spec!r}")
    if start < 0 or end > dim_size or start >= end:
        raise ValueError(
            f"--roi_{dim_name}={spec} out of range for dim size {dim_size}"
        )
    return start, end


def compute_fingerprint(cluster_data: np.ndarray, n_clusters_hint: Optional[int] = None) -> Dict[str, Any]:
    """Compute cluster-ratio fingerprint p(R) from a 2-D label array.

    NaN and -1 are both treated as noise/invalid.  The fingerprint is the
    probability distribution over valid cluster labels and lies on the
    simplex Delta^{k-1}.

    Parameters
    ----------
    cluster_data : np.ndarray
        2-D array of integer cluster labels.  -1 or NaN = noise/invalid.
    n_clusters_hint : int or None
        If supplied, the fingerprint vector has this many entries even
        when some clusters have zero pixels in this ROI (so fingerprints
        from the same run share the same simplex).

    Returns
    -------
    dict
        Fingerprint with simplex vector, per-cluster details, and counts.
    """
    flat = cluster_data.ravel()
    total_pixels = int(len(flat))
    nan_mask = np.isnan(flat.astype(float))

    labels = flat.copy()
    labels[nan_mask] = -1
    labels = labels.astype(int)

    valid_mask = labels >= 0
    valid_pixels = int(valid_mask.sum())
    noise_pixels = total_pixels - valid_pixels

    # Decide which cluster ids to report
    present = sorted(int(u) for u in np.unique(labels[valid_mask])) if valid_pixels else []
    if n_clusters_hint is not None and n_clusters_hint > 0:
        all_ids = list(range(n_clusters_hint))
    else:
        all_ids = present

    counts = {cid: int(np.sum(labels == cid)) for cid in all_ids}

    # Simplex vector p(R), aligned with all_ids
    simplex = (
        [counts[cid] / valid_pixels for cid in all_ids] if valid_pixels else [0.0] * len(all_ids)
    )

    per_cluster = OrderedDict()
    for cid in all_ids:
        c = counts[cid]
        per_cluster[cid] = OrderedDict([
            ("pixel_count", c),
            ("ratio", round(c / valid_pixels, 6) if valid_pixels else 0.0),
            ("percent", round(100.0 * c / valid_pixels, 3) if valid_pixels else 0.0),
        ])

    # Dominant cluster (the simplex vertex this fingerprint sits closest to)
    if valid_pixels and per_cluster:
        dominant_id = max(per_cluster, key=lambda k: per_cluster[k]["pixel_count"])
        dominant_pct = per_cluster[dominant_id]["percent"]
    else:
        dominant_id = None
        dominant_pct = 0.0

    return OrderedDict([
        ("simplex_vector", [round(p, 6) for p in simplex]),
        ("simplex_basis", all_ids),
        ("dominant_cluster", dominant_id),
        ("dominant_percent", dominant_pct),
        ("n_clusters_present", len(present)),
        ("n_clusters_basis", len(all_ids)),
        ("total_pixels", total_pixels),
        ("valid_pixels", valid_pixels),
        ("noise_pixels", noise_pixels),
        ("noise_percent",
         round(100.0 * noise_pixels / total_pixels, 3) if total_pixels else 0.0),
        ("per_cluster", per_cluster),
    ])


# ===========================================================================
# Centroid recovery from source Sv
# ===========================================================================

def compute_centroids(
    cluster_data: np.ndarray,
    sv_ds: xr.Dataset,
    sv_var: str,
    channel_indices,
    ping_slice: slice,
    range_slice: slice,
) -> Optional[OrderedDict]:
    """Compute per-cluster centroids in raw, colour, and loudness coordinates.

    Works for any dB-domain variable (Sv, TS, ...).  Centroid keys are
    templated by the variable name so a TS clustering produces
    TS_per_channel_dB rather than Sv_per_channel_dB.

    For each cluster id present in *cluster_data* (within the ROI), pull
    the corresponding pixels from the source dataset and compute the
    mean per channel, the mean loudness (per-pixel mean), and the mean
    colour (centered) vector.
    """
    V = sv_var  # variable symbol used for templating ("Sv", "TS", ...)
    try:
        sv = sv_ds[sv_var].isel(
            channel=channel_indices,
            ping_time=ping_slice,
        )
        # range dim may be range_sample, depth, or echo_range
        range_dim = next(d for d in sv.dims if d not in ("channel", "ping_time"))
        sv = sv.isel({range_dim: range_slice})
    except Exception as e:
        logger.warning(f"Could not align {V} source to cluster map: {e}")
        return None

    # sv has shape (channel, ping, range); cluster_data is (ping, range)
    sv_arr = sv.values  # (N, P, R)
    if sv_arr.shape[1:] != cluster_data.shape:
        logger.warning(
            f"{V} ROI shape {sv_arr.shape[1:]} does not match cluster ROI "
            f"{cluster_data.shape}; skipping centroids."
        )
        return None

    n_chan = sv_arr.shape[0]
    flat_sv = sv_arr.reshape(n_chan, -1).T  # (n_pixels, n_chan)
    flat_labels = cluster_data.ravel().astype(float)
    flat_labels[np.isnan(flat_labels)] = -1
    flat_labels = flat_labels.astype(int)

    # Mean per pixel ("loudness") and centered colour
    with np.errstate(invalid="ignore"):
        mean_pix = np.nanmean(flat_sv, axis=1)
        centered = flat_sv - mean_pix[:, None]

    centroids = OrderedDict()
    for cid in sorted(set(int(x) for x in flat_labels) - {-1}):
        mask = flat_labels == cid
        if not mask.any():
            continue
        with np.errstate(invalid="ignore"):
            mean_per_chan = np.nanmean(flat_sv[mask], axis=0)
            loudness_mean = float(np.nanmean(mean_pix[mask]))
            colour_mean = np.nanmean(centered[mask], axis=0)

        centroids[cid] = OrderedDict([
            (f"{V}_per_channel_dB",
             [round(float(v), 3) if np.isfinite(v) else None for v in mean_per_chan]),
            (f"loudness_{V}_mean_dB",
             round(loudness_mean, 3) if np.isfinite(loudness_mean) else None),
            ("colour_c_per_channel_dB",
             [round(float(v), 3) if np.isfinite(v) else None for v in colour_mean]),
            ("n_pixels", int(mask.sum())),
        ])

    return centroids


# ===========================================================================
# Equations block (self-documenting math)
# ===========================================================================

def build_equations_block(
    alpha: float,
    beta: float,
    n_channels: int,
    var_name: str = "Sv",
) -> "OrderedDict":
    """Render the equations from the unified-framework proposal as a YAML block.

    *var_name* templates the variable symbol throughout (e.g. "Sv" or
    "TS") so the equations describe what was actually clustered.

    The block is keyed so a human reader sees, in order:
        1. the raw multifrequency vector x,
        2. its decomposition into loudness (mean) and colour (centered),
        3. the KMeans feature record phi(x) parametrized by (alpha, beta),
        4. the KMeans objective in terms of those weights,
        5. the named-preset corners,
        6. the Hellinger distance used to compare fingerprints.

    Each equation is plain ASCII with light unicode; no LaTeX.
    """
    V = var_name                       # e.g. "Sv" or "TS"
    Vmean = f"{V}_mean"                # e.g. "Sv_mean" or "TS_mean"

    # For the "this run" form, drop trivial *1 multipliers so it reads cleanly
    def _coef(v):
        if not np.isfinite(v):
            return None
        if v == 1:
            return ""
        if v == 0:
            return "0*"
        return f"{v:g}*"

    a_coef = _coef(alpha)
    b_coef = _coef(beta)
    N = n_channels if n_channels else "N"

    # Pretty-print the (alpha, beta) substituted form of phi(x)
    if a_coef is None or b_coef is None:
        phi_substituted = (
            f"phi(x) = ( alpha*c_1, ..., alpha*c_N, beta*{Vmean} )  "
            f"[alpha,beta unknown]"
        )
    elif alpha == 0:
        phi_substituted = (
            f"phi(x) = ( {b_coef}{Vmean} )    "
            f"[colour block dropped: alpha=0]"
        )
    elif beta == 0:
        phi_substituted = (
            f"phi(x) = ( {a_coef}c_1, {a_coef}c_2, ..., {a_coef}c_{N} )    "
            f"[loudness dropped: beta=0]"
        )
    else:
        phi_substituted = (
            f"phi(x) = ( {a_coef}c_1, {a_coef}c_2, ..., {a_coef}c_{N}, "
            f"{b_coef}{Vmean} )"
        )

    # KMeans objective written on a single line so YAML doesn't fold it
    kmeans_obj = (
        f"||phi(x) - mu||^2 = "
        f"alpha^2 * sum_i (c_i - mu_i^c)^2 + "
        f"beta^2 * ({Vmean} - mu^mean)^2"
    )

    eqs = OrderedDict()

    eqs["reference"] = (
        "Ryan, M.C. (2026). Proposal: An Unsupervised Companion to the "
        "Acoustic Feature Library.  Sections 3.1, 3.3, 3.4."
    )

    eqs["acoustic_variable"] = V

    eqs["1_raw_vector"] = OrderedDict([
        ("formula", f"x = ( {V}(f_1), {V}(f_2), ..., {V}(f_N) )"),
        ("meaning",
         f"Per-pixel multifrequency {V} vector.  N is the number of "
         f"channels selected for clustering."),
    ])

    eqs["2_loudness_colour_decomposition"] = OrderedDict([
        ("loudness", OrderedDict([
            ("formula", f"{Vmean} = (1/N) * sum_{{i=1..N}} {V}(f_i)"),
            ("meaning",
             f"Common-mode component.  One scalar per pixel.  See "
             f"acoustic_variable.loudness_meaning for the biological "
             f"interpretation specific to {V}."),
        ])),
        ("colour", OrderedDict([
            ("formula", f"c_i = {V}(f_i) - {Vmean},    i = 1..N"),
            ("meaning",
             f"Inter-frequency contrast vector.  N entries per pixel "
             f"with sum_i c_i = 0, so N-1 independent degrees of "
             f"freedom.  Mathematically equivalent (up to a choice of "
             f"reference) to the relative frequency response R(f) used "
             f"in the Korneliussen lineage; additive contamination in "
             f"dB cancels.  See acoustic_variable.colour_meaning for "
             f"the biological interpretation specific to {V}."),
        ])),
        ("invertibility", OrderedDict([
            ("note",
             f"(c_1, ..., c_N, {Vmean}) is a full-rank linear transform "
             f"of ({V}_1, ..., {V}_N).  No information is gained or "
             f"lost by the decomposition; only the geometry seen by "
             f"KMeans changes when the two subspaces are reweighted."),
        ])),
    ])

    eqs["3_feature_construction_phi"] = OrderedDict([
        ("formula_general",
         f"phi(x) = ( alpha*c_1, alpha*c_2, ..., alpha*c_N,  beta*{Vmean} )"),
        ("formula_this_run", phi_substituted),
        ("alpha_role",
         f"alpha = {alpha:g}  -- weight on the colour (inter-frequency) block. "
         f"Scales every c_i identically."),
        ("beta_role",
         f"beta  = {beta:g}  -- weight on the loudness scalar {Vmean}."),
        ("meaning",
         "The single record fed to KMeans per pixel.  Up to N+1 "
         "entries: N from the colour block, plus one loudness entry. "
         "When alpha=0 the colour block drops out; when beta=0 the "
         "loudness entry drops out."),
    ])

    eqs["4_kmeans_objective"] = OrderedDict([
        ("formula", kmeans_obj),
        ("ratio_invariance",
         "Only the ratio beta/alpha affects the partition.  KMeans is "
         "invariant to a global metric scale, so (alpha, beta) and "
         "(k*alpha, k*beta) for any k>0 produce identical clusterings. "
         "Recommended workflow: fix alpha=1, sweep beta."),
        ("this_run_ratio",
         f"beta/alpha = {(beta/alpha) if alpha > 0 else float('inf'):g}"),
    ])

    eqs["5_named_presets"] = OrderedDict([
        ("direct",   "(alpha, beta) = (1, 1)  -> information-equivalent to raw vector"),
        ("contrast", "(alpha, beta) = (1, 0)  -> colour-only; pure inter-frequency relations"),
        ("loudness", "(alpha, beta) = (0, 1)  -> mean-only; inter-frequency relations discarded"),
    ])

    eqs["6_region_fingerprint"] = OrderedDict([
        ("formula",
         "p(R) = ( p_1, ..., p_k ),    "
         "p_j = #{pixels in R with cluster label j} / #{pixels in R}"),
        ("meaning",
         "Region-level descriptor: the share of each cluster within "
         "the ROI.  Lies on the simplex Delta^{k-1} (entries non-"
         "negative, sum to 1).  For a monospecific aggregation, p(R) "
         "concentrates near a vertex; for a mixed aggregation, mass "
         "distributes across multiple entries.  Note: 'pixels' here "
         "means whatever pixel_population is defined for this "
         "acoustic variable (see acoustic_variable block)."),
    ])

    eqs["7_hellinger_distance"] = OrderedDict([
        ("formula",
         "dH(p, q) = (1 / sqrt(2)) * || sqrt(p) - sqrt(q) ||_2"),
        ("meaning",
         "Natural metric on the probability simplex; well-behaved at "
         "zero entries (unlike KL divergence).  Used to compare two "
         "fingerprints and to cluster fingerprints into a data-driven "
         "taxonomy (proposal Sec. 3.4)."),
    ])

    return eqs


# ===========================================================================
# Report builder
# ===========================================================================

def _parse_channels_attr(channels_attr) -> list:
    """Parse the channels_used attribute (stored as a string by aa-kmeans)."""
    if isinstance(channels_attr, (list, tuple, np.ndarray)):
        return [int(x) for x in channels_attr]
    s = str(channels_attr).strip()
    if not s or s == "unknown":
        return []
    s = s.strip("[]() ")
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        return []


def build_report(
    ds: xr.Dataset,
    input_path: Path,
    tag: str,
    var: str,
    note: Optional[str],
    source_sv_path: Optional[Path],
    sv_var: str,
    roi_ping: Optional[str],
    roi_range: Optional[str],
) -> dict:
    """Assemble the full report dictionary."""

    cluster_da = ds[var]
    full_dims = cluster_da.dims
    full_shape = cluster_da.shape
    attrs = ds.attrs

    # ------------------------------------------------------------------
    # Resolve ROI
    # ------------------------------------------------------------------
    ping_dim = next((d for d in full_dims if "ping" in d.lower() or "time" in d.lower()), full_dims[0])
    range_dim = next((d for d in full_dims if d != ping_dim), full_dims[-1])
    ping_size = ds.sizes[ping_dim]
    range_size = ds.sizes[range_dim]

    p0, p1 = _parse_roi(roi_ping, ping_size, "ping")
    r0, r1 = _parse_roi(roi_range, range_size, "range")

    roi_full = (p0 == 0 and p1 == ping_size and r0 == 0 and r1 == range_size)
    cluster_roi = cluster_da.isel({ping_dim: slice(p0, p1), range_dim: slice(r0, r1)}).values

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    provenance = OrderedDict([
        ("file", str(input_path.resolve())),
        ("tool", attrs.get("source_tool", "unknown")),
        ("generated_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("report_version", REPORT_VERSION),
    ])

    # ------------------------------------------------------------------
    # Clustering configuration (the "what produced this map" block)
    # ------------------------------------------------------------------
    is_dbscan = attrs.get("algorithm", "") == "DBSCAN"

    config = OrderedDict()
    config["algorithm"] = "DBSCAN" if is_dbscan else "KMeans"
    config["clustering_variable"] = attrs.get("clustering_variable", "Sv")

    if is_dbscan:
        config["eps"] = float(attrs.get("eps", float("nan")))
        config["min_samples"] = int(attrs.get("min_samples", -1))
        config["metric"] = attrs.get("metric", "unknown")
    else:
        config["n_clusters"] = int(attrs.get("n_clusters", -1))

    # Unified-framework metadata (new in v2.0)
    if "alpha" in attrs or "beta" in attrs:
        config["feature_construction"] = OrderedDict([
            ("alpha", float(attrs.get("alpha", float("nan")))),
            ("beta", float(attrs.get("beta", float("nan")))),
            ("beta_over_alpha", float(attrs.get("beta_over_alpha", float("nan")))),
            ("preset", attrs.get("preset", "custom")),
            ("feature_columns", attrs.get("feature_columns", "")),
            ("interpretation", _interpret_alpha_beta(
                float(attrs.get("alpha", float("nan"))),
                float(attrs.get("beta", float("nan"))),
            )),
        ])
    elif "clustering_model" in attrs:
        # Legacy aa-kmeans output (pre-unified-framework)
        config["legacy_model"] = attrs["clustering_model"]

    config["channels_used"] = _parse_channels_attr(attrs.get("channels_used", ""))

    # ------------------------------------------------------------------
    # Region of interest
    # ------------------------------------------------------------------
    region = OrderedDict([
        ("scope", "full_grid" if roi_full else "subregion"),
        (ping_dim, OrderedDict([
            ("start_index", p0), ("end_index", p1), ("size", p1 - p0),
        ])),
        (range_dim, OrderedDict([
            ("start_index", r0), ("end_index", r1), ("size", r1 - r0),
        ])),
    ])
    # Add time bounds if available
    if np.issubdtype(cluster_da.coords[ping_dim].dtype, np.datetime64):
        coord_vals = cluster_da.coords[ping_dim].values
        region[ping_dim]["start_time"] = str(coord_vals[p0])
        region[ping_dim]["end_time"] = str(coord_vals[p1 - 1])

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------
    n_hint = int(attrs.get("n_clusters", 0)) if not is_dbscan else None
    fingerprint = compute_fingerprint(cluster_roi, n_clusters_hint=n_hint)

    # ------------------------------------------------------------------
    # Centroids (optional, requires source Sv)
    # ------------------------------------------------------------------
    centroids_block = OrderedDict([
        ("available", False),
        ("reason", "no --source_sv provided"),
    ])
    if source_sv_path is not None:
        if not source_sv_path.exists():
            centroids_block["reason"] = f"source file not found: {source_sv_path}"
        else:
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    sv_ds = xr.open_dataset(source_sv_path)
                channels_used = _parse_channels_attr(attrs.get("channels_used", ""))
                if not channels_used:
                    channels_used = list(range(sv_ds.sizes.get("channel", 0)))
                centroids = compute_centroids(
                    cluster_roi,
                    sv_ds=sv_ds,
                    sv_var=sv_var,
                    channel_indices=channels_used,
                    ping_slice=slice(p0, p1),
                    range_slice=slice(r0, r1),
                )
                if centroids is not None:
                    centroids_block = OrderedDict([
                        ("available", True),
                        ("source_file", str(source_sv_path.resolve())),
                        ("source_variable", sv_var),
                        ("description",
                         f"Per-cluster mean {sv_var} (raw, dB), loudness "
                         f"({sv_var}_mean, dB), and colour (c_i = "
                         f"{sv_var}_i - {sv_var}_mean, dB) over pixels "
                         f"assigned to each cluster within the ROI. "
                         f"These make cluster ids interpretable across "
                         f"runs."),
                        ("by_cluster", centroids),
                    ])
            except Exception as e:
                centroids_block["reason"] = f"failed to compute centroids: {e}"

    # ------------------------------------------------------------------
    # Echoclassification
    # ------------------------------------------------------------------
    echoclass = OrderedDict([
        ("tag", tag),
        ("description",
         "User-assigned label linking this fingerprint to a species, "
         "material, or biomass category."),
    ])

    # ------------------------------------------------------------------
    # Comparability hint (so a downstream Hellinger tool knows the rules)
    # ------------------------------------------------------------------
    comparability = OrderedDict([
        ("simplex_dimension", fingerprint["n_clusters_basis"]),
        ("hellinger_compatible_with",
         "fingerprints from the same clustering run "
         "(same source file, same alpha/beta, same k, same channels)"),
        ("recommended_distance",
         "dH(p,q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2"),
    ])

    # ------------------------------------------------------------------
    # Acoustic variable descriptor (NEW: the scientific identity block)
    # ------------------------------------------------------------------
    var_desc = read_variable_descriptor(attrs)
    var_name = var_desc.get("name", "Sv")

    acoustic_variable = OrderedDict([
        ("name", var_name),
        ("long_name", var_desc.get("long_name", "")),
        ("units", var_desc.get("units", "")),
        ("pixel_population", var_desc.get("pixel_population", "")),
        ("loudness_meaning", var_desc.get("loudness_meaning", "")),
        ("colour_meaning", var_desc.get("colour_meaning", "")),
        ("detection_filtered", var_desc["detection_filtered_bool"]),
        ("notes", var_desc.get("notes", "")),
    ])

    # ------------------------------------------------------------------
    # Equations block (self-documenting math, templated by variable)
    # ------------------------------------------------------------------
    alpha_val = float(attrs.get("alpha", float("nan")))
    beta_val  = float(attrs.get("beta",  float("nan")))
    n_chan_val = len(_parse_channels_attr(attrs.get("channels_used", "")))
    equations = build_equations_block(
        alpha_val, beta_val, n_chan_val, var_name=var_name
    )

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    report = OrderedDict([
        ("aa_report", OrderedDict([
            ("provenance", provenance),
            ("echoclassification", echoclass),
            ("acoustic_variable", acoustic_variable),
            ("clustering_config", config),
            ("equations", equations),
            ("region_of_interest", region),
            ("fingerprint", fingerprint),
            ("cluster_centroids", centroids_block),
            ("comparability", comparability),
        ])),
    ])

    if note:
        report["aa_report"]["note"] = note

    return report


def _interpret_alpha_beta(alpha: float, beta: float) -> str:
    """Human-readable interpretation of where (alpha, beta) sits on the dial."""
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        return "unknown"
    if beta == 0 and alpha > 0:
        return "colour-only (inter-frequency relations; loudness discarded)"
    if alpha == 0 and beta > 0:
        return "loudness-only (mean Sv; inter-frequency relations discarded)"
    ratio = beta / alpha
    if ratio < 0.25:
        return f"colour-dominant (beta/alpha = {ratio:.3g}; loudness as tie-breaker)"
    if ratio > 4:
        return f"loudness-dominant (beta/alpha = {ratio:.3g})"
    return f"balanced (beta/alpha = {ratio:.3g}; close to raw-Sv equivalent at 1.0)"


# ===========================================================================
# YAML emission with section headers for human readability
# ===========================================================================

SECTION_HEADERS = {
    "provenance":         "# Where this report came from",
    "echoclassification": "# What this fingerprint represents (user-assigned)",
    "acoustic_variable":  "# What was clustered (Sv, TS, ...) and what it means physically",
    "clustering_config":  "# How the cluster map was produced",
    "equations":          "# Mathematical definitions (proposal Sec. 3.1, 3.3, 3.4)",
    "region_of_interest": "# Spatial scope of this fingerprint",
    "fingerprint":        "# The cluster-ratio fingerprint p(R) on the simplex",
    "cluster_centroids":  "# Physical interpretation of each cluster id",
    "comparability":      "# Rules for comparing this fingerprint to others",
    "note":               "# Free-form note",
}


def write_report_yaml(report: dict, path: Path) -> None:
    """Dump the report to YAML with a banner and section headers."""
    body = yaml.dump(
        report,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        indent=2,
    )

    # Inject section headers as comments above each top-level key under aa_report
    lines = body.splitlines()
    out = []
    for line in lines:
        # Section keys appear as "  provenance:" etc. (indent 2)
        stripped = line.rstrip()
        if stripped.startswith("  ") and not stripped.startswith("   "):
            key = stripped.strip().rstrip(":").split(":")[0]
            if key in SECTION_HEADERS:
                if out and out[-1] != "":
                    out.append("")
                out.append("  " + SECTION_HEADERS[key])
        out.append(line)

    banner = (
        "# ============================================================\n"
        "# aa-report  v" + REPORT_VERSION + "\n"
        "# Cluster-ratio fingerprint for echosounder cluster maps\n"
        "# ============================================================\n"
    )

    with open(path, "w") as fh:
        fh.write(banner)
        fh.write("\n".join(out))
        fh.write("\n")


# ===========================================================================
# CLI
# ===========================================================================

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
            "Generate a cluster-ratio fingerprint report from a cluster_map "
            "NetCDF, with full clustering provenance and (optionally) "
            "physical centroids."
        )
    )

    parser.add_argument("input_path", type=Path, nargs="?",
                        help="Path to the NetCDF file containing a cluster_map variable.")
    parser.add_argument("-o", "--output_path", type=Path,
                        help="Path to save the YAML report (default: <stem>_report.yaml).")
    parser.add_argument("--tag", type=str, default="unclassified",
                        help="Echoclassification tag. Default: unclassified.")
    parser.add_argument("--source_sv", type=Path, default=None,
                        help="Path to the original Sv NetCDF, for centroid computation.")
    parser.add_argument("--roi_ping", type=str, default=None,
                        help="ROI on ping_time as 'start,end' indices.")
    parser.add_argument("--roi_range", type=str, default=None,
                        help="ROI on range_sample as 'start,end' indices.")
    parser.add_argument("--var", type=str, default="cluster_map",
                        help="Cluster-map variable name (default: cluster_map).")
    parser.add_argument("--sv_var", type=str, default="Sv",
                        help="Sv variable name in --source_sv (default: Sv).")
    parser.add_argument("--note", type=str, default=None,
                        help="Free-form note to embed in the report.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress informational logging.")

    args = parser.parse_args()

    # ----- input validation -----
    if args.input_path is None:
        args.input_path = Path(sys.stdin.readline().strip())
        if not args.quiet:
            logger.info(f"Read input path from stdin: {args.input_path}")

    if not args.input_path.exists():
        logger.error(f"File '{args.input_path}' does not exist.")
        sys.exit(1)

    if args.input_path.suffix.lower() not in {".nc", ".netcdf4"}:
        logger.error(f"'{args.input_path.name}' is not a .nc/.netcdf4 file.")
        sys.exit(1)

    # ----- load -----
    buf = io.StringIO()
    with redirect_stdout(buf):
        ds = xr.open_dataset(args.input_path)

    if args.var not in ds:
        logger.error(
            f"Variable '{args.var}' not found in {args.input_path.name}. "
            f"Available: {list(ds.data_vars)}"
        )
        sys.exit(1)

    if args.output_path is None:
        args.output_path = args.input_path.with_stem(
            args.input_path.stem + "_report"
        ).with_suffix(".yaml")

    # ----- build & write -----
    try:
        if not args.quiet:
            logger.debug(
                f"Executing aa-report with [OPTIONS]:\n{pprint.pformat(vars(args))}"
            )

        report = build_report(
            ds,
            input_path=args.input_path,
            tag=args.tag,
            var=args.var,
            note=args.note,
            source_sv_path=args.source_sv,
            sv_var=args.sv_var,
            roi_ping=args.roi_ping,
            roi_range=args.roi_range,
        )
        write_report_yaml(report, args.output_path)

        if not args.quiet:
            logger.info(
                f"Generated {args.output_path.resolve()} with aa-report. "
                f"Passing yaml path to stdout..."
            )

        print(args.output_path.resolve())

    except Exception as e:
        logger.exception(f"Error during report generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()