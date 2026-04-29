"""
hdbscan_core.py

Modular HDBSCAN clustering operations for echosounder NetCDF data.

This module is the HDBSCAN counterpart of :mod:`KMeans.kmeans_core` and
deliberately mirrors its API so the two algorithms operate in the same
feature space and produce structurally similar outputs.

Per pixel x = (Sv_1, ..., Sv_N), the record fed to HDBSCAN is

    phi(x) = ( alpha * c_1, ..., alpha * c_N,  beta * Sv_mean )

with c_i = Sv_i - Sv_mean.  The named (alpha, beta) presets
(``direct``, ``contrast``, ``loudness``) are imported from kmeans_core
unchanged.  Only the ratio beta/alpha matters to a Euclidean clusterer.

What HDBSCAN buys over DBSCAN
-----------------------------
HDBSCAN extends DBSCAN with a hierarchical density model and exposes
three quantities that are central to a meaningful fingerprint:

    cluster_persistence_   stability score per discovered cluster
                           (higher = more robust to scale changes)
    probabilities_         per-pixel membership strength in [0, 1]
    outlier_scores_        per-pixel GLOSH outlier score

All three are preserved in the output NetCDF so aa-report can build
persistence-weighted fingerprints and quote diagnostic distributions.

Designed to be consumed by the aa-hdbscan console tool following the
Unix philosophy of small, composable programs.
"""

from typing import List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

# Re-use all feature-matrix logic from kmeans_core — no duplication.
from KMeans.kmeans_core import (
    build_feature_matrix,
    get_variable_descriptor,
    list_channels,
    resolve_channel_indices,
    resolve_preset,
)


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------

def run_hdbscan(
    features: pd.DataFrame,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    core_dist_n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run HDBSCAN on a feature matrix.

    NaN rows are assigned a label of ``-1`` and a probability of ``0``.
    Valid rows receive a cluster id in ``[0, n_clusters_discovered)`` or
    ``-1`` if classified as noise.

    Parameters
    ----------
    features : pd.DataFrame
        The pre-clustering feature matrix (pixels x features).
    min_cluster_size : int
        The smallest size a final cluster is allowed to have.
    min_samples : int or None
        Conservativeness of clustering.  Larger -> more points labelled
        as noise.  Defaults to ``min_cluster_size`` when ``None``.
    cluster_selection_method : {"eom", "leaf"}
        Excess-of-mass (default) selects the most stable clusters; leaf
        selects the leaves of the condensed tree.
    cluster_selection_epsilon : float
        Distance threshold used to merge tiny micro-clusters.
    metric : str
        Distance metric (default ``"euclidean"``).
    core_dist_n_jobs : int
        Parallel jobs for core-distance computation (``-1`` = all cores).

    Returns
    -------
    labels : np.ndarray of int
        Cluster labels aligned with rows of *features*.  ``-1`` = noise.
    probabilities : np.ndarray of float
        Per-pixel membership strength in ``[0, 1]``.  Noise / NaN
        pixels receive ``0``.
    outlier_scores : np.ndarray of float
        GLOSH outlier scores in ``[0, 1]``.  NaN where input was NaN.
    cluster_persistence : np.ndarray of float
        Stability score per cluster, indexed by cluster id in
        ``[0, n_clusters_discovered)``.  Empty if no clusters.
    """
    n = len(features)
    labels = np.full(n, -1, dtype=int)
    probabilities = np.zeros(n, dtype=float)
    outlier_scores = np.full(n, np.nan, dtype=float)

    valid_mask = features.notna().all(axis=1).values
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        logger.warning("No valid (non-NaN) pixels — returning all noise.")
        return labels, probabilities, outlier_scores, np.array([], dtype=float)

    logger.info(
        f"Running HDBSCAN  min_cluster_size={min_cluster_size}  "
        f"min_samples={min_samples}  selection={cluster_selection_method}  "
        f"epsilon={cluster_selection_epsilon}  metric={metric}  "
        f"valid_pixels={n_valid}/{n}"
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        core_dist_n_jobs=core_dist_n_jobs,
        gen_min_span_tree=False,
    )
    clusterer.fit(features.values[valid_mask])

    labels[valid_mask] = clusterer.labels_
    probabilities[valid_mask] = clusterer.probabilities_
    outlier_scores[valid_mask] = clusterer.outlier_scores_
    cluster_persistence = np.asarray(
        clusterer.cluster_persistence_, dtype=float
    )

    n_clusters = int(len(cluster_persistence))
    n_noise = int((clusterer.labels_ == -1).sum())
    logger.info(
        f"HDBSCAN discovered {n_clusters} cluster(s); "
        f"{n_noise}/{n_valid} valid pixels classified as noise."
    )
    return labels, probabilities, outlier_scores, cluster_persistence


# ---------------------------------------------------------------------------
# Output dataset construction
# ---------------------------------------------------------------------------

def labels_to_dataset(
    ds: xr.Dataset,
    labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    cluster_persistence: np.ndarray,
    var: str = "Sv",
) -> xr.Dataset:
    """Reshape HDBSCAN outputs back into the spatial grid of the echogram.

    Output dataset variables
    ------------------------
    cluster_map              (ping, range)  int   -1 = noise
    membership_probability   (ping, range)  float in [0, 1]
    outlier_score            (ping, range)  float (GLOSH); NaN where invalid
    cluster_persistence      (cluster_id,)  float, stability per cluster

    The AcousticVariable descriptor for *var* is embedded in the global
    attributes (mirroring kmeans_core.labels_to_dataset) so aa-report
    can render the variable's physical meaning identically across
    KMeans and HDBSCAN runs.
    """
    template = ds[var].isel(channel=0)
    shape = template.shape
    dims = template.dims
    coords = {d: template.coords[d] for d in dims}

    cluster_2d = labels.reshape(shape)
    prob_2d = probabilities.reshape(shape)
    glosh_2d = outlier_scores.reshape(shape)

    desc = get_variable_descriptor(var)

    cluster_da = xr.DataArray(
        data=cluster_2d,
        dims=dims,
        coords=coords,
        name="cluster_map",
        attrs={
            "long_name": "HDBSCAN cluster assignment",
            "description": (
                f"Integer cluster labels produced by HDBSCAN clustering "
                f"of the unified (alpha, beta) feature matrix on multi-"
                f"frequency {desc.long_name} data.  -1 indicates noise."
            ),
            "units": "1",
        },
    )

    prob_da = xr.DataArray(
        data=prob_2d,
        dims=dims,
        coords=coords,
        name="membership_probability",
        attrs={
            "long_name": "HDBSCAN cluster membership probability",
            "description": (
                "Per-pixel strength of cluster membership in [0, 1]. "
                "Noise points receive 0."
            ),
            "units": "1",
        },
    )

    glosh_da = xr.DataArray(
        data=glosh_2d,
        dims=dims,
        coords=coords,
        name="outlier_score",
        attrs={
            "long_name": "GLOSH outlier score",
            "description": (
                "Global-Local Outlier Score from Hierarchies (GLOSH). "
                "Higher values indicate stronger outlying behaviour. "
                "NaN where the input feature row was invalid."
            ),
            "units": "1",
        },
    )

    persist_da = xr.DataArray(
        data=cluster_persistence,
        dims=("cluster_id",),
        coords={"cluster_id": np.arange(len(cluster_persistence), dtype=int)},
        name="cluster_persistence",
        attrs={
            "long_name": "HDBSCAN cluster persistence (stability)",
            "description": (
                "Stability score for each discovered cluster, indexed "
                "by the cluster id used in cluster_map.  Higher values "
                "indicate clusters that survive across a wider range "
                "of density thresholds."
            ),
            "units": "1",
        },
    )

    out_ds = xr.Dataset({
        "cluster_map": cluster_da,
        "membership_probability": prob_da,
        "outlier_score": glosh_da,
        "cluster_persistence": persist_da,
    })

    out_ds.attrs["source_tool"] = "aa-hdbscan"
    out_ds.attrs["clustering_variable"] = var

    # Embed the AcousticVariable descriptor for aa-report
    for key, val in desc.as_dict().items():
        out_ds.attrs[f"acoustic_variable_{key}"] = val

    return out_ds


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def cluster_dataset(
    ds: xr.Dataset,
    alpha: float = 1.0,
    beta: float = 1.0,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    channels: Optional[List[int]] = None,
    var: str = "Sv",
    core_dist_n_jobs: int = -1,
    preset: Optional[str] = None,
) -> xr.Dataset:
    """End-to-end: build features -> run HDBSCAN -> return labelled dataset.

    Mirrors :func:`KMeans.kmeans_core.cluster_dataset` so HDBSCAN runs
    in the same feature space as KMeans for any given preset or
    weight choice — meaning a fingerprint produced from an HDBSCAN
    run can be referenced against a codebook learned from KMeans
    runs at the same (alpha, beta).

    Parameters
    ----------
    ds : xr.Dataset
        Sv (or other variable) dataset.
    alpha : float, default 1.0
        Weight on the colour (centered) component.
    beta : float, default 1.0
        Weight on the loudness (mean) component.
    min_cluster_size : int, default 5
        Smallest size a final cluster is allowed to have.
    min_samples : int or None
        Conservativeness; defaults to ``min_cluster_size`` when ``None``.
    cluster_selection_method : {"eom", "leaf"}
        How clusters are extracted from the condensed tree.
    cluster_selection_epsilon : float, default 0.0
        Optional distance threshold for merging micro-clusters.
    metric : str, default "euclidean"
    channels : list of int or None
    var : str
    core_dist_n_jobs : int, default -1
        Parallel jobs for core-distance computation.
    preset : str or None
        If given, overrides *alpha* and *beta* with PRESETS[preset].

    Returns
    -------
    xr.Dataset
        Dataset with ``cluster_map``, ``membership_probability``,
        ``outlier_score``, and ``cluster_persistence`` variables, plus
        the (alpha, beta) and HDBSCAN-parameter metadata in the
        global attributes that aa-report consumes.
    """
    if preset is not None:
        alpha, beta = resolve_preset(preset)

    features = build_feature_matrix(
        ds, alpha=alpha, beta=beta, channel_indices=channels, var=var
    )
    labels, probabilities, outlier_scores, cluster_persistence = run_hdbscan(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        core_dist_n_jobs=core_dist_n_jobs,
    )

    out = labels_to_dataset(
        ds, labels, probabilities, outlier_scores, cluster_persistence, var=var
    )

    n_clusters = int(len(cluster_persistence))

    out.attrs["algorithm"] = "HDBSCAN"
    out.attrs["alpha"] = float(alpha)
    out.attrs["beta"] = float(beta)
    out.attrs["beta_over_alpha"] = (
        float(beta) / float(alpha) if alpha > 0 else float("inf")
    )
    out.attrs["preset"] = preset if preset is not None else "custom"
    out.attrs["min_cluster_size"] = int(min_cluster_size)
    out.attrs["min_samples"] = (
        int(min_samples) if min_samples is not None else int(min_cluster_size)
    )
    out.attrs["cluster_selection_method"] = str(cluster_selection_method)
    out.attrs["cluster_selection_epsilon"] = float(cluster_selection_epsilon)
    out.attrs["metric"] = str(metric)
    out.attrs["n_clusters_discovered"] = n_clusters
    out.attrs["n_noise_pixels"] = int((labels == -1).sum())
    out.attrs["channels_used"] = str(
        resolve_channel_indices(ds, channels, var=var)
    )
    out.attrs["feature_columns"] = ", ".join(features.columns)
    return out