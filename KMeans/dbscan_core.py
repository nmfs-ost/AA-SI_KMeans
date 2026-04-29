"""
dbscan_core.py

Modular DBSCAN clustering operations for echosounder NetCDF data.

This module is the DBSCAN counterpart of :mod:`KMeans.kmeans_core` and
:mod:`KMeans.hdbscan_core`, and deliberately mirrors their API so all
three clustering tools operate in the same feature space and produce
structurally identical output schemas.

Per pixel x = (Sv_1, ..., Sv_N), the record fed to DBSCAN is

    phi(x) = ( alpha * c_1, ..., alpha * c_N,  beta * Sv_mean )

with c_i = Sv_i - Sv_mean.  The named (alpha, beta) presets
(``direct``, ``contrast``, ``loudness``) are imported from kmeans_core
unchanged.  Only the ratio beta/alpha matters to the partition itself,
but ``eps`` is in phi-space units, so it must be rescaled when alpha
or beta change.

Unlike KMeans, DBSCAN does not require the number of clusters to be
specified in advance.  Instead it takes two parameters:

    eps          maximum neighbourhood radius in phi-space
    min_samples  minimum population for a point to be a core point

Points that do not belong to any cluster are labelled ``-1`` (noise).
DBSCAN produces no per-cluster persistence or per-pixel membership
scores — those are HDBSCAN-only — so the output here is the cluster
map only, with the same metadata schema KMeans and HDBSCAN write so
``aa-report`` consumes all three uniformly.

Designed to be consumed by the aa-dbscan console tool following the
Unix philosophy of small, composable programs.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from sklearn.cluster import DBSCAN

# Re-use all feature-matrix logic from kmeans_core — no duplication.
from KMeans.kmeans_core import (
    build_feature_matrix,
    get_variable_descriptor,
    list_channels,
    resolve_channel_indices,
    resolve_preset,
)


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

def run_dbscan(
    features: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Run DBSCAN on a feature matrix and return integer cluster labels.

    NaN rows are assigned a label of ``-1`` (same convention DBSCAN uses
    for noise points).  All other rows receive a cluster id in
    ``[0, n_discovered)`` or ``-1`` if classified as noise.

    Parameters
    ----------
    features : pd.DataFrame
        The pre-clustering feature matrix (pixels x features).
    eps : float
        Maximum neighbourhood radius in phi-space.
    min_samples : int
        Minimum points to form a dense region.
    metric : str
        Distance metric (default ``"euclidean"``).
    n_jobs : int or None
        Parallel jobs for distance computation (``-1`` = all cores).

    Returns
    -------
    np.ndarray of int
        Cluster labels aligned with the rows of *features*.
    """
    labels = np.full(len(features), -1, dtype=int)
    valid_mask = features.notna().all(axis=1).values
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        logger.warning("No valid (non-NaN) pixels — returning all -1 labels.")
        return labels

    logger.info(
        f"Running DBSCAN  eps={eps}  min_samples={min_samples}  "
        f"metric={metric}  valid_pixels={n_valid}/{len(features)}"
    )

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
    )
    db.fit(features.values[valid_mask])
    labels[valid_mask] = db.labels_

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = int((db.labels_ == -1).sum())
    logger.info(
        f"DBSCAN discovered {n_clusters} cluster(s); "
        f"{n_noise}/{n_valid} valid pixels classified as noise."
    )
    return labels


# ---------------------------------------------------------------------------
# Output dataset construction
# ---------------------------------------------------------------------------

def labels_to_dataset(
    ds: xr.Dataset,
    labels: np.ndarray,
    var: str = "Sv",
) -> xr.Dataset:
    """Reshape flat cluster labels back into the spatial grid of the
    original echogram and return a new :class:`xr.Dataset`.

    The output has the same ``ping_time`` and ``range_sample`` (or
    ``depth``, ``echo_range``) dimensions as the first channel of
    ``ds[var]``.  Instead of Sv values, it contains integer cluster
    labels stored under a single pseudo-channel.  Noise points are
    labelled ``-1``.

    The AcousticVariable descriptor for *var* is embedded in the global
    attributes (mirroring kmeans_core.labels_to_dataset and
    hdbscan_core.labels_to_dataset) so aa-report can render the
    variable's physical meaning identically across all three
    algorithms.
    """
    template = ds[var].isel(channel=0)
    shape = template.shape

    cluster_2d = labels.reshape(shape)

    dims = template.dims
    coords = {d: template.coords[d] for d in dims}

    desc = get_variable_descriptor(var)

    cluster_da = xr.DataArray(
        data=cluster_2d,
        dims=dims,
        coords=coords,
        name="cluster_map",
        attrs={
            "long_name": "DBSCAN cluster assignment",
            "description": (
                f"Integer cluster labels produced by DBSCAN clustering "
                f"of the unified (alpha, beta) feature matrix on multi-"
                f"frequency {desc.long_name} data.  -1 indicates noise."
            ),
            "units": "1",
        },
    )

    out_ds = cluster_da.to_dataset(name="cluster_map")
    out_ds.attrs["source_tool"] = "aa-dbscan"
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
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    channels: Optional[List[int]] = None,
    var: str = "Sv",
    n_jobs: Optional[int] = None,
    preset: Optional[str] = None,
) -> xr.Dataset:
    """End-to-end: build features -> run DBSCAN -> return labelled dataset.

    Mirrors :func:`KMeans.kmeans_core.cluster_dataset` and
    :func:`KMeans.hdbscan_core.cluster_dataset` so DBSCAN runs in the
    same feature space as KMeans / HDBSCAN for any given preset or
    weight choice.  This means a fingerprint produced from a DBSCAN
    run shares its phi-space with KMeans-derived codebooks at the
    same (alpha, beta).

    Parameters
    ----------
    ds : xr.Dataset
        Sv (or other variable) dataset.
    alpha : float, default 1.0
        Weight on the colour (centered) component.
    beta : float, default 1.0
        Weight on the loudness (mean) component.
    eps : float, default 0.5
        Maximum neighbourhood radius in phi-space.  Note: eps is in
        phi-space units, so it must be rescaled if alpha or beta
        change (unlike the partition itself, which depends only on
        the ratio beta/alpha).
    min_samples : int, default 5
        Minimum points to form a dense region.
    metric : str, default "euclidean"
    channels : list of int or None
    var : str
    n_jobs : int or None
        Parallel jobs for distance computation.
    preset : str or None
        If given, overrides *alpha* and *beta* with PRESETS[preset].

    Returns
    -------
    xr.Dataset
        Dataset with ``cluster_map`` variable, plus the (alpha, beta)
        and DBSCAN-parameter metadata in the global attributes that
        aa-report consumes.
    """
    if preset is not None:
        alpha, beta = resolve_preset(preset)

    features = build_feature_matrix(
        ds, alpha=alpha, beta=beta, channel_indices=channels, var=var
    )
    labels = run_dbscan(
        features,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
    )
    out = labels_to_dataset(ds, labels, var=var)

    n_clusters = len(set(int(x) for x in labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    out.attrs["algorithm"] = "DBSCAN"
    out.attrs["alpha"] = float(alpha)
    out.attrs["beta"] = float(beta)
    out.attrs["beta_over_alpha"] = (
        float(beta) / float(alpha) if alpha > 0 else float("inf")
    )
    out.attrs["preset"] = preset if preset is not None else "custom"
    out.attrs["eps"] = float(eps)
    out.attrs["min_samples"] = int(min_samples)
    out.attrs["metric"] = str(metric)
    out.attrs["n_clusters_discovered"] = int(n_clusters)
    out.attrs["n_noise_pixels"] = n_noise
    out.attrs["channels_used"] = str(
        resolve_channel_indices(ds, channels, var=var)
    )
    out.attrs["feature_columns"] = ", ".join(features.columns)
    return out