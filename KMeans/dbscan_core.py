"""
dbscan_core.py

Modular DBSCAN clustering operations for echosounder NetCDF data.

This module provides the DBSCAN clustering step and output construction
for xarray Datasets containing Sv (volume backscattering strength) data.
Feature-matrix construction (direct / absolute-differences models) is
shared with kmeans_core and imported from there.

Unlike KMeans, DBSCAN does not require the number of clusters to be
specified in advance.  Instead it takes two parameters:

    eps
        The maximum distance between two samples for them to be
        considered neighbours.

    min_samples
        The minimum number of points required to form a dense region
        (core point threshold).

Points that do not belong to any cluster are labelled ``-1`` (noise).

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
    list_channels,
    resolve_channel_indices,
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
        The pre-clustering feature matrix (pixels × features).
    eps : float
        Maximum neighbourhood radius.
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
    n_valid = valid_mask.sum()

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
    n_noise = (db.labels_ == -1).sum()
    logger.info(
        f"DBSCAN discovered {n_clusters} cluster(s), "
        f"{n_noise} noise point(s) out of {n_valid} valid pixels"
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
    """Reshape flat cluster labels back into the spatial grid of the original
    echogram and return a new :class:`xr.Dataset`.

    The output has the same ``ping_time`` and ``range_sample`` (or ``depth``,
    ``echo_range``) dimensions as the first channel of *ds[var]*.  Instead
    of Sv values, it contains integer cluster labels stored under a single
    pseudo-channel.  Noise points are labelled ``-1``.

    Parameters
    ----------
    ds : xr.Dataset
        The original Sv dataset (used only for coordinate metadata).
    labels : np.ndarray
        1-D array of cluster labels produced by :func:`run_dbscan`.
    var : str
        Data variable name used to pull spatial dimensions.

    Returns
    -------
    xr.Dataset
        A lightweight dataset with variable ``cluster_map`` and the same
        spatial dimensions as a single-channel echogram slice.
    """
    template = ds[var].isel(channel=0)
    shape = template.shape

    cluster_2d = labels.reshape(shape)

    dims = template.dims
    coords = {d: template.coords[d] for d in dims}

    cluster_da = xr.DataArray(
        data=cluster_2d,
        dims=dims,
        coords=coords,
        name="cluster_map",
        attrs={
            "long_name": "DBSCAN cluster assignment",
            "description": (
                "Integer cluster labels produced by DBSCAN clustering "
                "of multi-frequency Sv data.  -1 indicates noise."
            ),
            "units": "1",
        },
    )

    out_ds = cluster_da.to_dataset(name="cluster_map")
    out_ds.attrs["source_tool"] = "aa-dbscan"
    out_ds.attrs["clustering_variable"] = var
    return out_ds


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def cluster_dataset(
    ds: xr.Dataset,
    model: str = "abd",
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    channels: Optional[List[int]] = None,
    var: str = "Sv",
    n_jobs: Optional[int] = None,
) -> xr.Dataset:
    """End-to-end: build features → run DBSCAN → return labelled dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Sv dataset.
    model : {"abd", "dir"}
        Feature-matrix construction model.
    eps : float
        Maximum neighbourhood radius.
    min_samples : int
        Minimum points to form a dense region.
    metric : str
        Distance metric.
    channels : list of int or None
        Channel indices to use (default: all).
    var : str
        Data variable.
    n_jobs : int or None
        Parallel jobs for distance computation.

    Returns
    -------
    xr.Dataset
        Dataset with ``cluster_map`` variable.
    """
    features = build_feature_matrix(ds, model=model, channel_indices=channels, var=var)
    labels = run_dbscan(
        features,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
    )
    out = labels_to_dataset(ds, labels, var=var)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    out.attrs["clustering_model"] = model
    out.attrs["algorithm"] = "DBSCAN"
    out.attrs["eps"] = eps
    out.attrs["min_samples"] = min_samples
    out.attrs["metric"] = metric
    out.attrs["n_clusters_discovered"] = n_clusters
    out.attrs["channels_used"] = str(
        resolve_channel_indices(ds, channels, var=var)
    )
    return out