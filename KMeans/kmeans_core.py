"""
kmeans_core.py

Modular KMeans clustering operations for echosounder NetCDF data.

This module provides atomic functions for constructing pre-clustering
feature matrices and performing KMeans clustering on xarray Datasets
containing Sv (volume backscattering strength) data. It is designed
to be consumed by console tools (aa-kmeans) following the Unix
philosophy of small, composable programs.

Two clustering models are supported:

    direct (dir)
        Each pixel is represented by a vector of its Sv values across
        all user-selected channels. The raw multi-frequency Sv values
        are fed directly into KMeans.

    absolute_differences (abd)  [default]
        For every unique pair of user-selected channels, the absolute
        difference |Sv_A - Sv_B| is computed.  These pairwise
        difference columns become the feature matrix for KMeans.
        Two identical channels would produce a zero column, so only
        genuinely distinct frequency information contributes.
"""

import itertools
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Channel / frequency helpers
# ---------------------------------------------------------------------------

def list_channels(ds: xr.Dataset, var: str = "Sv") -> List[str]:
    """Return the list of channel coordinate values in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        An Sv-style dataset with a ``channel`` dimension.
    var : str
        The data variable to inspect (default ``"Sv"``).

    Returns
    -------
    list of str
        Channel coordinate values.
    """
    if "channel" not in ds[var].dims:
        raise ValueError(f"Variable '{var}' has no 'channel' dimension.")
    return [str(c) for c in ds["channel"].values]


def resolve_channel_indices(
    ds: xr.Dataset,
    channels: Optional[List[int]] = None,
    var: str = "Sv",
) -> List[int]:
    """Return validated integer indices into the channel dimension.

    If *channels* is ``None``, all channels are returned.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``channel`` dimension.
    channels : list of int or None
        0-based indices selected by the user.
    var : str
        Data variable to inspect.

    Returns
    -------
    list of int
    """
    n_channels = ds.sizes["channel"]
    if channels is None:
        return list(range(n_channels))
    for idx in channels:
        if idx < 0 or idx >= n_channels:
            raise IndexError(
                f"Channel index {idx} is out of range. "
                f"Dataset has {n_channels} channels (0-{n_channels - 1})."
            )
    return list(channels)


def _channel_label(ds: xr.Dataset, idx: int) -> str:
    """Best-effort human-readable label for a channel index."""
    raw = str(ds["channel"].values[idx])
    # Try to extract a numeric frequency from common EK80 patterns
    for prefix in ("GPT", "ES"):
        if prefix in raw:
            try:
                after = raw.split(prefix)[1]
                freq = after.split("kHz")[0].split("-")[0].strip()
                return f"{freq}kHz"
            except (IndexError, ValueError):
                pass
    return raw


# ---------------------------------------------------------------------------
# Feature-matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix_direct(
    ds: xr.Dataset,
    channel_indices: List[int],
    var: str = "Sv",
) -> pd.DataFrame:
    """Build the pre-clustering feature matrix using the **direct** model.

    Each column is the flattened Sv for one channel.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``Sv[channel, ping_time, range_sample]``.
    channel_indices : list of int
        Which channels to include.
    var : str
        Data variable name.

    Returns
    -------
    pd.DataFrame
        Columns are labelled with frequency/channel names; rows are pixels.
    """
    columns = {}
    for idx in channel_indices:
        label = _channel_label(ds, idx)
        arr = ds[var].isel(channel=idx).values.ravel()
        columns[label] = arr
    return pd.DataFrame(columns)


def build_feature_matrix_abd(
    ds: xr.Dataset,
    channel_indices: List[int],
    var: str = "Sv",
) -> pd.DataFrame:
    """Build the pre-clustering feature matrix using the **absolute_differences** model.

    For every unique pair (A, B) of channels, a column
    ``|Sv(A) - Sv(B)|`` is produced.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``Sv[channel, ping_time, range_sample]``.
    channel_indices : list of int
        Which channels to include.
    var : str
        Data variable name.

    Returns
    -------
    pd.DataFrame
    """
    pairs = list(itertools.combinations(channel_indices, 2))
    if not pairs:
        raise ValueError(
            "Absolute-differences model requires at least 2 channels. "
            f"Got {len(channel_indices)} channel(s)."
        )
    columns = {}
    for a, b in pairs:
        label_a = _channel_label(ds, a)
        label_b = _channel_label(ds, b)
        col_name = f"abs(Sv({label_a})-Sv({label_b}))"
        arr_a = ds[var].isel(channel=a).values.ravel()
        arr_b = ds[var].isel(channel=b).values.ravel()
        columns[col_name] = np.abs(arr_a - arr_b)
    return pd.DataFrame(columns)


def build_feature_matrix(
    ds: xr.Dataset,
    model: str = "abd",
    channel_indices: Optional[List[int]] = None,
    var: str = "Sv",
) -> pd.DataFrame:
    """Dispatch to the appropriate feature-matrix builder.

    Parameters
    ----------
    ds : xr.Dataset
    model : {"abd", "dir"}
    channel_indices : list of int or None
    var : str

    Returns
    -------
    pd.DataFrame
    """
    indices = resolve_channel_indices(ds, channel_indices, var=var)
    logger.info(
        f"Building feature matrix  model={model}  "
        f"channels={indices}  var={var}"
    )
    if model == "dir":
        return build_feature_matrix_direct(ds, indices, var=var)
    elif model == "abd":
        return build_feature_matrix_abd(ds, indices, var=var)
    else:
        raise ValueError(f"Unknown clustering model: '{model}'. Use 'dir' or 'abd'.")


# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------

def run_kmeans(
    features: pd.DataFrame,
    n_clusters: int = 3,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Run KMeans on a feature matrix and return integer cluster labels.

    NaN rows are assigned a label of ``-1``; all other rows receive
    a cluster id in ``[0, n_clusters)``.

    Parameters
    ----------
    features : pd.DataFrame
        The pre-clustering feature matrix (pixels × features).
    n_clusters : int
        Number of clusters (k).
    n_init : int
        Number of KMeans initialisations.
    max_iter : int
        Maximum iterations per run.
    random_state : int or None
        Seed for reproducibility.

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
        f"Running KMeans  k={n_clusters}  valid_pixels={n_valid}/{len(features)}"
    )

    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    km.fit(features.values[valid_mask])
    labels[valid_mask] = km.labels_

    logger.info(f"KMeans inertia={km.inertia_:.4f}")
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
    pseudo-channel called ``"cluster"``.

    Parameters
    ----------
    ds : xr.Dataset
        The original Sv dataset (used only for coordinate metadata).
    labels : np.ndarray
        1-D array of cluster labels produced by :func:`run_kmeans`.
    var : str
        Data variable name used to pull spatial dimensions.

    Returns
    -------
    xr.Dataset
        A lightweight dataset with variable ``cluster_map`` and the same
        spatial dimensions as a single-channel echogram slice.
    """
    # Use first channel as the spatial template
    template = ds[var].isel(channel=0)
    shape = template.shape  # (ping_time, range_sample/depth/echo_range)

    cluster_2d = labels.reshape(shape)

    # Build the output DataArray
    dims = template.dims
    coords = {d: template.coords[d] for d in dims}

    cluster_da = xr.DataArray(
        data=cluster_2d,
        dims=dims,
        coords=coords,
        name="cluster_map",
        attrs={
            "long_name": "KMeans cluster assignment",
            "description": (
                "Integer cluster labels produced by KMeans clustering "
                "of multi-frequency Sv data."
            ),
            "units": "1",
        },
    )

    out_ds = cluster_da.to_dataset(name="cluster_map")
    out_ds.attrs["source_tool"] = "aa-kmeans"
    out_ds.attrs["clustering_variable"] = var
    return out_ds


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def cluster_dataset(
    ds: xr.Dataset,
    model: str = "abd",
    n_clusters: int = 3,
    channels: Optional[List[int]] = None,
    var: str = "Sv",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: Optional[int] = None,
) -> xr.Dataset:
    """End-to-end: build features → run KMeans → return labelled dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Sv dataset.
    model : {"abd", "dir"}
        Clustering model.
    n_clusters : int
        Number of clusters.
    channels : list of int or None
        Channel indices to use (default: all).
    var : str
        Data variable.
    n_init, max_iter, random_state
        Forwarded to :func:`run_kmeans`.

    Returns
    -------
    xr.Dataset
        Dataset with ``cluster_map`` variable.
    """
    features = build_feature_matrix(ds, model=model, channel_indices=channels, var=var)
    labels = run_kmeans(
        features,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    out = labels_to_dataset(ds, labels, var=var)
    out.attrs["clustering_model"] = model
    out.attrs["n_clusters"] = n_clusters
    out.attrs["channels_used"] = str(
        resolve_channel_indices(ds, channels, var=var)
    )
    return out