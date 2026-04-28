"""
kmeans_core.py

Modular KMeans clustering operations for echosounder NetCDF data.

This module implements a unified feature-construction framework for
multifrequency Sv data, parametrized by two non-negative weights
(alpha, beta).  For an N-frequency pixel x = (Sv_1, ..., Sv_N), the
record fed to KMeans is

    phi(x) = ( alpha * c_1, alpha * c_2, ..., alpha * c_N,  beta * Sv_mean )

where

    Sv_mean = (1/N) sum_i Sv_i           ("loudness", common-mode dB)
    c_i     = Sv_i - Sv_mean             ("colour",   relative response)

The three previously named recipes are corners of this construction:

    preset       alpha   beta   notes
    -----------  ------  -----  -------------------------------------------
    "direct"        1      1    Information-equivalent to raw Sv vector
    "contrast"      1      0    Colour-only; replaces the old "abd" recipe
    "loudness"      0      1    Mean-only (Sv_mean is the only feature)

A "hybrid" anywhere in alpha > 0, beta > 0 is also supported, and the
recommended workflow is to sweep beta with alpha fixed at 1 (only the
ratio matters to KMeans).

NOTE on the change from the old "abd" model:
    The previous implementation produced N*(N-1)/2 pairwise columns of
    |Sv_A - Sv_B|.  The new "contrast" preset produces N columns of
    c_i = Sv_i - mean(Sv).  These two representations encode the same
    inter-frequency information (up to a choice of reference) but yield
    different distance geometries in feature space.  The centered
    representation is the one used in the proposal and is the canonical
    feature for the unified framework.
"""

from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Named (alpha, beta) presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Tuple[float, float]] = {
    # Canonical names from the unified framework
    "direct":   (1.0, 1.0),  # info-equivalent to raw Sv vector
    "contrast": (1.0, 0.0),  # colour-only  (a.k.a. relative-response)
    "loudness": (0.0, 1.0),  # mean-only

    # Back-compat aliases for the old --model values
    "dir": (1.0, 1.0),
    "abd": (1.0, 0.0),       # NB: now centered c_i, not pairwise |A-B|
    "mean": (0.0, 1.0),
}


def resolve_preset(preset: str) -> Tuple[float, float]:
    """Look up (alpha, beta) for a named preset.  Case-insensitive."""
    key = preset.lower()
    if key not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'.  Available: {sorted(set(PRESETS))}"
        )
    return PRESETS[key]


# ---------------------------------------------------------------------------
# Channel / frequency helpers
# ---------------------------------------------------------------------------

def list_channels(ds: xr.Dataset, var: str = "Sv") -> List[str]:
    """Return the list of channel coordinate values in the dataset."""
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
# Core decomposition: split loudness from colour
# ---------------------------------------------------------------------------

def split_loudness_colour(
    ds: xr.Dataset,
    channel_indices: List[int],
    var: str = "Sv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose multifrequency Sv into (loudness, colour) components.

    For each pixel x = (Sv_1, ..., Sv_N) across the selected channels:

        Sv_mean = (1/N) * sum_i Sv_i           shape (n_pixels,)
        c_i     = Sv_i - Sv_mean               shape (n_pixels, N)

    NaN propagation: if *any* selected channel is NaN at a given pixel,
    both Sv_mean and every c_i at that pixel become NaN, so downstream
    KMeans masking removes the pixel cleanly.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``Sv[channel, ping_time, range_sample]``.
    channel_indices : list of int
        Channels to include in the decomposition.
    var : str
        Data variable name (default "Sv").

    Returns
    -------
    sv_mean : np.ndarray, shape (n_pixels,)
        Per-pixel common-mode component (loudness).
    centered : np.ndarray, shape (n_pixels, N)
        Per-pixel, per-channel deviation from the mean (colour).
    """
    arrays = [ds[var].isel(channel=i).values.ravel() for i in channel_indices]
    stacked = np.stack(arrays, axis=1)            # (n_pixels, N)
    sv_mean = stacked.mean(axis=1)                # NaN propagates
    centered = stacked - sv_mean[:, None]         # (n_pixels, N)
    return sv_mean, centered


# ---------------------------------------------------------------------------
# Unified feature-matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    ds: xr.Dataset,
    alpha: float = 1.0,
    beta: float = 1.0,
    channel_indices: Optional[List[int]] = None,
    var: str = "Sv",
) -> pd.DataFrame:
    """Build the unified KMeans feature matrix.

    The record per pixel is

        phi(x) = ( alpha * c_1, ..., alpha * c_N,  beta * Sv_mean )

    Columns whose weight is exactly zero are omitted (so contrast-only
    drops the mean column, mean-only drops the colour columns).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``Sv[channel, ping_time, range_sample]``.
    alpha : float, default 1.0
        Weight on the colour (centered) component.  Must be >= 0.
    beta : float, default 1.0
        Weight on the loudness (mean) component.  Must be >= 0.
    channel_indices : list of int or None
        Channel indices to include (default: all).
    var : str
        Data variable name.

    Returns
    -------
    pd.DataFrame
        Up to N+1 columns: N weighted-centered + 1 weighted-mean.
    """
    if alpha < 0 or beta < 0:
        raise ValueError(
            f"alpha and beta must be non-negative; got alpha={alpha}, beta={beta}"
        )
    if alpha == 0 and beta == 0:
        raise ValueError(
            "alpha and beta cannot both be zero — feature matrix would be empty."
        )

    indices = resolve_channel_indices(ds, channel_indices, var=var)
    if alpha > 0 and len(indices) < 2:
        raise ValueError(
            "Colour component (alpha > 0) requires at least 2 channels; "
            f"got {len(indices)}."
        )

    sv_mean, centered = split_loudness_colour(ds, indices, var=var)

    columns: Dict[str, np.ndarray] = {}
    if alpha > 0:
        for j, idx in enumerate(indices):
            label = _channel_label(ds, idx)
            columns[f"alpha*c({label})"] = alpha * centered[:, j]
    if beta > 0:
        columns["beta*Sv_mean"] = beta * sv_mean

    logger.info(
        f"Built feature matrix  alpha={alpha}  beta={beta}  "
        f"channels={indices}  n_columns={len(columns)}  var={var}"
    )
    return pd.DataFrame(columns)


def build_feature_matrix_from_preset(
    ds: xr.Dataset,
    preset: str = "direct",
    channel_indices: Optional[List[int]] = None,
    var: str = "Sv",
) -> pd.DataFrame:
    """Convenience wrapper: build feature matrix using a named (alpha, beta) preset."""
    alpha, beta = resolve_preset(preset)
    logger.info(f"Preset '{preset}' -> alpha={alpha}, beta={beta}")
    return build_feature_matrix(
        ds, alpha=alpha, beta=beta, channel_indices=channel_indices, var=var
    )


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

    NaN rows are assigned label ``-1``; valid rows get a label in
    ``[0, n_clusters)``.
    """
    labels = np.full(len(features), -1, dtype=int)
    valid_mask = features.notna().all(axis=1).values
    n_valid = int(valid_mask.sum())

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
    """Reshape flat cluster labels back into the spatial grid of the
    original echogram and return a new :class:`xr.Dataset`.
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
            "long_name": "KMeans cluster assignment",
            "description": (
                "Integer cluster labels produced by KMeans clustering "
                "of the unified (alpha, beta) feature matrix on multi-"
                "frequency Sv data."
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
    alpha: float = 1.0,
    beta: float = 1.0,
    n_clusters: int = 3,
    channels: Optional[List[int]] = None,
    var: str = "Sv",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: Optional[int] = None,
    preset: Optional[str] = None,
) -> xr.Dataset:
    """End-to-end: build features -> run KMeans -> return labelled dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Sv dataset.
    alpha : float, default 1.0
        Weight on colour (centered) component.
    beta : float, default 1.0
        Weight on loudness (mean) component.
    n_clusters : int, default 3
    channels : list of int or None
    var : str
    n_init, max_iter, random_state
        Forwarded to :func:`run_kmeans`.
    preset : str or None
        If given, overrides *alpha* and *beta* with PRESETS[preset].

    Returns
    -------
    xr.Dataset
        Dataset with ``cluster_map`` variable, plus (alpha, beta) and
        related metadata in the global attributes.
    """
    if preset is not None:
        alpha, beta = resolve_preset(preset)

    features = build_feature_matrix(
        ds, alpha=alpha, beta=beta, channel_indices=channels, var=var
    )
    labels = run_kmeans(
        features,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    out = labels_to_dataset(ds, labels, var=var)
    out.attrs["alpha"] = float(alpha)
    out.attrs["beta"] = float(beta)
    out.attrs["beta_over_alpha"] = (
        float(beta) / float(alpha) if alpha > 0 else float("inf")
    )
    out.attrs["preset"] = preset if preset is not None else "custom"
    out.attrs["n_clusters"] = int(n_clusters)
    out.attrs["channels_used"] = str(
        resolve_channel_indices(ds, channels, var=var)
    )
    out.attrs["feature_columns"] = ", ".join(features.columns)
    return out