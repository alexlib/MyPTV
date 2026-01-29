"""Bridge utilities between Trackpy DataFrames and MyPTV tracking/smoothing.

This module is intentionally "trackpy-light": it does not import trackpy.
It only operates on pandas DataFrames in the conventional trackpy schema
(`frame`, `x`, `y`, and `particle` for linked results).

Typical usage (image-space tracking):

    import trackpy as tp
    import pandas as pd
    from myptv.trackpy_bridge import (
        link_with_myptv_2d,
        smooth_myptv_tracks,
    )

    # df_loc has at least: frame, x, y
    df_loc = ...  # output of tp.locate over many frames (concatenated)

    df_tracks = link_with_myptv_2d(
        df_loc,
        algorithm="multiframe",
        max_dt=2,
        Ns=5,
        d_max=15.0,
        dv_max=10.0,
    )

    df_smooth = smooth_myptv_tracks(df_tracks, window=7, polyorder=2)

Notes
- MyPTV expects time steps of 1 frame. If your `frame` column has gaps,
  prefer using the multiframe tracker with interpolation, or fill gaps
  before smoothing.
- MyPTV smoothing trims edges for long trajectories (window/2+1 samples at
  each side).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from myptv.tracking_2D_mod import track_2D, track_2D_multiframe
from myptv.traj_smoothing_mod import smooth_trajectories


Algorithm = Literal["four_frames", "multiframe"]


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_mean_flow(mean_flow: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
    """Return a 3-vector for MyPTV trackers.

    In 2D usage, you likely want (Ux, Uy, 0). If you pass a scalar, it will be
    treated as (Ux, Uy, 0) = (scalar, scalar, 0).
    """
    if isinstance(mean_flow, (int, float, np.floating)):
        return np.array([float(mean_flow), float(mean_flow), 0.0], dtype=float)

    arr = np.asarray(mean_flow, dtype=float).reshape(-1)
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0], dtype=float)
    if arr.size == 3:
        return arr.astype(float)

    raise ValueError("mean_flow must be a scalar, (Ux,Uy), or (Ux,Uy,Uz)")


def trackpy_locate_df_to_myptv_blobs_buffer(
    df_loc: pd.DataFrame,
    *,
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
    size_x_col: Optional[str] = None,
    size_y_col: Optional[str] = None,
    area_col: Optional[str] = None,
) -> StringIO:
    """Convert a Trackpy locate DataFrame to MyPTV blob TSV (in-memory).

    MyPTV's blob format is a TSV with 6 columns:
      eta, zeta, size_x, size_y, area, frame

    Only eta/zeta/frame are required by MyPTV's 2D tracking; the size/area
    fields are kept for compatibility.
    """
    _require_columns(df_loc, [frame_col, x_col, y_col])

    df = df_loc.copy()
    df = df[[x_col, y_col, frame_col]].rename(columns={x_col: "eta", y_col: "zeta", frame_col: "frame"})

    if size_x_col and size_x_col in df_loc.columns:
        df["size_x"] = df_loc[size_x_col].astype(float)
    else:
        df["size_x"] = 0.0

    if size_y_col and size_y_col in df_loc.columns:
        df["size_y"] = df_loc[size_y_col].astype(float)
    else:
        df["size_y"] = 0.0

    if area_col and area_col in df_loc.columns:
        df["area"] = df_loc[area_col].astype(float)
    else:
        # trackpy often has "mass"; callers can pass area_col="mass".
        df["area"] = 0.0

    # MyPTV expects frame in column index 5 and groups by it.
    df = df[["eta", "zeta", "size_x", "size_y", "area", "frame"]]

    # Ensure frames are integers where possible.
    df["frame"] = df["frame"].astype(int)

    buf = StringIO()
    df.to_csv(buf, sep="\t", header=False, index=False, float_format="%.6f")
    buf.seek(0)
    return buf


def _connected_particles_to_trackpy_df(
    particles: Sequence[np.ndarray],
    *,
    id_is_one_based: bool,
) -> pd.DataFrame:
    # particle rows are MyPTV-internal arrays like:
    # [traj_id, x, y, z, ... , frame]
    if len(particles) == 0:
        return pd.DataFrame(columns=["frame", "particle", "x", "y", "z"])

    arr = np.asarray(particles, dtype=float)
    traj_id = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    z = arr[:, 3]
    frame = arr[:, -1].astype(int)

    # Map -1 (unlinked) to NA
    particle_ids = traj_id.astype(int)
    linked = particle_ids != -1

    out = pd.DataFrame({
        "frame": frame,
        "particle": pd.array([pd.NA] * len(particle_ids), dtype="Int64"),
        "x": x,
        "y": y,
        "z": z,
    })

    if id_is_one_based:
        out.loc[linked, "particle"] = (particle_ids[linked] - 1).astype(int)
    else:
        out.loc[linked, "particle"] = particle_ids[linked].astype(int)

    # Trackpy convention: sort by frame then particle.
    out = out.sort_values(["frame", "particle"], kind="mergesort").reset_index(drop=True)
    return out


def link_with_myptv_2d(
    df_loc: pd.DataFrame,
    *,
    algorithm: Algorithm = "multiframe",
    camera=None,
    z_particles: float = 0.0,
    mean_flow: Union[float, Sequence[float], np.ndarray] = 0.0,
    d_max: float = 1e10,
    dv_max: float = 1e10,
    reverse_eta_zeta: bool = False,
    # multiframe params
    max_dt: int = 2,
    Ns: int = 5,
    NSR_th: float = 0.25,
    interpolate: bool = True,
    # trackpy columns
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Run MyPTV 2D tracking on Trackpy detections.

    Returns a DataFrame compatible with trackpy linked output:
      columns: frame, particle, x, y, z

    If `camera` is None, tracking is performed in image space (pixels).
    If `camera` is provided, detections are projected to lab-space at
    constant `z_particles` before tracking.
    """
    _require_columns(df_loc, [frame_col, x_col, y_col])

    U = _normalize_mean_flow(mean_flow)
    blob_buf = trackpy_locate_df_to_myptv_blobs_buffer(
        df_loc,
        frame_col=frame_col,
        x_col=x_col,
        y_col=y_col,
    )

    if algorithm == "four_frames":
        t = track_2D(
            camera,
            blob_buf,
            z_particles,
            mean_flow=U,
            d_max=d_max,
            dv_max=dv_max,
            reverse_eta_zeta=reverse_eta_zeta,
        )
        t.blobs_to_particles()
        t.track_all_frames()
        particles = t.return_connected_particles()
        # four-frame assigns ids starting at 0
        return _connected_particles_to_trackpy_df(particles, id_is_one_based=False)

    if algorithm == "multiframe":
        t = track_2D_multiframe(
            camera,
            blob_buf,
            z_particles,
            max_dt,
            Ns,
            mean_flow=U,
            d_max=d_max,
            dv_max=dv_max,
            NSR_th=NSR_th,
            reverse_eta_zeta=reverse_eta_zeta,
        )
        t.blobs_to_particles()
        # Build trajectories forward/backward; this populates t.trajs
        t.track_frames()
        if interpolate:
            t.interpolate_trajs()

        # Collect connected particles from trajectories only.
        particles: list[np.ndarray] = []
        for tr in t.trajs:
            # Save format expects id in col 0; assign trajectory ids (1-based) like save_results
            tr2 = np.array(tr, dtype=float, copy=True)
            particles.extend(list(tr2))

        # multiframe save_results uses 1-based ids, but the in-memory trajectories still
        # have whatever id was in the particles (often -1). Assign stable ids now.
        # Assign ids per trajectory in list order.
        for tidx, tr in enumerate(t.trajs, start=1):
            tr[:, 0] = tidx

        particles = [p for tr in t.trajs for p in tr]
        return _connected_particles_to_trackpy_df(particles, id_is_one_based=True)

    raise ValueError("algorithm must be 'four_frames' or 'multiframe'")


def smooth_myptv_tracks(
    df_tracks: pd.DataFrame,
    *,
    window: int,
    polyorder: int,
    repetitions: int = 1,
    min_traj_length: int = 4,
) -> pd.DataFrame:
    """Apply MyPTV polynomial smoothing to a trackpy-style tracks DataFrame.

    Input `df_tracks` should contain at least: frame, particle, x, y.
    If z is missing, it is treated as 0.

    Returns a DataFrame with columns:
      frame, particle, x, y, z, vx, vy, vz, ax, ay, az

    Note: for long trajectories MyPTV drops edge samples (window/2+1 on each side).
    """
    _require_columns(df_tracks, ["frame", "particle", "x", "y"])

    df = df_tracks.copy()
    if "z" not in df.columns:
        df["z"] = 0.0

    # Keep linked points only.
    df = df[df["particle"].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["frame", "particle", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"])

    # Build MyPTV traj_list: [traj_id, x, y, z, frame]
    df["particle"] = df["particle"].astype("Int64")
    traj_arr = np.column_stack([
        df["particle"].astype(int).to_numpy(),
        df["x"].to_numpy(float),
        df["y"].to_numpy(float),
        df["z"].to_numpy(float),
        df["frame"].to_numpy(int),
    ])

    sm = smooth_trajectories(
        traj_arr,
        window=window,
        polyorder=polyorder,
        repetitions=repetitions,
        min_traj_length=min_traj_length,
    )
    sm.smooth()

    out = pd.DataFrame(
        sm.smoothed_trajs,
        columns=["particle", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "frame"],
    )

    out["frame"] = out["frame"].astype(int)
    out["particle"] = out["particle"].astype(int)

    # Trackpy-style ordering
    out = out.sort_values(["frame", "particle"], kind="mergesort").reset_index(drop=True)
    return out
