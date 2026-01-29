# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from myptv.trackpy_bridge import (
    link_with_myptv_2d,
    smooth_myptv_tracks,
    trackpy_locate_df_to_myptv_blobs_buffer,
)


def _make_two_linear_tracks(n_frames: int = 11) -> pd.DataFrame:
    # Two particles, far apart, constant velocity, no ambiguity.
    rows = []
    for frame in range(n_frames):
        rows.append({"frame": frame, "x": 10.0 + 1.0 * frame, "y": 20.0 + 0.0 * frame})
        rows.append({"frame": frame, "x": 200.0 + 1.0 * frame, "y": 100.0 + 0.0 * frame})
    return pd.DataFrame(rows)


def test_locate_df_to_blobs_buffer_has_six_columns_and_frame_int():
    df_loc = _make_two_linear_tracks(n_frames=3)
    buf = trackpy_locate_df_to_myptv_blobs_buffer(df_loc)

    out = pd.read_csv(buf, sep="\t", header=None)
    assert out.shape[1] == 6
    # frame column index is 5
    assert pd.api.types.is_integer_dtype(out[5])


def test_link_with_myptv_2d_four_frames_tracks_two_particles():
    df_loc = _make_two_linear_tracks(n_frames=8)

    df_tracks = link_with_myptv_2d(
        df_loc,
        algorithm="four_frames",
        camera=None,
        z_particles=0.0,
        d_max=5.0,
        dv_max=5.0,
    )

    # Expect 2 linked trajectories spanning all frames.
    assert set(df_tracks.columns) >= {"frame", "particle", "x", "y", "z"}

    linked = df_tracks[df_tracks["particle"].notna()].copy()
    # every frame should have exactly 2 linked points
    counts = linked.groupby("frame")["particle"].count().to_dict()
    assert all(counts[f] == 2 for f in range(8))

    # and exactly two particle ids
    pids = sorted(linked["particle"].astype(int).unique().tolist())
    assert len(pids) == 2


def test_link_with_myptv_2d_multiframe_tracks_two_particles_and_interpolates_ok():
    df_loc = _make_two_linear_tracks(n_frames=11)

    df_tracks = link_with_myptv_2d(
        df_loc,
        algorithm="multiframe",
        camera=None,
        z_particles=0.0,
        max_dt=2,
        Ns=5,
        d_max=5.0,
        dv_max=5.0,
        NSR_th=1.0,  # allow very permissive pruning for this synthetic case
        interpolate=True,
    )

    linked = df_tracks[df_tracks["particle"].notna()].copy()
    assert linked.groupby("frame")["particle"].count().min() == 2
    assert linked.groupby("frame")["particle"].count().max() == 2

    pids = sorted(linked["particle"].astype(int).unique().tolist())
    assert len(pids) == 2


def test_smoothing_returns_velocity_columns_and_reasonable_vx():
    # Use multiframe to ensure we have full tracks and a stable schema.
    df_loc = _make_two_linear_tracks(n_frames=15)
    df_tracks = link_with_myptv_2d(
        df_loc,
        algorithm="multiframe",
        camera=None,
        z_particles=0.0,
        max_dt=2,
        Ns=5,
        d_max=5.0,
        dv_max=5.0,
        NSR_th=1.0,
        interpolate=True,
    )

    df_smooth = smooth_myptv_tracks(df_tracks, window=7, polyorder=2)

    assert set(df_smooth.columns) >= {
        "frame",
        "particle",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "ax",
        "ay",
        "az",
    }

    # For linear motion x = x0 + t, vx should be ~1 in the interior.
    # Smoothing drops edges; just check median vx is close to 1.
    vx_med = float(df_smooth["vx"].median())
    assert np.isfinite(vx_med)
    assert abs(vx_med - 1.0) < 0.2
