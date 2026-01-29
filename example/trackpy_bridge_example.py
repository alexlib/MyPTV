"""Example: Trackpy detections -> MyPTV 2D tracking -> smoothing.

This mirrors the usage shown in `myptv/trackpy_bridge.py`.

Run (from repo root):
    `.venv/bin/python example/trackpy_bridge_example.py`

Optional: if you have trackpy installed, you can replace the synthetic
`df_loc` with actual `trackpy.locate()` output and concatenate across frames.
"""

from __future__ import annotations

import pandas as pd

from myptv.trackpy_bridge import link_with_myptv_2d, smooth_myptv_tracks


def make_synthetic_locate_df(n_frames: int = 20) -> pd.DataFrame:
    rows = []
    for frame in range(n_frames):
        # Two particles moving at constant velocity
        rows.append({"frame": frame, "x": 10.0 + 1.0 * frame, "y": 20.0})
        rows.append({"frame": frame, "x": 200.0 + 0.5 * frame, "y": 100.0})
    return pd.DataFrame(rows)


def main() -> None:
    df_loc = make_synthetic_locate_df()

    df_tracks = link_with_myptv_2d(
        df_loc,
        algorithm="multiframe",
        max_dt=2,
        Ns=5,
        d_max=10.0,
        dv_max=10.0,
        NSR_th=1.0,
        interpolate=True,
        camera=None,  # image-space tracking (pixels)
        z_particles=0.0,
    )

    df_smooth = smooth_myptv_tracks(df_tracks, window=7, polyorder=2)

    print("Tracks head:\n", df_tracks.head(10))
    print("\nSmoothed head:\n", df_smooth.head(10))


if __name__ == "__main__":
    main()
