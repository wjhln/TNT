"""Evaluate and visualize TNT forecasted trajectories against ground truth."""

import argparse
import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate/visualize TNT trajectory predictions.")
    parser.add_argument("--forecast", default="", type=str,
                        help="Path to forecasted_trajectories.pkl (Dict[int, List[np.ndarray]])")
    parser.add_argument("--gt", default="", type=str,
                        help="Path to gt_trajectories.pkl (Dict[int, np.ndarray])")
    parser.add_argument("--data_root", default="/root/autodl-tmp/dataset/interm_data", type=str,
                        help="Root dir of the intermediate dataset (contains val_intermediate/raw/)")
    parser.add_argument("--split", default="val", type=str,
                        help="Dataset split: val or test")
    parser.add_argument("--obs_len", default=20, type=int,
                        help="Number of observed timesteps")
    parser.add_argument("--horizon", default=30, type=int,
                        help="Forecast horizon (future timesteps)")
    parser.add_argument("--miss_threshold", default=2.0, type=float,
                        help="Miss rate distance threshold (meters)")
    parser.add_argument("--max_n_guesses", default=6, type=int,
                        help="Number of predicted trajectories to evaluate")
    parser.add_argument("--metrics", action="store_true",
                        help="Compute ADE/FDE/MR metrics")
    parser.add_argument("--viz", action="store_true",
                        help="Visualize predictions")
    parser.add_argument("--viz_seq_id", default="", type=str,
                        help="Comma-separated sequence ids to visualize, e.g. '1,10,100'. "
                             "Leave empty to visualize all.")
    parser.add_argument("--save_dir", default="", type=str,
                        help="Directory to save visualization plots. If empty, plots are shown interactively.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Raw feature loading helpers
# ---------------------------------------------------------------------------

def _raw_pkl_path(data_root: str, split: str, seq_id: int) -> str:
    return os.path.join(data_root, f"{split}_intermediate", "raw", f"features_{seq_id}.pkl")


def load_raw_features(data_root: str, split: str, seq_id: int) -> Optional[pd.DataFrame]:
    path = _raw_pkl_path(data_root, split, seq_id)
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def get_obs_traj(raw_df: pd.DataFrame, obs_len: int = 20) -> np.ndarray:
    """Return the observed trajectory for the target agent (index 0) in world coordinates."""
    trajs = raw_df["trajs"].values[0]
    steps = raw_df["steps"].values[0]
    target_traj = np.array(trajs[0])
    target_steps = np.array(steps[0])
    obs_mask = target_steps < obs_len
    return target_traj[obs_mask]           # (obs_len, 2)


def get_other_agents_obs(raw_df: pd.DataFrame, obs_len: int = 20) -> List[np.ndarray]:
    """Return observed trajectories for all agents except the target (index 0)."""
    trajs = raw_df["trajs"].values[0]
    steps = raw_df["steps"].values[0]
    others = []
    for traj, step in zip(trajs[1:], steps[1:]):
        traj_arr = np.array(traj)
        step_arr = np.array(step)
        obs_mask = step_arr < obs_len
        if obs_mask.sum() >= 2:           # need at least 2 points to draw a line
            others.append(traj_arr[obs_mask])
    return others


def get_lane_graph(raw_df: pd.DataFrame) -> List[np.ndarray]:
    """Return lane polylines in world coordinates, grouped by lane id.

    graph['ctrs'] and graph['lane_idcs'] are in local coordinates; we convert
    each lane's ordered centerpoints back to world coordinates.
    """
    graph = raw_df["graph"].values[0]
    ctrs_local = graph["ctrs"]           # (N, 2) local
    lane_idcs = graph["lane_idcs"]       # (N,)
    orig = raw_df["orig"].values[0]      # (2,)
    rot = raw_df["rot"].values[0]        # (2, 2)
    inv_rot = np.linalg.inv(rot)

    lanes = []
    for lid in np.unique(lane_idcs):
        mask = lane_idcs == lid
        pts_local = ctrs_local[mask]                              # (k, 2)
        pts_world = np.matmul(inv_rot, pts_local.T).T + orig      # (k, 2)
        lanes.append(pts_world)
    return lanes                         # List of (k, 2) arrays


def get_city(raw_df: pd.DataFrame) -> str:
    return str(raw_df["city"].values[0])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_n_guesses: int,
    horizon: int,
    miss_threshold: float,
):
    metric_results = get_displacement_errors_and_miss_rate(
        forecasted_trajectories,
        gt_trajectories,
        max_n_guesses,
        horizon,
        miss_threshold,
    )
    print("\n[Metrics]")
    for k, v in metric_results.items():
        print(f"  {k}: {v:.4f}")
    return metric_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Palette: one color per predicted trajectory (cycles if k > len)
_PRED_COLORS = [
    "#d33e4c", "#f0821d", "#e8c720", "#34a853", "#4285f4", "#9c27b0",
]

PAD_M = 10       # metres of padding around trajectories
DPI   = 200      # output resolution


def _traj_bbox(trajs: List[np.ndarray], pad: float = PAD_M):
    """Compute axis limits around a collection of trajectories."""
    all_pts = np.concatenate(trajs, axis=0)
    xmin, ymin = all_pts.min(axis=0) - pad
    xmax, ymax = all_pts.max(axis=0) + pad
    return xmin, xmax, ymin, ymax


def viz_single(
    seq_id: int,
    obs_traj: np.ndarray,
    gt_traj: np.ndarray,
    pred_trajs: List[np.ndarray],
    lanes: Optional[List[np.ndarray]] = None,
    other_agents: Optional[List[np.ndarray]] = None,
    city: str = "",
    save_path: Optional[str] = None,
):
    """Visualize one sequence: lane polylines + other agents + observed + GT + predictions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    # -- compute crop window first so we can filter lanes --
    xmin, xmax, ymin, ymax = _traj_bbox(
        [obs_traj, gt_traj] + list(pred_trajs)
    )

    # -- lane polylines (only those that fall inside crop area) --
    if lanes:
        for pts in lanes:
            # keep lane if any point is inside the crop window (with some slack)
            in_view = (
                (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
            )
            if not in_view.any():
                continue
            ax.plot(pts[:, 0], pts[:, 1],
                    color="#c8c8c8", linewidth=1.5, zorder=1, solid_capstyle="round")

    # -- other agents (observed portion only) --
    if other_agents:
        for ag in other_agents:
            ax.plot(ag[:, 0], ag[:, 1],
                    color="#e08a00", linewidth=1.5, zorder=2,
                    solid_capstyle="round", alpha=0.75)
            ax.plot(ag[-1, 0], ag[-1, 1],
                    "o", color="#e08a00", markersize=5, zorder=3, alpha=0.75)

    # -- observed trajectory --
    ax.plot(obs_traj[:, 0], obs_traj[:, 1],
            color="#2b4590", linewidth=2.5, zorder=3,
            solid_capstyle="round", label="Observed")
    ax.plot(obs_traj[0, 0], obs_traj[0, 1],
            "o", color="#2b4590", markersize=8, zorder=5)   # start
    ax.plot(obs_traj[-1, 0], obs_traj[-1, 1],
            "o", color="#2b4590", markersize=10, zorder=5)  # end / agent pos

    # -- predicted trajectories --
    for i, pred in enumerate(pred_trajs):
        c = _PRED_COLORS[i % len(_PRED_COLORS)]
        label = "Predicted" if i == 0 else None
        ax.plot(pred[:, 0], pred[:, 1],
                color=c, linewidth=2, linestyle="--", alpha=0.85,
                zorder=2, solid_capstyle="round", label=label)
        ax.plot(pred[-1, 0], pred[-1, 1],
                "*", color=c, markersize=11, zorder=6,
                markeredgecolor="white", markeredgewidth=0.5)

    # -- ground truth --
    ax.plot(gt_traj[:, 0], gt_traj[:, 1],
            color="#1a9850", linewidth=2.5, zorder=4,
            solid_capstyle="round", label="Ground Truth")
    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1],
            "*", color="#1a9850", markersize=14, zorder=7,
            markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.85,
              edgecolor="#cccccc", fancybox=False)

    plt.tight_layout(pad=0.5)

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def viz_predictions_helper(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    data_root: str,
    split: str,
    obs_len: int,
    viz_seq_ids: Optional[List[int]],
    save_dir: str,
):
    seq_ids = list(gt_trajectories.keys()) if viz_seq_ids is None else viz_seq_ids

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for seq_id in seq_ids:
        if seq_id not in gt_trajectories or seq_id not in forecasted_trajectories:
            print(f"[Warning] seq_id {seq_id} not found in predictions/GT, skipping.")
            continue

        gt_traj = gt_trajectories[seq_id]              # (horizon, 2)
        pred_trajs = forecasted_trajectories[seq_id]   # List[(horizon, 2)]

        # Try to load raw features for obs trajectory and lane graph
        raw_df = load_raw_features(data_root, split, seq_id)
        if raw_df is not None:
            obs_traj = get_obs_traj(raw_df, obs_len)
            other_agents = get_other_agents_obs(raw_df, obs_len)
            lanes = get_lane_graph(raw_df)
            city = get_city(raw_df)
        else:
            print(f"[Warning] Raw features for seq {seq_id} not found at "
                  f"{_raw_pkl_path(data_root, split, seq_id)}")
            obs_traj = np.expand_dims(gt_traj[0], 0)
            other_agents = None
            lanes = None
            city = ""

        save_path = os.path.join(save_dir, f"seq_{seq_id}.png") if save_dir else None
        viz_single(seq_id, obs_traj, gt_traj, pred_trajs,
                   lanes=lanes, other_agents=other_agents,
                   city=city, save_path=save_path)

        if save_path:
            print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()

    if not args.forecast or not args.gt:
        raise ValueError("Both --forecast and --gt pkl paths are required.")

    with open(args.forecast, "rb") as f:
        forecasted_trajectories: Dict[int, List[np.ndarray]] = pickle.load(f)

    with open(args.gt, "rb") as f:
        gt_trajectories: Dict[int, np.ndarray] = pickle.load(f)

    print(f"[Info] Loaded {len(forecasted_trajectories)} forecasted sequences "
          f"and {len(gt_trajectories)} GT sequences.")

    # ---------- Metrics ----------
    if args.metrics:
        compute_metrics(
            forecasted_trajectories,
            gt_trajectories,
            args.max_n_guesses,
            args.horizon,
            args.miss_threshold,
        )

    # ---------- Visualization ----------
    if args.viz:
        viz_seq_ids = None
        if args.viz_seq_id:
            viz_seq_ids = [int(s.strip()) for s in args.viz_seq_id.split(",") if s.strip()]

        viz_predictions_helper(
            forecasted_trajectories,
            gt_trajectories,
            data_root=args.data_root,
            split=args.split,
            obs_len=args.obs_len,
            viz_seq_ids=viz_seq_ids,
            save_dir=args.save_dir,
        )
