"""
消融对比实验：三种控制方法对比
- Baseline (Reactive Lloyd): 纯反应式 Lloyd，在当前场上做 Voronoi + Lloyd
- Current Predictive (Hotspot-Chasing): PredictionDrivenController
- Improved (Region Coverage): PredictiveRegionCoverageController

运行方式:
    python simulations/ablation_experiment.py [--agents N] [--time T]
                                              [--preset P] [--seed S]
                                              [--no-animation]
"""
import argparse
import copy
import os
import sys
import time as wall_time
import warnings
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
for _p in [_ROOT, _SCRIPT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.environment.sensitivity_field import DynamicSensitivityField
from src.agents.uav import UAVSwarm
from src.coverage.voronoi import WeightedVoronoi
from src.coverage.lloyd_controller import LloydController
from src.coverage.coverage_controllers import PredictionDrivenController
from src.coverage.region_coverage_controller import PredictiveRegionCoverageController
from src.prediction.gp_predictor import SpatioTemporalGP

warnings.filterwarnings('ignore')


# ── constants ─────────────────────────────────────────────────────────────────
DOMAIN = (0, 100, 0, 100)
RESOLUTION = 50
METHOD_NAMES = ['Baseline\n(Reactive Lloyd)',
                'Current Predictive\n(Hotspot-Chasing)',
                'Improved\n(Region Coverage)']
METHOD_LABELS = ['Baseline', 'Predictive', 'Improved']
COLORS = ['#2196F3', '#FF9800', '#4CAF50']


# ══════════════════════════════════════════════════════════════════════════════
#  Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_coverage_cost(positions: np.ndarray,
                          field: np.ndarray,
                          voronoi: WeightedVoronoi) -> float:
    return voronoi.compute_coverage_cost(positions, field)


def compute_worst_cost(field: np.ndarray,
                       domain: tuple,
                       resolution: int) -> float:
    """Approximate H_worst: all agents at one corner."""
    x_grid = np.linspace(domain[0], domain[1], resolution)
    y_grid = np.linspace(domain[2], domain[3], resolution)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])

    # Worst case: agent at corner (0, 0)
    corner = np.array([[domain[0], domain[2]]])
    dist_sq = np.sum((grid_pts - corner) ** 2, axis=1)
    total_mass = np.sum(field.ravel()) * dx * dy
    diag_sq = (domain[1] - domain[0]) ** 2 + (domain[3] - domain[2]) ** 2
    return total_mass * diag_sq


def compute_overlap_ratio(positions: np.ndarray,
                          field: np.ndarray,
                          sensing_radius: float,
                          domain: tuple,
                          resolution: int,
                          density_threshold_ratio: float = 0.5) -> float:
    """
    Fraction of high-sensitivity grid points covered by ≥2 agents.
    """
    x_grid = np.linspace(domain[0], domain[1], resolution)
    y_grid = np.linspace(domain[2], domain[3], resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])
    field_flat = field.ravel()

    base = field_flat.min()
    peak = field_flat.max()
    threshold = base + density_threshold_ratio * (peak - base)
    high_mask = field_flat >= threshold
    if not np.any(high_mask):
        return 0.0

    high_pts = grid_pts[high_mask]

    # Count how many agents cover each high-sensitivity point
    coverage_count = np.zeros(len(high_pts), dtype=int)
    for pos in positions:
        dists = np.linalg.norm(high_pts - pos, axis=1)
        coverage_count += (dists <= sensing_radius).astype(int)

    overlap_points = np.sum(coverage_count >= 2)
    return overlap_points / len(high_pts)


def compute_mean_pairwise_distance(positions: np.ndarray) -> float:
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, j in combinations(range(n), 2):
        total += np.linalg.norm(positions[i] - positions[j])
        count += 1
    return total / count


def compute_min_hotspot_distance(positions: np.ndarray,
                                 hotspot_positions: np.ndarray) -> float:
    if len(hotspot_positions) == 0:
        return 0.0
    total = 0.0
    for hp in hotspot_positions:
        dists = np.linalg.norm(positions - hp, axis=1)
        total += dists.min()
    return total / len(hotspot_positions)


# ══════════════════════════════════════════════════════════════════════════════
#  GP prediction helper
# ══════════════════════════════════════════════════════════════════════════════

def build_gp_prediction(gp: SpatioTemporalGP,
                        domain: tuple,
                        resolution: int,
                        predict_time: float) -> np.ndarray:
    """Return predicted field grid (resolution × resolution)."""
    x_grid = np.linspace(domain[0], domain[1], resolution)
    y_grid = np.linspace(domain[2], domain[3], resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    test_pts = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(resolution ** 2, predict_time)])
    mean, _ = gp.predict(test_pts)
    mean = np.maximum(mean, 0.0)
    return mean.reshape(resolution, resolution)


# ══════════════════════════════════════════════════════════════════════════════
#  Single-method simulation runner
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(args):
    os.makedirs('output/ablation', exist_ok=True)

    # ── Shared environment ─────────────────────────────────────────────────
    field = DynamicSensitivityField(DOMAIN, resolution=RESOLUTION)
    field.add_preset_hotspots(args.preset, seed=args.seed)

    # Pre-generate field snapshots for all steps so all three simulations
    # experience the exact same field evolution
    n_steps = int(args.time / args.dt)
    print(f'Pre-generating {n_steps} field snapshots ...')
    field_snapshots = []   # list of (current_field, hotspot_positions)
    _field_gen = DynamicSensitivityField(DOMAIN, resolution=RESOLUTION)
    _field_gen.add_preset_hotspots(args.preset, seed=args.seed)
    for _ in range(n_steps):
        field_snapshots.append(
            (_field_gen.get_field_grid(), _field_gen.get_hotspot_positions().copy()))
        _field_gen.update(args.dt)

    voronoi = WeightedVoronoi(DOMAIN, resolution=RESOLUTION)

    # ── Controllers ────────────────────────────────────────────────────────
    lloyd_ctrl = LloydController(DOMAIN, gain=1.0, max_velocity=5.0, resolution=RESOLUTION)
    predictive_ctrl = PredictionDrivenController(DOMAIN, max_velocity=5.0)
    improved_ctrl = PredictiveRegionCoverageController(DOMAIN, resolution=RESOLUTION,
                                                       max_velocity=5.0)

    # ── GPs (one per GP-using method) ──────────────────────────────────────
    gp_predictive = SpatioTemporalGP(length_scale_space=15.0,
                                     length_scale_time=8.0,
                                     noise_variance=0.1)
    gp_improved = SpatioTemporalGP(length_scale_space=15.0,
                                   length_scale_time=8.0,
                                   noise_variance=0.1)

    # ── Swarms (same seed → same initial positions) ────────────────────────
    def make_swarm():
        np.random.seed(args.seed)
        return UAVSwarm(args.agents, DOMAIN)

    swarm_baseline = make_swarm()
    swarm_predictive = make_swarm()
    swarm_improved = make_swarm()

    # ── Run three methods over pre-generated field ─────────────────────────
    all_metrics = {}

    for method_key, label, swarm, ctrl, gp in [
        ('baseline',   'Baseline',   swarm_baseline,   lloyd_ctrl,      None),
        ('predictive', 'Predictive', swarm_predictive, predictive_ctrl, gp_predictive),
        ('improved',   'Improved',   swarm_improved,   improved_ctrl,   gp_improved),
    ]:
        print(f'\nRunning {label} ...')
        t0 = wall_time.time()

        use_gp = gp is not None
        n_agents = swarm.num_agents
        positions = swarm.get_positions().copy()
        prev_positions = positions.copy()
        dist_traveled = np.zeros(n_agents)
        gp_trained = False
        gp_update_interval = 5

        traj_snapshot = None

        m = {k: [] for k in ['coverage_cost', 'weighted_coverage', 'overlap_ratio',
                              'mean_pairwise_dist', 'total_dist_traveled',
                              'min_hotspot_dist', 'time']}

        for step in range(n_steps):
            t = step * args.dt
            current_field, hotspot_pos = field_snapshots[step]

            # GP sense + fit
            predicted_field = None
            prediction_weight = 0.0
            if use_gp:
                swarm.sense_all(
                    # Provide a wrapper that uses the snapshot field
                    _FieldWrapper(current_field, DOMAIN, RESOLUTION), t)

                if step % gp_update_interval == 0 and step > 0:
                    X, y = swarm.get_all_sensed_data(time_window=30.0, current_time=t)
                    if len(X) >= 10:
                        try:
                            gp.fit(X, y)
                            gp_trained = True
                        except Exception:
                            pass

                if gp_trained:
                    predict_t = t + args.dt * gp_update_interval
                    try:
                        predicted_field = build_gp_prediction(
                            gp, DOMAIN, RESOLUTION, predict_t)
                        prediction_weight = 0.6
                    except Exception:
                        predicted_field = None
                        prediction_weight = 0.0

            # Control
            positions = swarm.get_positions()
            if method_key == 'baseline':
                velocities, _ = ctrl.compute_control(positions, current_field)
            else:
                if predicted_field is not None:
                    velocities = ctrl.compute_control(
                        positions, current_field, predicted_field, prediction_weight)
                else:
                    velocities = ctrl.compute_control(
                        positions, current_field, None, 0.0)

            swarm.update_all(velocities, args.dt, use_velocity=True)
            positions = swarm.get_positions()

            dist_traveled += np.linalg.norm(positions - prev_positions, axis=1)
            prev_positions = positions.copy()

            # Trajectory snapshot at t≈60s
            if traj_snapshot is None and t >= 60.0 - args.dt * 0.6:
                traj_snapshot = positions.copy()

            # Metrics every 10 steps
            if step % 10 == 0:
                cost = compute_coverage_cost(positions, current_field, voronoi)
                worst = compute_worst_cost(current_field, DOMAIN, RESOLUTION)
                w_cov = float(np.clip(1.0 - cost / max(worst, 1e-10), 0, 1))
                overlap = compute_overlap_ratio(positions, current_field,
                                               15.0, DOMAIN, RESOLUTION)
                mpd = compute_mean_pairwise_distance(positions)
                mhd = compute_min_hotspot_distance(positions, hotspot_pos)

                m['coverage_cost'].append(cost)
                m['weighted_coverage'].append(w_cov)
                m['overlap_ratio'].append(overlap)
                m['mean_pairwise_dist'].append(mpd)
                m['total_dist_traveled'].append(float(np.sum(dist_traveled)))
                m['min_hotspot_dist'].append(mhd)
                m['time'].append(t)

        m['traj_snapshot'] = traj_snapshot if traj_snapshot is not None \
            else swarm.get_positions().copy()

        all_metrics[method_key] = m
        print(f'  Done in {wall_time.time() - t0:.1f}s')

    # ── Visualize ──────────────────────────────────────────────────────────
    _plot_curves(all_metrics, args)
    _plot_stats(all_metrics, args)
    _plot_trajectories(all_metrics, field_snapshots, args)
    _print_table(all_metrics, args)

    print('\n✓ Results saved to output/ablation/')


# ── Field wrapper for UAVSwarm.sense_all ─────────────────────────────────────

class _FieldWrapper:
    """Wraps a pre-computed field snapshot to satisfy the sense() interface."""

    def __init__(self, field_grid: np.ndarray, domain: tuple, resolution: int):
        x_min, x_max, y_min, y_max = domain
        self._x = np.linspace(x_min, x_max, resolution)
        self._y = np.linspace(y_min, y_max, resolution)
        self._field = field_grid  # (res, res)
        self._domain = domain
        self._res = resolution

    def get_density(self, positions: np.ndarray) -> np.ndarray:
        positions = np.atleast_2d(positions)
        x_min, x_max, y_min, y_max = self._domain
        # Bilinear interpolation via nearest-grid lookup
        xi = np.clip(
            np.round((positions[:, 0] - x_min) / (x_max - x_min) *
                     (self._res - 1)).astype(int), 0, self._res - 1)
        yi = np.clip(
            np.round((positions[:, 1] - y_min) / (y_max - y_min) *
                     (self._res - 1)).astype(int), 0, self._res - 1)
        return self._field[yi, xi]


# ── Plotting helpers ──────────────────────────────────────────────────────────

_METRIC_LABELS = {
    'coverage_cost':       ('Coverage Cost H', 'lower is better'),
    'weighted_coverage':   ('Weighted Coverage', 'higher is better'),
    'overlap_ratio':       ('Overlap Ratio', 'lower is better'),
    'mean_pairwise_dist':  ('Mean Pairwise Distance (m)', 'higher is better'),
    'total_dist_traveled': ('Total Distance Traveled (m)', ''),
    'min_hotspot_dist':    ('Min Hotspot Distance (m)', 'lower is better'),
}


def _plot_curves(all_metrics: dict, args):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    keys = list(_METRIC_LABELS.keys())
    for ax_idx, key in enumerate(keys):
        ax = axes[ax_idx]
        for mi, (method_key, label, color) in enumerate(
                zip(['baseline', 'predictive', 'improved'],
                    METHOD_LABELS, COLORS)):
            m = all_metrics[method_key]
            ax.plot(m['time'], m[key], color=color, label=label, linewidth=1.5)

        title, hint = _METRIC_LABELS[key]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        if hint:
            ax.set_title(f'{title}\n({hint})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Ablation Study — preset={args.preset}, agents={args.agents}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/ablation/ablation_curves.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  Saved: output/ablation/ablation_curves.png')


def _second_half_mean(m: dict, key: str, total_time: float) -> float:
    """Average metric value in second half of simulation (t > total_time/2)."""
    times = np.array(m['time'])
    vals = np.array(m[key])
    mask = times > total_time / 2
    if not np.any(mask):
        return float(np.mean(vals))
    return float(np.mean(vals[mask]))


def _plot_stats(all_metrics: dict, args):
    keys = list(_METRIC_LABELS.keys())
    n_keys = len(keys)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()

    x = np.arange(3)
    width = 0.6

    for ax_idx, key in enumerate(keys):
        ax = axes[ax_idx]
        vals = [_second_half_mean(all_metrics[mk], key, args.time)
                for mk in ['baseline', 'predictive', 'improved']]
        bars = ax.bar(x, vals, width, color=COLORS, alpha=0.8, edgecolor='black',
                      linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, fontsize=8)
        title, _ = _METRIC_LABELS[key]
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        # value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'Second-Half Averages (t > {args.time/2:.0f}s)\n'
                 f'preset={args.preset}, agents={args.agents}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/ablation/ablation_stats.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  Saved: output/ablation/ablation_stats.png')


def _plot_trajectories(all_metrics: dict, field_snapshots: list, args):
    # Use field snapshot closest to t=60s
    step_60 = min(int(60.0 / args.dt), len(field_snapshots) - 1)
    field_60, _ = field_snapshots[step_60]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x_min, x_max, y_min, y_max = DOMAIN

    for ax_idx, (method_key, label, color) in enumerate(
            zip(['baseline', 'predictive', 'improved'], METHOD_LABELS, COLORS)):
        ax = axes[ax_idx]
        m = all_metrics[method_key]
        pos = m['traj_snapshot']  # (N, 2) positions at t≈60s

        # Heat-map of field
        im = ax.imshow(field_60, origin='lower',
                       extent=[x_min, x_max, y_min, y_max],
                       cmap='hot', alpha=0.6, aspect='auto')

        # Hotspot positions (from snapshot)
        _, hp = field_snapshots[step_60]
        ax.scatter(hp[:, 0], hp[:, 1], marker='*', s=200,
                   color='yellow', edgecolors='black', linewidth=0.8, zorder=5,
                   label='Hotspots')

        # Agent positions
        ax.scatter(pos[:, 0], pos[:, 1], s=80, color=color,
                   edgecolors='black', linewidth=0.8, zorder=6, label='Agents')

        # Sensing radius circles
        for p in pos:
            circle = plt.Circle(p, 15.0, color=color, fill=False,
                                 alpha=0.3, linewidth=1.0)
            ax.add_patch(circle)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'{label}\n(t ≈ 60s)', fontsize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend(fontsize=7, loc='upper right')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Agent Positions at t≈60s — preset={args.preset}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/ablation/ablation_trajectories.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  Saved: output/ablation/ablation_trajectories.png')


def _print_table(all_metrics: dict, args):
    print('\n' + '=' * 80)
    print(f'{"Metric":<28} {"Baseline":>12} {"Predictive":>12} {"Improved":>12}  '
          f'{"Improv. vs Base":>16}')
    print('-' * 80)

    better_is_lower = {'coverage_cost', 'overlap_ratio', 'min_hotspot_dist'}

    for key in _METRIC_LABELS:
        base_val = _second_half_mean(all_metrics['baseline'], key, args.time)
        pred_val = _second_half_mean(all_metrics['predictive'], key, args.time)
        impr_val = _second_half_mean(all_metrics['improved'], key, args.time)

        if abs(base_val) > 1e-10:
            if key in better_is_lower:
                pct = 100.0 * (base_val - impr_val) / abs(base_val)
            else:
                pct = 100.0 * (impr_val - base_val) / abs(base_val)
        else:
            pct = 0.0

        label, _ = _METRIC_LABELS[key]
        print(f'{label:<28} {base_val:>12.3f} {pred_val:>12.3f} {impr_val:>12.3f}  '
              f'{pct:>+14.1f}%')

    print('=' * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Ablation experiment: Baseline vs Predictive vs Improved controller')
    p.add_argument('--agents', type=int, default=5, help='Number of UAV agents (default 5)')
    p.add_argument('--time', type=float, default=80.0, help='Total simulation time in s (default 80)')
    p.add_argument('--preset', type=str, default='circular',
                   choices=['static', 'linear', 'circular', 'mixed'],
                   help='Hotspot preset (default circular)')
    p.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    p.add_argument('--no-animation', action='store_true',
                   help='Skip animation (currently animations are not generated by default)')
    p.add_argument('--dt', type=float, default=0.1, help='Time step in s (default 0.1)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'Ablation experiment: agents={args.agents}, time={args.time}s, '
          f'preset={args.preset}, seed={args.seed}')
    run_ablation(args)
