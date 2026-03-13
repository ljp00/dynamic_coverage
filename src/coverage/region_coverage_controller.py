"""
预测区域协同覆盖控制器
PredictiveRegionCoverageController

核心改进：把"追逐预测热点中心点"改为"根据预测热点区域进行协同覆盖部署"
"""
import numpy as np
from typing import Tuple, Optional, List
from .voronoi import WeightedVoronoi, VoronoiCell


class RegionDescriptor:
    """
    预测区域描述符
    从预测场中提取高敏感度区域的几何属性
    """

    def __init__(self, threshold_ratio: float = 0.6):
        """
        Args:
            threshold_ratio: 阈值比例，threshold = base + ratio * (max - base)
        """
        self.threshold_ratio = threshold_ratio

    def extract(self, field: np.ndarray,
                grid_points: np.ndarray) -> dict:
        """
        从敏感度场中提取高敏感度区域的几何属性

        Args:
            field: 敏感度场 shape (resolution, resolution)
            grid_points: 网格点坐标 shape (resolution^2, 2)

        Returns:
            dict containing:
                'mask': bool array of high-sensitivity grid points
                'centroid': weighted centroid (2,)
                'cov': weighted 2×2 covariance matrix
                'eigenvalues': sorted descending eigenvalues of cov
                'eigenvectors': corresponding eigenvectors (2, 2), rows are vecs
                'effective_radius': scalar effective radius
                'area_ratio': fraction of domain that is high-sensitivity
                'total_mass': total density mass in high-sensitivity region
        """
        field_flat = field.ravel()
        base_density = np.min(field_flat)
        max_density = np.max(field_flat)

        threshold = base_density + self.threshold_ratio * (max_density - base_density)
        mask = field_flat >= threshold

        if not np.any(mask):
            # Fallback: take top 10% as region
            threshold = np.percentile(field_flat, 90)
            mask = field_flat >= threshold

        region_points = grid_points[mask]
        region_density = field_flat[mask]

        total_mass = np.sum(region_density)
        if total_mass < 1e-10:
            centroid = grid_points.mean(axis=0)
            cov = np.eye(2)
            eigenvalues = np.array([1.0, 1.0])
            eigenvectors = np.eye(2)
            effective_radius = 10.0
        else:
            # Weighted centroid
            centroid = np.sum(region_points * region_density[:, None], axis=0) / total_mass

            # Weighted covariance
            diff = region_points - centroid
            weights = region_density / total_mass
            cov = (diff * weights[:, None]).T @ diff
            cov += 1e-6 * np.eye(2)  # regularization

            # Eigendecomposition for ellipse axes
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx].T  # rows are eigenvectors

            # Effective radius: sqrt of mean eigenvalue
            effective_radius = float(np.sqrt(np.mean(eigenvalues)))

        area_ratio = np.sum(mask) / len(field_flat)

        return {
            'mask': mask,
            'centroid': centroid,
            'cov': cov,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'effective_radius': effective_radius,
            'area_ratio': area_ratio,
            'total_mass': total_mass,
        }


class PredictiveRegionCoverageController:
    """
    预测区域协同覆盖控制器

    接口兼容 PredictionDrivenController：
        compute_control(positions, current_field, predicted_field, prediction_weight)
        → velocities shape (N, 2)

    控制律：
        u_i = w_lloyd   * vel_lloyd       (基于预测场的Voronoi质心方向)
            + w_spread  * vel_spread      (区域自适应展开)
            + w_overlap * vel_overlap     (重叠惩罚)
            + w_boundary * vel_boundary   (边界力)
    """

    def __init__(self,
                 domain: Tuple[float, float, float, float] = (0, 100, 0, 100),
                 resolution: int = 50,
                 max_velocity: float = 5.0,
                 sensing_radius: float = 15.0,
                 # Base weights
                 w_lloyd: float = 1.0,
                 w_spread: float = 0.8,
                 w_overlap: float = 0.6,
                 w_boundary: float = 1.2,
                 # Region extraction
                 threshold_ratio: float = 0.6,
                 # Boundary margin and strength
                 boundary_margin: float = 8.0,
                 boundary_strength: float = 3.0):
        """
        Args:
            domain: (x_min, x_max, y_min, y_max)
            resolution: grid resolution for Voronoi
            max_velocity: maximum agent speed
            sensing_radius: effective sensing radius per agent (for overlap calculation)
            w_lloyd: weight for Lloyd (Voronoi centroid) term
            w_spread: weight for spread deployment term
            w_overlap: weight for overlap penalty term
            w_boundary: weight for boundary repulsion term
            threshold_ratio: ratio used in region threshold extraction
            boundary_margin: distance from boundary where repulsion starts
            boundary_strength: strength of boundary repulsion
        """
        self.domain = domain
        self.resolution = resolution
        self.max_velocity = max_velocity
        self.sensing_radius = sensing_radius
        self.w_lloyd = w_lloyd
        self.w_spread = w_spread
        self.w_overlap = w_overlap
        self.w_boundary = w_boundary
        self.boundary_margin = boundary_margin
        self.boundary_strength = boundary_strength

        self.voronoi = WeightedVoronoi(domain, resolution=resolution)
        self.region_descriptor = RegionDescriptor(threshold_ratio=threshold_ratio)

        # Build grid for field operations
        x_grid = np.linspace(domain[0], domain[1], resolution)
        y_grid = np.linspace(domain[2], domain[3], resolution)
        self.dx = x_grid[1] - x_grid[0]
        self.dy = y_grid[1] - y_grid[0]
        X, Y = np.meshgrid(x_grid, y_grid)
        self.grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_control(self,
                        positions: np.ndarray,
                        current_field: np.ndarray,
                        predicted_field: Optional[np.ndarray] = None,
                        prediction_weight: float = 0.0) -> np.ndarray:
        """
        计算控制速度

        Args:
            positions: agent positions shape (N, 2)
            current_field: current sensitivity field shape (50, 50)
            predicted_field: predicted future field shape (50, 50), optional
            prediction_weight: weight for predicted field [0, 1]

        Returns:
            velocities: shape (N, 2)
        """
        # 1. Blend fields
        if predicted_field is not None and prediction_weight > 0:
            blended_field = ((1 - prediction_weight) * current_field
                             + prediction_weight * predicted_field)
        else:
            blended_field = current_field

        # 2. Extract region descriptor from blended field
        region = self.region_descriptor.extract(blended_field, self.grid_points)

        # 3. Compute Voronoi cells on blended field
        cells = self.voronoi.compute_voronoi(positions, blended_field)

        # 4. Adaptive weight adjustment based on region area
        # Large high-sensitivity area → increase spread weight
        area_ratio = region['area_ratio']
        # Scale spread weight: at area_ratio=0.5, multiply by 1.5; at 0.1, no change
        spread_scale = 1.0 + max(0.0, area_ratio - 0.1) * 1.5
        w_spread_eff = self.w_spread * spread_scale

        # 5. Compute individual control components
        vel_lloyd = self._compute_lloyd_velocities(positions, cells)
        vel_spread = self._compute_spread_velocities(positions, cells, blended_field, region)
        vel_overlap = self._compute_overlap_velocities(positions, cells, blended_field)
        vel_boundary = self._compute_boundary_velocities(positions)

        # 6. Synthesize
        velocities = (self.w_lloyd * vel_lloyd
                      + w_spread_eff * vel_spread
                      + self.w_overlap * vel_overlap
                      + self.w_boundary * vel_boundary)

        # 7. Clamp velocity
        velocities = self._limit_velocity(velocities)
        return velocities

    # ------------------------------------------------------------------
    # 1b. Lloyd (Voronoi centroid) term
    # ------------------------------------------------------------------

    def _compute_lloyd_velocities(self,
                                  positions: np.ndarray,
                                  cells: List[VoronoiCell]) -> np.ndarray:
        """
        Standard weighted Lloyd: move each agent toward its Voronoi cell centroid
        (computed on the blended/predicted field).
        """
        velocities = np.zeros_like(positions)
        for i, cell in enumerate(cells):
            velocities[i] = cell.centroid - positions[i]
        return velocities

    # ------------------------------------------------------------------
    # 1b. Spread / shape-adaptive deployment term
    # ------------------------------------------------------------------

    def _compute_assignments(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute Voronoi assignments for all grid points.

        Returns:
            assignments: shape (G,) — index of nearest agent for each grid point
        """
        diff = self.grid_points[:, None, :] - positions[None, :, :]  # (G, N, 2)
        dists = np.linalg.norm(diff, axis=2)                          # (G, N)
        return np.argmin(dists, axis=1)                               # (G,)

    def _compute_spread_velocities(self,
                                   positions: np.ndarray,
                                   cells: List[VoronoiCell],
                                   field: np.ndarray,
                                   region: dict) -> np.ndarray:
        """
        Encourage agents to spread along the principal axes of the hot region.

        For each agent push it toward the grid point in its Voronoi cell that has
        the best combination of high density AND large distance from the agent.
        The push magnitude is scaled by the ellipse axes: displacement along the
        major axis is amplified by sqrt(λ_major/λ_minor).
        """
        velocities = np.zeros_like(positions)
        field_flat = field.ravel()

        eigenvalues = region['eigenvalues']   # [λ_major, λ_minor]
        eigenvectors = region['eigenvectors']  # rows: [v_major, v_minor]

        if eigenvalues[1] > 1e-6:
            anisotropy = float(np.sqrt(max(eigenvalues[0], 1e-6) / eigenvalues[1]))
            anisotropy = float(np.clip(anisotropy, 1.0, 3.0))
        else:
            anisotropy = 1.0

        major_axis = eigenvectors[0]  # unit vector along major axis

        # Compute assignments once for all agents
        assignments = self._compute_assignments(positions)

        for i, cell in enumerate(cells):
            if cell.mass < 1e-10:
                continue

            cell_mask = assignments == i
            if not np.any(cell_mask):
                continue

            cell_density = field_flat[cell_mask]
            cell_pts = self.grid_points[cell_mask]

            distances_to_agent = np.linalg.norm(cell_pts - positions[i], axis=1)
            composite = cell_density * distances_to_agent
            best_idx = np.argmax(composite)
            target = cell_pts[best_idx]

            direction = target - positions[i]
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            direction = direction / dist

            proj_major = np.dot(direction, major_axis)
            scale = 1.0 + (anisotropy - 1.0) * abs(proj_major)
            velocities[i] = scale * direction

        return velocities

    # ------------------------------------------------------------------
    # 1c. Overlap penalty term
    # ------------------------------------------------------------------

    def _compute_overlap_velocities(self,
                                    positions: np.ndarray,
                                    cells: List[VoronoiCell],
                                    field: np.ndarray) -> np.ndarray:
        """
        Compute repulsive velocities based on pairwise sensing-area overlap.

        Overlap ratio between agent i and j:
            ov_ij = max(0, 1 - dist_ij / (r_i + r_j))
        where r_i is proportional to the square root of agent i's Voronoi cell area.

        Repulsion direction: along the Voronoi boundary normal (i.e., away from j,
        but projected perpendicular to the inter-agent vector so agents stay on their
        respective sides of the Voronoi boundary).

        Only applied when the overlapping region has significant density.
        """
        velocities = np.zeros_like(positions)
        n = len(positions)
        field_flat = field.ravel()

        # Effective radii from Voronoi cell areas
        radii = np.array([max(self.sensing_radius * 0.5,
                              np.sqrt(cells[i].area / np.pi))
                          for i in range(n)])

        for i in range(n):
            for j in range(i + 1, n):
                d_vec = positions[i] - positions[j]
                dist = np.linalg.norm(d_vec)
                if dist < 1e-6:
                    # Degenerate: push in random direction
                    repulsion = np.random.randn(2)
                    repulsion /= np.linalg.norm(repulsion)
                    velocities[i] += repulsion
                    velocities[j] -= repulsion
                    continue

                # Overlap ratio
                combined_radius = radii[i] + radii[j]
                if dist >= combined_radius:
                    continue

                overlap_ratio = 1.0 - dist / combined_radius  # in (0, 1]

                # Estimate density in the overlap region (midpoint approximation)
                midpoint = 0.5 * (positions[i] + positions[j])
                mid_dists = np.linalg.norm(self.grid_points - midpoint, axis=1)
                overlap_region_mask = mid_dists < 0.5 * combined_radius
                if np.any(overlap_region_mask):
                    overlap_density = np.mean(field_flat[overlap_region_mask])
                else:
                    overlap_density = 1.0

                # Repulsion strength: stronger if overlap is large AND density is high
                strength = overlap_ratio * overlap_density

                # Direction: push i away from j along the connection vector
                direction = d_vec / dist

                velocities[i] += strength * direction
                velocities[j] -= strength * direction

        return velocities

    # ------------------------------------------------------------------
    # Boundary term
    # ------------------------------------------------------------------

    def _compute_boundary_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Repel agents from domain boundaries."""
        x_min, x_max, y_min, y_max = self.domain
        m = self.boundary_margin
        s = self.boundary_strength
        velocities = np.zeros_like(positions)

        for i, pos in enumerate(positions):
            if pos[0] < x_min + m:
                velocities[i, 0] += s * (x_min + m - pos[0])
            if pos[0] > x_max - m:
                velocities[i, 0] -= s * (pos[0] - (x_max - m))
            if pos[1] < y_min + m:
                velocities[i, 1] += s * (y_min + m - pos[1])
            if pos[1] > y_max - m:
                velocities[i, 1] -= s * (pos[1] - (y_max - m))

        return velocities

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _limit_velocity(self, velocities: np.ndarray) -> np.ndarray:
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-10)
        scale = np.minimum(1.0, self.max_velocity / speeds)
        return velocities * scale
