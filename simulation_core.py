"""
simulation_core.py - Flood simulation engine using height-difference water flow.

Features:
- Rainfall-driven flooding over terrain
- Diffusion-based water flow between cells (scaled by timestep and resolution)
- Dam/barrier support by elevation modification (per-cell heights)
- Edge drainage and infiltration for realistic water balance
- Mass-balance tracking and optional convergence-based early stopping
- Pluggable backend architecture for different solvers

Units:
- Rainfall intensity: mm/hour
- Rainfall duration: hours
- Timestep: hours
- Diffusion coefficient: m²/s (physical units, scaled internally)
- Infiltration/evaporation: mm/hour
- Edge drainage: fraction per step (0-1)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Protocol
from abc import ABC, abstractmethod
from terrain_loader import TerrainData


@dataclass
class SimulationParams:
    """
    Parameters controlling the flood simulation.
    
    Attributes:
        rainfall_intensity: Rainfall rate in mm/hour (typical: 10-100 mm/hr)
        rainfall_duration: How long rain falls in hours
        timestep: Simulation timestep in hours (smaller = more accurate but slower)
        diffusion_coef: Base flow coefficient in m²/s. Controls how fast water spreads.
                       Higher values = faster flow. Typical range: 0.1 - 10.0
        flow_threshold: Minimum water depth (m) required for flow to occur
        max_flow_fraction: Maximum fraction of cell's water that can move per step (stability)
        evaporation_rate: Water loss to atmosphere in mm/hour (typical: 0-5 mm/hr)
        infiltration_rate: Water absorption into soil in mm/hour 
                          (sandy: 25-50, loam: 10-25, clay: 1-5 mm/hr)
        edge_drainage_rate: Fraction of water removed at domain edges per step (0-1)
                           Simulates water leaving the DEM to downstream areas
        enable_edge_drainage: Whether to enable edge drainage
        convergence_threshold: Stop early if max_depth changes less than this (m) for N steps
        convergence_steps: Number of stable steps before early stopping (0 = disabled)
    """
    rainfall_intensity: float = 10.0        # mm/hour
    rainfall_duration: float = 1.0          # hours
    timestep: float = 0.1                   # hours
    diffusion_coef: float = 1.0             # m²/s (physical units)
    flow_threshold: float = 0.001           # minimum water depth for flow (m)
    max_flow_fraction: float = 0.25         # max fraction of water that can move per step
    evaporation_rate: float = 0.0           # mm/hour
    infiltration_rate: float = 5.0          # mm/hour (non-zero default for realism)
    edge_drainage_rate: float = 0.1         # fraction of edge water removed per step
    enable_edge_drainage: bool = True       # whether to drain at boundaries
    convergence_threshold: float = 0.001    # m - threshold for early stopping
    convergence_steps: int = 10             # steps of stability before stopping (0 = disabled)


@dataclass
class MassBalance:
    """Tracks water mass balance for diagnostics."""
    total_rainfall: float = 0.0         # m³ - cumulative rainfall added
    total_infiltration: float = 0.0     # m³ - cumulative water infiltrated into soil
    total_evaporation: float = 0.0      # m³ - cumulative water evaporated
    total_edge_drainage: float = 0.0    # m³ - cumulative water drained at edges
    current_volume: float = 0.0         # m³ - current water volume in domain
    
    @property
    def expected_volume(self) -> float:
        """Expected volume based on inputs minus outputs."""
        return (self.total_rainfall - self.total_infiltration - 
                self.total_evaporation - self.total_edge_drainage)
    
    @property
    def balance_error(self) -> float:
        """Absolute difference between expected and actual volume."""
        return abs(self.current_volume - self.expected_volume)
    
    @property
    def relative_error(self) -> float:
        """Relative mass balance error (0-1)."""
        if self.expected_volume > 0:
            return self.balance_error / self.expected_volume
        return 0.0 if self.current_volume < 1e-6 else 1.0


@dataclass
class SimulationState:
    """Current state of the flood simulation."""
    terrain: np.ndarray           # Base terrain elevation (m)
    water_depth: np.ndarray       # Water depth at each cell (m)
    time: float = 0.0             # Current simulation time (hours)
    step: int = 0                 # Current timestep number
    mass_balance: MassBalance = field(default_factory=MassBalance)
    
    @property
    def water_surface(self) -> np.ndarray:
        """Water surface elevation = terrain + water depth."""
        return self.terrain + self.water_depth
    
    @property
    def flooded_mask(self) -> np.ndarray:
        """Boolean mask of flooded cells."""
        return self.water_depth > 0.001  # 1mm threshold


@dataclass
class SimulationResult:
    """Complete simulation result with all timesteps."""
    params: SimulationParams
    terrain: TerrainData
    depth_history: List[np.ndarray]
    time_history: List[float]
    stats_history: List[dict]
    final_mass_balance: MassBalance = None
    stopped_early: bool = False
    stop_reason: str = ""
    
    @property
    def final_depth(self) -> np.ndarray:
        return self.depth_history[-1]
    
    @property
    def max_depth(self) -> np.ndarray:
        """Maximum depth reached at each cell across all timesteps."""
        return np.maximum.reduce(self.depth_history)


class BaseFloodModel(ABC):
    """
    Abstract base class for flood simulation backends.
    
    Implement this interface to add new solver backends (e.g., LISFLOOD-FP, ANUGA).
    """
    
    @abstractmethod
    def run(
        self,
        duration: Optional[float] = None,
        max_steps: int = 1000,
        record_interval: int = 1,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None
    ) -> SimulationResult:
        """Run the simulation and return results."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset simulation to initial state."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the solver name for display."""
        pass


class FloodSimulator(BaseFloodModel):
    """
    Flood simulation engine using a diffusion-based water flow model.
    
    This is the "Fast (NumPy)" backend - a simplified but efficient solver
    suitable for interactive exploration and rapid iteration.
    
    The simulation works by:
    1. Adding rainfall to all cells (during rainfall duration)
    2. Computing water flow between neighboring cells based on water surface difference
    3. Moving water from higher to lower cells (scaled by timestep and grid resolution)
    4. Applying infiltration, evaporation, and edge drainage
    5. Repeating for multiple timesteps with mass balance tracking
    
    Physical scaling:
    - Flow rate is proportional to: diffusion_coef * timestep / cell_area
    - This ensures consistent behavior across different DEM resolutions and timesteps
    """
    
    def __init__(self, terrain: TerrainData, params: Optional[SimulationParams] = None):
        """
        Initialize the simulator with terrain data.
        
        Args:
            terrain: TerrainData object with elevation data
            params: Simulation parameters (uses defaults if None)
        """
        self.terrain = terrain
        self.params = params or SimulationParams()
        
        # Pre-compute cell dimensions
        self.cell_area = self._calculate_cell_area()  # m²
        self.cell_size = np.sqrt(self.cell_area)  # approximate cell size in m
        
        # Pre-compute effective diffusion factor for current params
        self._update_diffusion_factor()
        
        # Initialize state
        self.state = SimulationState(
            terrain=terrain.elevation.copy(),
            water_depth=np.zeros_like(terrain.elevation, dtype=np.float32)
        )
        
        # Track dams/barriers with per-cell heights
        self.dam_mask = np.zeros_like(terrain.elevation, dtype=bool)
        self.dam_heights = np.zeros_like(terrain.elevation, dtype=np.float32)  # Per-cell dam heights
        self.original_terrain = terrain.elevation.copy()
        
        # Edge mask for drainage
        self._edge_mask = self._create_edge_mask()
        
        # Convergence tracking
        self._recent_max_depths = []
    
    @property
    def name(self) -> str:
        return "Fast (NumPy)"
    
    def _calculate_cell_area(self) -> float:
        """Calculate cell area in m², accounting for geographic coordinates."""
        import math
        
        bounds = self.terrain.bounds
        crs = self.terrain.crs
        
        # Check if coordinates are in degrees (lat/lon)
        if 'EPSG:4326' in crs or (abs(bounds[0]) <= 180 and abs(bounds[2]) <= 180):
            lat_center = (bounds[1] + bounds[3]) / 2
            
            # Approximate meters per degree at this latitude
            m_per_deg_lat = 111320
            m_per_deg_lon = 111320 * math.cos(math.radians(lat_center))
            
            cell_width_m = self.terrain.resolution[0] * m_per_deg_lon
            cell_height_m = self.terrain.resolution[1] * m_per_deg_lat
            
            return cell_width_m * cell_height_m
        else:
            # Projected coordinates - assume meters
            return self.terrain.resolution[0] * self.terrain.resolution[1]
    
    def _update_diffusion_factor(self):
        """
        Calculate effective diffusion factor based on physical parameters.
        
        The diffusion equation discretization gives:
            effective_k = D * dt / dx²
        
        Where:
            D = diffusion coefficient (m²/s)
            dt = timestep (converted to seconds)
            dx = cell size (m)
        
        For stability, effective_k should be < 0.25 for 2D diffusion.
        """
        dt_seconds = self.params.timestep * 3600  # hours to seconds
        dx_squared = self.cell_area  # cell_size² ≈ cell_area
        
        # Calculate physical diffusion factor
        raw_factor = self.params.diffusion_coef * dt_seconds / dx_squared
        
        # Clamp for numerical stability (4-neighbor scheme requires k < 0.25)
        self._effective_diffusion = min(raw_factor, 0.24)
        
        # Warn if clamped significantly
        if raw_factor > 0.25:
            import warnings
            warnings.warn(
                f"Diffusion factor {raw_factor:.3f} exceeds stability limit. "
                f"Clamped to {self._effective_diffusion:.3f}. "
                f"Consider reducing diffusion_coef or timestep."
            )
    
    def _create_edge_mask(self) -> np.ndarray:
        """Create a boolean mask for edge cells."""
        mask = np.zeros_like(self.state.terrain, dtype=bool)
        mask[0, :] = True   # top row
        mask[-1, :] = True  # bottom row
        mask[:, 0] = True   # left column
        mask[:, -1] = True  # right column
        return mask
    
    def add_dam(
        self, 
        mask: np.ndarray, 
        height: float = 10.0
    ) -> None:
        """
        Add a dam/barrier by increasing terrain elevation.
        
        Args:
            mask: Boolean mask where True indicates dam cells
            height: Height to add to dam cells (meters)
        """
        self.dam_mask |= mask
        self.dam_heights[mask] = height  # Store per-cell height
        self.state.terrain[mask] = self.original_terrain[mask] + height
    
    def remove_dam(self, mask: Optional[np.ndarray] = None) -> None:
        """
        Remove dam/barrier by restoring original terrain.
        
        Args:
            mask: Boolean mask of dam cells to remove. If None, remove all dams.
        """
        if mask is None:
            mask = self.dam_mask
        
        self.state.terrain[mask] = self.original_terrain[mask]
        self.dam_mask[mask] = False
        self.dam_heights[mask] = 0.0
    
    def apply_rainfall(self, intensity: Optional[float] = None) -> float:
        """
        Apply rainfall to the domain.
        
        Args:
            intensity: Rainfall intensity in mm/hour. Uses params if None.
        
        Returns:
            Total volume of water added (m³)
        """
        if intensity is None:
            intensity = self.params.rainfall_intensity
        
        # Convert mm/hour to meters for this timestep
        depth_added = (intensity / 1000.0) * self.params.timestep
        
        # Apply rainfall (skip NaN areas)
        valid_mask = ~np.isnan(self.state.terrain)
        self.state.water_depth[valid_mask] += depth_added
        
        # Calculate total volume
        volume = depth_added * np.sum(valid_mask) * self.cell_area
        self.state.mass_balance.total_rainfall += volume
        
        return volume
    
    def apply_infiltration(self) -> float:
        """
        Apply soil infiltration (water absorption into ground).
        
        Returns:
            Total volume of water removed (m³)
        """
        if self.params.infiltration_rate <= 0:
            return 0.0
        
        # Convert mm/hour to meters for this timestep
        infil_depth = (self.params.infiltration_rate / 1000.0) * self.params.timestep
        
        # Only infiltrate where there's water
        infiltrated = np.minimum(self.state.water_depth, infil_depth)
        self.state.water_depth -= infiltrated
        
        # Calculate volume removed
        volume = float(np.sum(infiltrated) * self.cell_area)
        self.state.mass_balance.total_infiltration += volume
        
        return volume
    
    def apply_evaporation(self) -> float:
        """
        Apply evaporation (water loss to atmosphere).
        
        Returns:
            Total volume of water removed (m³)
        """
        if self.params.evaporation_rate <= 0:
            return 0.0
        
        # Convert mm/hour to meters for this timestep
        evap_depth = (self.params.evaporation_rate / 1000.0) * self.params.timestep
        
        # Only evaporate where there's water
        evaporated = np.minimum(self.state.water_depth, evap_depth)
        self.state.water_depth -= evaporated
        
        # Calculate volume removed
        volume = float(np.sum(evaporated) * self.cell_area)
        self.state.mass_balance.total_evaporation += volume
        
        return volume
    
    def apply_edge_drainage(self) -> float:
        """
        Remove water at domain boundaries to simulate outflow.
        
        This prevents water from pooling indefinitely at the edges
        and simulates drainage to areas outside the DEM.
        
        Returns:
            Total volume of water removed (m³)
        """
        if not self.params.enable_edge_drainage or self.params.edge_drainage_rate <= 0:
            return 0.0
        
        # Calculate drainage amount at edges
        drained = self.state.water_depth[self._edge_mask] * self.params.edge_drainage_rate
        self.state.water_depth[self._edge_mask] -= drained
        
        # Calculate volume removed
        volume = float(np.sum(drained) * self.cell_area)
        self.state.mass_balance.total_edge_drainage += volume
        
        return volume
    
    def compute_flow(self) -> np.ndarray:
        """
        Compute water flow between cells using 4-neighbor diffusion.
        
        Uses physically-scaled diffusion factor that accounts for:
        - The diffusion coefficient (m²/s)
        - The timestep (converted to seconds)
        - The cell area (m²)
        
        Returns:
            Net change in water depth for each cell
        """
        terrain = self.state.terrain
        water = self.state.water_depth
        surface = self.state.water_surface
        
        # Initialize flow accumulator
        flow = np.zeros_like(water)
        
        # Get dimensions
        rows, cols = water.shape
        
        # Use pre-computed effective diffusion factor
        k = self._effective_diffusion
        
        # Flow directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            # Create shifted arrays for neighbor comparison
            # Current cell values
            curr_surface = surface[1:-1, 1:-1]
            curr_water = water[1:-1, 1:-1]
            curr_terrain = terrain[1:-1, 1:-1]
            
            # Neighbor values (shifted)
            if dr == -1:
                neigh_surface = surface[:-2, 1:-1]
                neigh_water = water[:-2, 1:-1]
                neigh_terrain = terrain[:-2, 1:-1]
            elif dr == 1:
                neigh_surface = surface[2:, 1:-1]
                neigh_water = water[2:, 1:-1]
                neigh_terrain = terrain[2:, 1:-1]
            elif dc == -1:
                neigh_surface = surface[1:-1, :-2]
                neigh_water = water[1:-1, :-2]
                neigh_terrain = terrain[1:-1, :-2]
            else:  # dc == 1
                neigh_surface = surface[1:-1, 2:]
                neigh_water = water[1:-1, 2:]
                neigh_terrain = terrain[1:-1, 2:]
            
            # Calculate surface difference (positive means current is higher)
            surface_diff = curr_surface - neigh_surface
            
            # Only flow where current surface is higher and has water
            can_flow = (surface_diff > 0) & (curr_water > self.params.flow_threshold)
            
            # Calculate flow amount using physical diffusion factor
            # Flow is proportional to surface difference * effective diffusion
            flow_amount = np.zeros_like(curr_water)
            flow_amount[can_flow] = np.minimum(
                surface_diff[can_flow] * k,
                curr_water[can_flow] * self.params.max_flow_fraction
            )
            
            # Ensure we don't create negative depths
            flow_amount = np.minimum(flow_amount, curr_water)
            
            # Apply flow to accumulator (outflow from current cell)
            flow[1:-1, 1:-1] -= flow_amount
            
            # Apply flow to neighbor (inflow to neighbor)
            if dr == -1:
                flow[:-2, 1:-1] += flow_amount
            elif dr == 1:
                flow[2:, 1:-1] += flow_amount
            elif dc == -1:
                flow[1:-1, :-2] += flow_amount
            else:
                flow[1:-1, 2:] += flow_amount
        
        return flow
    
    def _check_convergence(self, max_depth: float) -> Tuple[bool, str]:
        """
        Check if simulation has converged (steady state reached).
        
        Returns:
            Tuple of (converged: bool, reason: str)
        """
        if self.params.convergence_steps <= 0:
            return False, ""
        
        self._recent_max_depths.append(max_depth)
        
        # Keep only recent history
        if len(self._recent_max_depths) > self.params.convergence_steps:
            self._recent_max_depths.pop(0)
        
        # Check if we have enough history
        if len(self._recent_max_depths) < self.params.convergence_steps:
            return False, ""
        
        # Check if max depth has stabilized
        depth_range = max(self._recent_max_depths) - min(self._recent_max_depths)
        if depth_range < self.params.convergence_threshold:
            return True, f"Converged: max depth stable within {depth_range:.4f}m for {self.params.convergence_steps} steps"
        
        return False, ""
    
    def step(self, apply_rain: bool = True) -> dict:
        """
        Advance simulation by one timestep.
        
        Args:
            apply_rain: Whether to apply rainfall this step
        
        Returns:
            Dictionary with step statistics
        """
        # Apply rainfall if within duration
        rain_volume = 0.0
        if apply_rain and self.state.time < self.params.rainfall_duration:
            rain_volume = self.apply_rainfall()
        
        # Compute and apply flow
        flow = self.compute_flow()
        self.state.water_depth += flow
        
        # Apply water losses
        infil_volume = self.apply_infiltration()
        evap_volume = self.apply_evaporation()
        drain_volume = self.apply_edge_drainage()
        
        # Ensure no negative depths
        self.state.water_depth = np.maximum(0, self.state.water_depth)
        
        # Update current volume in mass balance
        self.state.mass_balance.current_volume = float(
            np.sum(self.state.water_depth) * self.cell_area
        )
        
        # Update time
        self.state.time += self.params.timestep
        self.state.step += 1
        
        # Calculate statistics
        flooded = self.state.flooded_mask
        max_depth = float(np.max(self.state.water_depth))
        
        stats = {
            'time': self.state.time,
            'step': self.state.step,
            'max_depth': max_depth,
            'mean_depth': float(np.mean(self.state.water_depth[flooded])) if flooded.any() else 0.0,
            'flooded_cells': int(np.sum(flooded)),
            'flooded_area_km2': float(np.sum(flooded) * self.cell_area / 1e6),
            'total_volume_m3': self.state.mass_balance.current_volume,
            'rainfall_volume_m3': rain_volume,
            'infiltration_volume_m3': infil_volume,
            'evaporation_volume_m3': evap_volume,
            'edge_drainage_volume_m3': drain_volume,
            'mass_balance_error': self.state.mass_balance.relative_error
        }
        
        return stats
    
    def run(
        self, 
        duration: Optional[float] = None,
        max_steps: int = 1000,
        record_interval: int = 1,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None
    ) -> SimulationResult:
        """
        Run the complete simulation.
        
        Args:
            duration: Total simulation time (hours). Uses rainfall_duration * 2 if None.
            max_steps: Maximum number of steps to run
            record_interval: Record depth every N steps
            progress_callback: Optional callback(step, total_steps, stats)
        
        Returns:
            SimulationResult with complete history
        """
        if duration is None:
            duration = self.params.rainfall_duration * 2
        
        # Update diffusion factor in case params changed
        self._update_diffusion_factor()
        
        # Reset convergence tracking
        self._recent_max_depths = []
        
        # Calculate total steps
        total_steps = min(int(duration / self.params.timestep), max_steps)
        
        # Initialize history
        depth_history = [self.state.water_depth.copy()]
        time_history = [0.0]
        stats_history = []
        
        stopped_early = False
        stop_reason = ""
        
        # Run simulation
        for i in range(total_steps):
            stats = self.step()
            
            # Record state at intervals
            if (i + 1) % record_interval == 0:
                depth_history.append(self.state.water_depth.copy())
                time_history.append(self.state.time)
                stats_history.append(stats)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_steps, stats)
            
            # Check convergence
            converged, reason = self._check_convergence(stats['max_depth'])
            if converged:
                stopped_early = True
                stop_reason = reason
                # Ensure final state is recorded
                if time_history[-1] != self.state.time:
                    depth_history.append(self.state.water_depth.copy())
                    time_history.append(self.state.time)
                    stats_history.append(stats)
                break
            
            # Check for mass balance issues
            if self.state.mass_balance.relative_error > 0.1:  # 10% error threshold
                import warnings
                warnings.warn(
                    f"Mass balance error exceeds 10%: {self.state.mass_balance.relative_error:.1%}. "
                    "Results may be unreliable."
                )
        
        return SimulationResult(
            params=self.params,
            terrain=self.terrain,
            depth_history=depth_history,
            time_history=time_history,
            stats_history=stats_history,
            final_mass_balance=MassBalance(
                total_rainfall=self.state.mass_balance.total_rainfall,
                total_infiltration=self.state.mass_balance.total_infiltration,
                total_evaporation=self.state.mass_balance.total_evaporation,
                total_edge_drainage=self.state.mass_balance.total_edge_drainage,
                current_volume=self.state.mass_balance.current_volume
            ),
            stopped_early=stopped_early,
            stop_reason=stop_reason
        )
    
    def reset(self) -> None:
        """Reset simulation to initial state, preserving per-cell dam heights."""
        # Restore terrain with per-cell dam heights
        terrain_with_dams = self.original_terrain.copy()
        terrain_with_dams[self.dam_mask] += self.dam_heights[self.dam_mask]
        
        self.state = SimulationState(
            terrain=terrain_with_dams,
            water_depth=np.zeros_like(self.original_terrain, dtype=np.float32)
        )
        
        # Reset convergence tracking
        self._recent_max_depths = []


class RealisticFloodModel(BaseFloodModel):
    """
    Realistic flood model using Manning's equation-based flow.
    
    This model uses a stable, physically-based approach:
    - Flow velocity based on Manning's equation: v = (1/n) * R^(2/3) * S^(1/2)
    - For shallow flow: R ≈ h (hydraulic radius ≈ depth)
    - Explicit timestep with CFL-based stability
    
    More accurate than simple diffusion, suitable for flood inundation studies.
    """
    
    def __init__(self, terrain: TerrainData, params: Optional[SimulationParams] = None):
        self.terrain = terrain
        self.params = params or SimulationParams()
        self._available = True
        
        # Physical constants
        self.manning_n = 0.035  # Manning's roughness (natural channels)
        self.min_depth = 0.001  # minimum depth for computation (m)
        
        # Pre-compute cell dimensions
        self.cell_area = self._calculate_cell_area()
        self.dx = np.sqrt(self.cell_area)
        
        # Initialize terrain
        self.z = terrain.elevation.astype(np.float64)
        self.valid_mask = ~np.isnan(self.z)
        
        # Fill NaN with high values (impassable)
        z_max = np.nanmax(self.z) if not np.all(np.isnan(self.z)) else 0
        self.z = np.nan_to_num(self.z, nan=z_max + 1000)
        
        # Initialize water depth
        self.h = np.zeros_like(self.z, dtype=np.float64)
        
        # Mass balance
        self.mass_balance = MassBalance()
        self.current_time = 0.0
    
    @property
    def name(self) -> str:
        return "Realistic (Shallow Water)"
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _calculate_cell_area(self) -> float:
        """Calculate cell area in m²."""
        import math
        bounds = self.terrain.bounds
        crs = self.terrain.crs
        
        if 'EPSG:4326' in crs or (abs(bounds[0]) <= 180 and abs(bounds[2]) <= 180):
            lat_center = (bounds[1] + bounds[3]) / 2
            m_per_deg_lat = 111320
            m_per_deg_lon = 111320 * math.cos(math.radians(lat_center))
            cell_width_m = self.terrain.resolution[0] * m_per_deg_lon
            cell_height_m = self.terrain.resolution[1] * m_per_deg_lat
            return cell_width_m * cell_height_m
        else:
            return self.terrain.resolution[0] * self.terrain.resolution[1]
    
    def _compute_manning_flow(self, dt_seconds: float) -> np.ndarray:
        """
        Compute water flow using Manning's equation with 4-neighbor scheme.
        
        Args:
            dt_seconds: Timestep in seconds
            
        Returns:
            Net depth change for each cell (m)
        """
        h = self.h
        z = self.z
        dx = self.dx
        n = self.manning_n
        
        # Water surface elevation
        eta = z + h
        
        # Flow accumulator (depth change in meters)
        dh = np.zeros_like(h)
        
        rows, cols = h.shape
        
        # Process interior cells with 4-neighbor scheme
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Skip dry cells
                if h[i, j] < self.min_depth:
                    continue
                
                # Check each neighbor
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    
                    # Surface elevation difference
                    deta = eta[i, j] - eta[ni, nj]
                    
                    # Only flow downhill from current cell
                    if deta <= 0:
                        continue
                    
                    # Slope (dimensionless)
                    slope = deta / dx
                    slope = min(slope, 0.5)  # Cap slope for stability
                    
                    # Manning velocity: v = (1/n) * h^(2/3) * S^(1/2)
                    h_flow = min(h[i, j], deta)  # Can't flow more than height difference
                    velocity = (1.0 / n) * (h_flow ** (2.0/3.0)) * np.sqrt(slope)
                    velocity = min(velocity, 5.0)  # Cap velocity at 5 m/s
                    
                    # Flow rate per unit width: q = v * h
                    q = velocity * h_flow
                    
                    # Volume transferred in this timestep
                    vol_transfer = q * dx * dt_seconds  # m³
                    
                    # Convert to depth change
                    depth_transfer = vol_transfer / self.cell_area
                    
                    # Don't transfer more than available
                    depth_transfer = min(depth_transfer, h[i, j] * 0.2)  # Max 20% per neighbor per step
                    
                    # Apply transfer
                    dh[i, j] -= depth_transfer
                    dh[ni, nj] += depth_transfer
        
        return dh
    
    def run(
        self,
        duration: Optional[float] = None,
        max_steps: int = 1000,
        record_interval: int = 1,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None
    ) -> SimulationResult:
        """Run Manning's equation-based flood simulation."""
        if duration is None:
            duration = self.params.rainfall_duration * 2
        
        # Reset state
        self.h = np.zeros_like(self.z, dtype=np.float64)
        self.current_time = 0.0
        self.mass_balance = MassBalance()
        
        # Use fixed small timestep for stability (in hours)
        dt_hours = min(self.params.timestep, 0.01)  # Max 0.01 hours = 36 seconds
        dt_seconds = dt_hours * 3600
        
        total_steps = int(duration / dt_hours)
        total_steps = min(total_steps, max_steps * 10)  # Allow more internal steps
        
        # Initialize history
        depth_history = [self.h.copy().astype(np.float32)]
        time_history = [0.0]
        stats_history = []
        
        # Record every N steps to get ~50 frames
        record_every = max(1, total_steps // 50)
        
        for step in range(total_steps):
            # Apply rainfall if within duration
            if self.current_time < self.params.rainfall_duration:
                rain_depth = (self.params.rainfall_intensity / 1000.0) * dt_hours
                self.h[self.valid_mask] += rain_depth
                self.mass_balance.total_rainfall += rain_depth * np.sum(self.valid_mask) * self.cell_area
            
            # Compute Manning-based flow
            dh = self._compute_manning_flow(dt_seconds)
            self.h += dh
            
            # Apply infiltration
            if self.params.infiltration_rate > 0:
                infil_depth = (self.params.infiltration_rate / 1000.0) * dt_hours
                infiltrated = np.minimum(self.h, infil_depth)
                self.h -= infiltrated
                self.mass_balance.total_infiltration += float(np.sum(infiltrated) * self.cell_area)
            
            # Edge drainage
            if self.params.enable_edge_drainage and self.params.edge_drainage_rate > 0:
                edge_mask = np.zeros_like(self.h, dtype=bool)
                edge_mask[0, :] = True
                edge_mask[-1, :] = True
                edge_mask[:, 0] = True
                edge_mask[:, -1] = True
                
                drained = self.h[edge_mask] * self.params.edge_drainage_rate
                self.h[edge_mask] -= drained
                self.mass_balance.total_edge_drainage += float(np.sum(drained) * self.cell_area)
            
            # Ensure non-negative depths and apply bounds
            self.h = np.clip(self.h, 0, 100)  # Cap at 100m
            self.h[~self.valid_mask] = 0
            
            self.current_time += dt_hours
            
            # Record at intervals
            if step % record_every == 0 or step == total_steps - 1:
                flooded = self.h > 0.001
                self.mass_balance.current_volume = float(np.sum(self.h) * self.cell_area)
                
                stats = {
                    'time': self.current_time,
                    'step': step,
                    'max_depth': float(np.max(self.h)),
                    'mean_depth': float(np.mean(self.h[flooded])) if flooded.any() else 0.0,
                    'flooded_cells': int(np.sum(flooded)),
                    'flooded_area_km2': float(np.sum(flooded) * self.cell_area / 1e6),
                    'total_volume_m3': self.mass_balance.current_volume,
                    'rainfall_volume_m3': 0.0,
                    'infiltration_volume_m3': 0.0,
                    'evaporation_volume_m3': 0.0,
                    'edge_drainage_volume_m3': 0.0,
                    'mass_balance_error': self.mass_balance.relative_error
                }
                
                depth_history.append(self.h.copy().astype(np.float32))
                time_history.append(self.current_time)
                stats_history.append(stats)
            
            # Progress callback
            if progress_callback and step % max(1, total_steps // 100) == 0:
                flooded = self.h > 0.001
                stats = {
                    'time': self.current_time,
                    'max_depth': float(np.max(self.h)),
                    'flooded_area_km2': float(np.sum(flooded) * self.cell_area / 1e6),
                }
                progress_callback(step + 1, total_steps, stats)
        
        return SimulationResult(
            params=self.params,
            terrain=self.terrain,
            depth_history=depth_history,
            time_history=time_history,
            stats_history=stats_history,
            final_mass_balance=MassBalance(
                total_rainfall=self.mass_balance.total_rainfall,
                total_infiltration=self.mass_balance.total_infiltration,
                total_evaporation=self.mass_balance.total_evaporation,
                total_edge_drainage=self.mass_balance.total_edge_drainage,
                current_volume=self.mass_balance.current_volume
            ),
            stopped_early=False,
            stop_reason=""
        )
    
    def reset(self) -> None:
        """Reset simulation state."""
        self.h = np.zeros_like(self.z, dtype=np.float64)
        self.current_time = 0.0
        self.mass_balance = MassBalance()


def get_available_solvers(terrain: TerrainData, params: SimulationParams = None) -> dict:
    """
    Get dictionary of available flood simulation solvers.
    
    Returns:
        Dict mapping solver names to (solver_instance, is_available) tuples
    """
    fast_solver = FloodSimulator(terrain, params)
    realistic_solver = RealisticFloodModel(terrain, params)
    
    return {
        "Fast (NumPy)": (fast_solver, True),
        "Realistic (LISFLOOD-FP)": (realistic_solver, realistic_solver.is_available)
    }


def create_dam_line(
    shape: Tuple[int, int],
    start: Tuple[int, int],
    end: Tuple[int, int],
    width: int = 3
) -> np.ndarray:
    """
    Create a dam mask as a line between two points.
    
    Args:
        shape: Shape of the terrain array (rows, cols)
        start: Starting point (row, col)
        end: Ending point (row, col)
        width: Width of the dam in cells
    
    Returns:
        Boolean mask where True indicates dam cells
    """
    from scipy import ndimage
    
    mask = np.zeros(shape, dtype=bool)
    
    # Use Bresenham's line algorithm (simplified)
    r0, c0 = start
    r1, c1 = end
    
    # Calculate steps
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    steps = max(dr, dc)
    
    if steps == 0:
        mask[r0, c0] = True
    else:
        for i in range(steps + 1):
            t = i / steps
            r = int(r0 + t * (r1 - r0))
            c = int(c0 + t * (c1 - c0))
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
    
    # Dilate to create width
    if width > 1:
        struct = ndimage.generate_binary_structure(2, 1)
        for _ in range(width // 2):
            mask = ndimage.binary_dilation(mask, struct)
    
    return mask


def create_dam_polygon(
    shape: Tuple[int, int],
    vertices: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Create a dam mask from polygon vertices.
    
    Args:
        shape: Shape of the terrain array (rows, cols)
        vertices: List of (row, col) vertices defining the polygon
    
    Returns:
        Boolean mask where True indicates dam cells
    """
    from matplotlib.path import Path
    
    # Create coordinate grid
    rows, cols = shape
    r, c = np.mgrid[:rows, :cols]
    points = np.vstack((r.ravel(), c.ravel())).T
    
    # Create path and check containment
    path = Path(vertices)
    mask = path.contains_points(points).reshape(shape)
    
    return mask


# Quick test
if __name__ == "__main__":
    from terrain_loader import load_dem
    import matplotlib.pyplot as plt
    
    # Load terrain
    print("Loading terrain...")
    terrain = load_dem("data/daklak.tif", max_size=200)
    
    print(f"Terrain shape: {terrain.elevation.shape}")
    print(f"Elevation range: {np.nanmin(terrain.elevation):.1f} - {np.nanmax(terrain.elevation):.1f} m")
    
    # Create simulator with improved parameters
    params = SimulationParams(
        rainfall_intensity=50.0,    # Heavy rain (mm/hr)
        rainfall_duration=2.0,      # 2 hours
        timestep=0.05,              # 3 minutes
        diffusion_coef=1.0,         # m²/s - physical units
        infiltration_rate=5.0,      # mm/hr - some soil absorption
        edge_drainage_rate=0.1,     # 10% of edge water drains per step
        enable_edge_drainage=True,
        convergence_steps=20        # Enable early stopping
    )
    
    sim = FloodSimulator(terrain, params)
    
    print(f"\nCell area: {sim.cell_area:.1f} m²")
    print(f"Effective diffusion factor: {sim._effective_diffusion:.4f}")
    
    # Run simulation
    print("\nRunning simulation...")
    
    def progress(step, total, stats):
        if step % 10 == 0:
            print(f"  Step {step}/{total}: max_depth={stats['max_depth']:.3f}m, "
                  f"flooded_area={stats['flooded_area_km2']:.2f} km², "
                  f"mass_error={stats['mass_balance_error']:.1%}")
    
    result = sim.run(duration=6.0, record_interval=5, progress_callback=progress)
    
    print(f"\nSimulation complete!")
    print(f"  Total timesteps: {len(result.time_history)}")
    print(f"  Final max depth: {np.max(result.final_depth):.3f} m")
    print(f"  Max depth reached: {np.max(result.max_depth):.3f} m")
    print(f"  Stopped early: {result.stopped_early}")
    if result.stop_reason:
        print(f"  Stop reason: {result.stop_reason}")
    
    # Mass balance summary
    mb = result.final_mass_balance
    print(f"\nMass Balance:")
    print(f"  Total rainfall:    {mb.total_rainfall/1e6:.2f} million m³")
    print(f"  Infiltration:      {mb.total_infiltration/1e6:.2f} million m³")
    print(f"  Evaporation:       {mb.total_evaporation/1e6:.2f} million m³")
    print(f"  Edge drainage:     {mb.total_edge_drainage/1e6:.2f} million m³")
    print(f"  Current volume:    {mb.current_volume/1e6:.2f} million m³")
    print(f"  Balance error:     {mb.relative_error:.2%}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Terrain
    axes[0].imshow(terrain.elevation, cmap='terrain')
    axes[0].set_title('Terrain Elevation')
    axes[0].axis('off')
    
    # Final water depth
    im1 = axes[1].imshow(result.final_depth, cmap='Blues', vmin=0)
    axes[1].set_title('Final Water Depth')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Depth (m)')
    
    # Max depth overlay
    axes[2].imshow(terrain.elevation, cmap='terrain', alpha=0.7)
    im2 = axes[2].imshow(result.max_depth, cmap='Blues', alpha=0.6, vmin=0)
    axes[2].set_title('Max Flood Extent')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='Max Depth (m)')
    
    plt.tight_layout()
    plt.show()
