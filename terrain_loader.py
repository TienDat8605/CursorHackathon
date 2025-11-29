"""
terrain_loader.py - Load and process Digital Elevation Model (DEM) data from GeoTIFF files.

Features:
- Load DEM from GeoTIFF using rasterio
- Handle nodata values and NaN interpolation
- Optional downsampling for large DEMs
- Extract geographic bounds and CRS
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


@dataclass
class TerrainData:
    """Container for terrain data and metadata."""
    elevation: np.ndarray          # 2D array of elevations (meters)
    bounds: Tuple[float, float, float, float]  # (west, south, east, north)
    transform: rasterio.Affine     # Affine transform for pixel-to-geo conversion
    crs: str                       # Coordinate Reference System
    resolution: Tuple[float, float]  # (x_res, y_res) in CRS units
    nodata_value: Optional[float]  # Original nodata value


def load_dem(
    filepath: str,
    max_size: Optional[int] = None,
    fill_nodata: bool = True,
    band: int = 1
) -> TerrainData:
    """
    Load a DEM from a GeoTIFF file.
    
    Args:
        filepath: Path to the GeoTIFF file
        max_size: Maximum dimension (width or height). If exceeded, downsample.
        fill_nodata: If True, interpolate nodata values
        band: Band number to read (default 1)
    
    Returns:
        TerrainData object with elevation array and metadata
    """
    with rasterio.open(filepath) as src:
        # Get metadata
        nodata = src.nodata
        crs = str(src.crs) if src.crs else "EPSG:4326"
        transform = src.transform
        bounds = src.bounds
        
        # Calculate resolution
        resolution = (abs(transform.a), abs(transform.e))
        
        # Determine if downsampling is needed
        if max_size and (src.width > max_size or src.height > max_size):
            scale_factor = max_size / max(src.width, src.height)
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)
            
            elevation = src.read(
                band,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear
            )
            
            # Update transform for downsampled data
            transform = src.transform * src.transform.scale(
                src.width / new_width,
                src.height / new_height
            )
            resolution = (resolution[0] / scale_factor, resolution[1] / scale_factor)
        else:
            elevation = src.read(band)
        
        # Convert to float for processing
        elevation = elevation.astype(np.float32)
        
        # Handle nodata values
        if nodata is not None:
            elevation[elevation == nodata] = np.nan
        
        # Fill nodata with interpolation
        if fill_nodata:
            elevation = _fill_nodata(elevation)
        
        return TerrainData(
            elevation=elevation,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            transform=transform,
            crs=crs,
            resolution=resolution,
            nodata_value=nodata
        )


def _fill_nodata(elevation: np.ndarray) -> np.ndarray:
    """
    Fill NaN values using simple nearest-neighbor interpolation.
    For more complex cases, consider scipy.ndimage or rasterio.fill.
    """
    from scipy import ndimage
    
    mask = np.isnan(elevation)
    if not mask.any():
        return elevation
    
    # Use distance transform to find nearest valid values
    indices = ndimage.distance_transform_edt(
        mask, 
        return_distances=False, 
        return_indices=True
    )
    filled = elevation[tuple(indices)]
    
    return filled


def normalize_elevation(elevation: np.ndarray) -> np.ndarray:
    """
    Normalize elevation to 0-1 range for visualization.
    """
    valid_mask = ~np.isnan(elevation)
    if not valid_mask.any():
        return np.zeros_like(elevation)
    
    min_val = np.nanmin(elevation)
    max_val = np.nanmax(elevation)
    
    if max_val - min_val < 1e-6:
        return np.zeros_like(elevation)
    
    return (elevation - min_val) / (max_val - min_val)


def get_elevation_colormap(
    elevation: np.ndarray,
    colormap: str = 'terrain',
    add_hillshade: bool = True,
    azimuth: float = 315,
    altitude: float = 45
) -> np.ndarray:
    """
    Generate a colored elevation map with optional hillshade.
    
    Args:
        elevation: 2D elevation array
        colormap: Matplotlib colormap name
        add_hillshade: If True, blend with hillshade for 3D effect
        azimuth: Light source azimuth angle (degrees)
        altitude: Light source altitude angle (degrees)
    
    Returns:
        RGBA array suitable for display
    """
    normalized = normalize_elevation(elevation)
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)
    
    if add_hillshade:
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        # Create hillshade
        hillshade = ls.hillshade(elevation, vert_exag=1.5)
        # Blend colors with hillshade
        for i in range(3):
            colored[:, :, i] = colored[:, :, i] * 0.6 + hillshade * 0.4
    
    return (colored * 255).astype(np.uint8)


def plot_elevation(
    terrain: TerrainData,
    title: str = "Digital Elevation Model",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization of the elevation data.
    
    Args:
        terrain: TerrainData object
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create extent from bounds for proper georeferencing
    extent = [terrain.bounds[0], terrain.bounds[2], 
              terrain.bounds[1], terrain.bounds[3]]
    
    # Plot with hillshade
    colored = get_elevation_colormap(terrain.elevation)
    ax.imshow(colored, extent=extent, origin='upper')
    
    # Add colorbar for elevation
    im = ax.imshow(
        terrain.elevation, 
        extent=extent, 
        origin='upper',
        cmap='terrain',
        alpha=0
    )
    cbar = plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _calculate_area_km2(terrain: TerrainData) -> float:
    """
    Calculate approximate area in km² accounting for geographic coordinates.
    """
    bounds = terrain.bounds
    
    # Check if coordinates are in degrees (lat/lon)
    if 'EPSG:4326' in terrain.crs or (abs(bounds[0]) <= 180 and abs(bounds[2]) <= 180):
        # Geographic coordinates - approximate conversion
        import math
        lat_center = (bounds[1] + bounds[3]) / 2
        
        # Approximate meters per degree at this latitude
        m_per_deg_lat = 111320  # roughly constant
        m_per_deg_lon = 111320 * math.cos(math.radians(lat_center))
        
        width_m = (bounds[2] - bounds[0]) * m_per_deg_lon
        height_m = (bounds[3] - bounds[1]) * m_per_deg_lat
        
        return (width_m * height_m) / 1e6
    else:
        # Projected coordinates - assume meters
        return (terrain.elevation.shape[0] * terrain.resolution[1] * 
                terrain.elevation.shape[1] * terrain.resolution[0]) / 1e6


def get_terrain_stats(terrain: TerrainData) -> dict:
    """
    Calculate basic statistics about the terrain.
    """
    elevation = terrain.elevation
    valid = elevation[~np.isnan(elevation)]
    
    return {
        'min_elevation': float(np.min(valid)),
        'max_elevation': float(np.max(valid)),
        'mean_elevation': float(np.mean(valid)),
        'std_elevation': float(np.std(valid)),
        'shape': elevation.shape,
        'resolution': terrain.resolution,
        'crs': terrain.crs,
        'bounds': terrain.bounds,
        'area_km2': _calculate_area_km2(terrain)
    }


# Quick test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/daklak.tif"
    
    print(f"Loading DEM from: {filepath}")
    terrain = load_dem(filepath, max_size=500)
    
    stats = get_terrain_stats(terrain)
    print("\nTerrain Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nGenerating visualization...")
    fig = plot_elevation(terrain, title=f"DEM: {filepath}")
    plt.show()

