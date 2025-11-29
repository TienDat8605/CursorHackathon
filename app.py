"""
FloodSim Sandbox - Interactive Flood Simulation App

A Streamlit-based application for simulating and visualizing flood propagation
over real-world terrain using Digital Elevation Models (DEM).

Version 2.0 - Stability & Realism Improvements:
- Physical diffusion scaling (respects timestep and grid resolution)
- Edge drainage for realistic water outflow
- Mass balance tracking and convergence detection
- Improved parameter documentation with clear units
"""

import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import tempfile
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from terrain_loader import load_dem, get_terrain_stats, get_elevation_colormap, TerrainData
from simulation_core import (
    FloodSimulator, SimulationParams, SimulationResult,
    get_available_solvers, RealisticFloodModel
)


# Page configuration
st.set_page_config(
    page_title="🌊 FloodSim Sandbox",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 50%, #90e0ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #000000 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetric"] label {
        color: #64748b;
        font-weight: 500;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: #0077b6;
    }
    
    /* Cards and containers */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    .video-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .video-title {
        color: #f8fafc;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0f172a;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e40af;
    }
    
    /* Radio buttons */
    .stRadio > div {
        gap: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation keyframes */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .loading {
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'terrain' not in st.session_state:
        st.session_state.terrain = None
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    if 'dam_points' not in st.session_state:
        st.session_state.dam_points = []
    if 'video_bytes' not in st.session_state:
        st.session_state.video_bytes = None


init_session_state()


def get_sample_files():
    """Get list of sample DEM files in the data directory."""
    data_dir = Path("data")
    if data_dir.exists():
        return list(data_dir.glob("*.tif"))
    return []


def create_water_overlay_image(
    water_depth: np.ndarray,
    max_depth: float = None,
    use_discrete_colors: bool = True
) -> np.ndarray:
    """
    Create a semi-transparent water overlay image with blue colors.
    
    Color levels (light blue to deep blue based on depth):
    - 0 - 0.3m: Very light blue (#b3d9ff)
    - 0.3 - 0.5m: Light blue (#66b3ff)
    - 0.5 - 1.0m: Medium blue (#3399ff)
    - 1.0 - 2.0m: Dark blue (#0066cc)
    - > 2.0m: Deep navy (#003366)
    """
    colored = np.zeros((*water_depth.shape, 4), dtype=np.float32)
    
    if use_discrete_colors:
        # Discrete color levels based on depth thresholds
        # Level 1: 0 - 0.3m (very light blue)
        mask1 = (water_depth > 0.001) & (water_depth <= 0.3)
        colored[mask1, 0] = 0.70  # R
        colored[mask1, 1] = 0.85  # G
        colored[mask1, 2] = 1.00  # B
        colored[mask1, 3] = 0.50  # A
        
        # Level 2: 0.3 - 0.5m (light blue)
        mask2 = (water_depth > 0.3) & (water_depth <= 0.5)
        colored[mask2, 0] = 0.40  # R
        colored[mask2, 1] = 0.70  # G
        colored[mask2, 2] = 1.00  # B
        colored[mask2, 3] = 0.60  # A
        
        # Level 3: 0.5 - 1.0m (medium blue)
        mask3 = (water_depth > 0.5) & (water_depth <= 1.0)
        colored[mask3, 0] = 0.20  # R
        colored[mask3, 1] = 0.60  # G
        colored[mask3, 2] = 1.00  # B
        colored[mask3, 3] = 0.70  # A
        
        # Level 4: 1.0 - 2.0m (dark blue)
        mask4 = (water_depth > 1.0) & (water_depth <= 2.0)
        colored[mask4, 0] = 0.00  # R
        colored[mask4, 1] = 0.40  # G
        colored[mask4, 2] = 0.80  # B
        colored[mask4, 3] = 0.80  # A
        
        # Level 5: > 2.0m (deep navy)
        mask5 = water_depth > 2.0
        colored[mask5, 0] = 0.00  # R
        colored[mask5, 1] = 0.20  # G
        colored[mask5, 2] = 0.40  # B
        colored[mask5, 3] = 0.90  # A
    else:
        # Continuous gradient (original behavior)
        if max_depth is None:
            max_depth = np.max(water_depth) if np.max(water_depth) > 0 else 1.0
        
        normalized = np.clip(water_depth / max_depth, 0, 1)
        
        # Deep blue gradient
        colored[:, :, 0] = 0.1 * (1 - normalized) + 0.0 * normalized
        colored[:, :, 1] = 0.3 * (1 - normalized) + 0.1 * normalized
        colored[:, :, 2] = 0.7 * (1 - normalized) + 0.6 * normalized
        colored[:, :, 3] = np.where(water_depth > 0.001, 0.4 + 0.5 * normalized, 0)
    
    return (colored * 255).astype(np.uint8)


def create_water_contour_overlay(
    water_depth: np.ndarray,
    levels: list = None
) -> np.ndarray:
    """
    Create water visualization as contour lines/filled regions.
    
    Args:
        water_depth: 2D array of water depths
        levels: Contour levels in meters. Default: [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    Returns:
        RGBA array with contour visualization
    """
    if levels is None:
        levels = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    # Filter levels to only those present in data
    max_depth = np.max(water_depth)
    levels = [l for l in levels if l <= max_depth * 1.5]
    if not levels:
        levels = [0.1]
    
    # Create figure for contour
    fig, ax = plt.subplots(figsize=(water_depth.shape[1]/100, water_depth.shape[0]/100), dpi=100)
    ax.set_position([0, 0, 1, 1])
    
    # Blue colormap for water depths
    from matplotlib.colors import LinearSegmentedColormap
    water_colors = [
        '#b3d9ff',  # very light blue
        '#66b3ff',  # light blue
        '#3399ff',  # medium blue
        '#0066cc',  # dark blue
        '#003366',  # deep navy
        '#001a33'   # very deep
    ][:len(levels)]
    
    # Filled contours
    if len(levels) > 1:
        cf = ax.contourf(water_depth, levels=[0] + levels + [max_depth * 2], 
                        colors=['#ffffff00'] + water_colors, extend='max')
    
    # Contour lines
    cs = ax.contour(water_depth, levels=levels, colors='#003366', linewidths=0.8, alpha=0.8)
    
    # Add labels on contours
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.1fm')
    
    ax.axis('off')
    ax.set_xlim(0, water_depth.shape[1])
    ax.set_ylim(water_depth.shape[0], 0)
    
    # Convert to image
    fig.canvas.draw()
    
    # Get the RGBA buffer
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf)
    
    plt.close(fig)
    
    # Resize to match water_depth shape
    pil_img = Image.fromarray(rgba)
    pil_img = pil_img.resize((water_depth.shape[1], water_depth.shape[0]), Image.Resampling.LANCZOS)
    
    return np.array(pil_img)


def create_terrain_overlay(terrain: TerrainData, use_contour: bool = False) -> np.ndarray:
    """Create terrain visualization overlay without blue colors."""
    if use_contour:
        return create_contour_overlay(terrain.elevation)
    # Use earth tones colormap (no blue): YlOrBr (yellow-orange-brown)
    return get_elevation_colormap(terrain.elevation, colormap='YlOrBr', add_hillshade=True)


def create_contour_overlay(elevation: np.ndarray, num_levels: int = 15) -> np.ndarray:
    """Create a contour map visualization of elevation."""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create figure for contour
    fig, ax = plt.subplots(figsize=(elevation.shape[1]/100, elevation.shape[0]/100), dpi=100)
    ax.set_position([0, 0, 1, 1])
    
    # Create contour levels
    valid_elev = elevation[~np.isnan(elevation)]
    levels = np.linspace(np.min(valid_elev), np.max(valid_elev), num_levels)
    
    # Earth tone colors for contours
    colors = ['#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', 
              '#238443', '#006837', '#004529', '#543005', '#8c510a',
              '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1'][:num_levels]
    
    # Filled contours
    ax.contourf(elevation, levels=levels, colors=colors, extend='both')
    
    # Contour lines
    ax.contour(elevation, levels=levels, colors='#333333', linewidths=0.3, alpha=0.5)
    
    ax.axis('off')
    ax.set_xlim(0, elevation.shape[1])
    ax.set_ylim(elevation.shape[0], 0)
    
    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert ARGB to RGBA
    rgba = np.zeros_like(img)
    rgba[:, :, 0] = img[:, :, 1]  # R
    rgba[:, :, 1] = img[:, :, 2]  # G
    rgba[:, :, 2] = img[:, :, 3]  # B
    rgba[:, :, 3] = 255           # A
    
    plt.close(fig)
    
    # Resize to match elevation shape
    from PIL import Image
    pil_img = Image.fromarray(rgba)
    pil_img = pil_img.resize((elevation.shape[1], elevation.shape[0]), Image.Resampling.LANCZOS)
    
    return np.array(pil_img)


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG."""
    img = Image.fromarray(arr)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def create_folium_map(
    terrain: TerrainData,
    water_depth: np.ndarray = None,
    show_terrain: bool = True,
    show_water: bool = True,
    use_contour: bool = False,
    use_water_contour: bool = False,
    use_discrete_colors: bool = True,
    base_map: str = 'OpenStreetMap'
) -> folium.Map:
    """Create an interactive Folium map with terrain and water overlays.
    
    Args:
        terrain: TerrainData object with elevation data
        water_depth: Optional water depth array
        show_terrain: Whether to show terrain overlay
        show_water: Whether to show water overlay
        use_contour: Use contour lines instead of gradient for terrain
        use_water_contour: Use contour lines for water depth visualization
        use_discrete_colors: Use discrete depth-based colors (vs continuous gradient)
        base_map: Base map tiles ('OpenStreetMap', 'CartoDB positron', 'Terrain')
    """
    
    # Calculate center
    bounds = terrain.bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Map tile options
    tile_options = {
        'OpenStreetMap': 'OpenStreetMap',
        'CartoDB positron': 'CartoDB positron',
        'CartoDB dark': 'CartoDB dark_matter',
        'Terrain': 'Stamen Terrain'
    }
    
    # Create map with selected base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tile_options.get(base_map, 'OpenStreetMap')
    )
    
    # Add terrain layer (optional)
    if show_terrain:
        terrain_img = create_terrain_overlay(terrain, use_contour=use_contour)
        terrain_b64 = numpy_to_base64_png(terrain_img)
        
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{terrain_b64}",
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=0.7,
            name="Terrain",
            overlay=True
        ).add_to(m)
    
    # Add water layer
    if show_water and water_depth is not None and np.max(water_depth) > 0:
        if use_water_contour:
            water_img = create_water_contour_overlay(water_depth)
        else:
            water_img = create_water_overlay_image(water_depth, use_discrete_colors=use_discrete_colors)
        water_b64 = numpy_to_base64_png(water_img)
        
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{water_b64}",
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=0.85,
            name="Flood Water",
            overlay=True
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Fit bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    return m


def create_simulation_video(
    terrain: TerrainData,
    depth_history: list,
    time_history: list,
    fps: int = 10,
    use_discrete_colors: bool = True
) -> bytes:
    """Create a video/GIF animation of the flood simulation.
    
    Args:
        terrain: TerrainData object
        depth_history: List of water depth arrays over time
        time_history: List of time values
        fps: Frames per second
        use_discrete_colors: Use discrete depth-based color levels
    
    Returns:
        Video bytes (GIF format)
    """
    import io
    from PIL import Image
    
    frames = []
    max_depth = max(np.max(d) for d in depth_history) if depth_history else 1.0
    
    for i, (depth, time) in enumerate(zip(depth_history, time_history)):
        # Create frame
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show terrain (earth tones)
        terrain_colored = get_elevation_colormap(terrain.elevation, colormap='YlOrBr', add_hillshade=True)
        ax.imshow(terrain_colored, extent=[terrain.bounds[0], terrain.bounds[2], 
                                            terrain.bounds[1], terrain.bounds[3]])
        
        # Overlay water with discrete colors
        if np.max(depth) > 0.001:
            water_rgba = create_water_overlay_image(depth, use_discrete_colors=use_discrete_colors)
            ax.imshow(water_rgba, extent=[terrain.bounds[0], terrain.bounds[2],
                                          terrain.bounds[1], terrain.bounds[3]],
                     alpha=0.8)
        
        # Add time label
        ax.set_title(f'Flood Simulation - Time: {time:.1f} hours', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add color legend with actual colors matching the depth levels
        from matplotlib.patches import Rectangle
        legend_items = [
            ('0-0.3m', (0.70, 0.85, 1.00)),      # Very light blue
            ('0.3-0.5m', (0.40, 0.70, 1.00)),    # Light blue
            ('0.5-1m', (0.20, 0.60, 1.00)),      # Medium blue
            ('1-2m', (0.00, 0.40, 0.80)),        # Dark blue
            ('>2m', (0.00, 0.20, 0.40))          # Deep navy
        ]
        
        # Create legend box
        legend_y_start = 0.98
        legend_x = 0.98
        box_height = 0.03
        box_width = 0.025
        spacing = 0.04
        
        # Background box for legend
        legend_bg = Rectangle((legend_x - 0.12, legend_y_start - len(legend_items) * spacing - 0.02),
                             0.13, len(legend_items) * spacing + 0.03,
                             transform=ax.transAxes, facecolor='white', 
                             edgecolor='black', alpha=0.85, linewidth=1)
        ax.add_patch(legend_bg)
        
        # Add title
        ax.text(legend_x - 0.06, legend_y_start - 0.01, 'Water Depth',
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                verticalalignment='top', horizontalalignment='center')
        
        # Add each depth level with colored box
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y_start - (i + 1) * spacing
            
            # Color box
            rect = Rectangle((legend_x - 0.11, y_pos - box_height/2),
                           box_width, box_height,
                           transform=ax.transAxes, facecolor=color,
                           edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Label text
            ax.text(legend_x - 0.08, y_pos, label,
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='center', horizontalalignment='left')
        
        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
    
    # Create GIF
    gif_buffer = io.BytesIO()
    if frames:
        frames[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
    
    gif_buffer.seek(0)
    return gif_buffer.getvalue()


def display_video_inline(video_bytes: bytes, width: int = None):
    """
    Display a GIF video inline in Streamlit with beautiful styling.
    
    Args:
        video_bytes: GIF video as bytes
        width: Display width in pixels (auto if None)
    """
    video_b64 = base64.b64encode(video_bytes).decode()
    width_style = f'width="{width}"' if width else 'style="width: 100%; max-width: 900px;"'
    
    html = f'''
    <div class="video-container">
        <div class="video-title">
            🎬 Flood Simulation Animation
        </div>
        <div style="display: flex; justify-content: center;">
            <img src="data:image/gif;base64,{video_b64}" 
                 {width_style}
                 style="border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);"
                 alt="Flood Simulation Animation">
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def create_depth_legend():
    """Create a visual legend for water depth colors."""
    st.markdown("""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0;">
        <span style="display: flex; align-items: center; gap: 0.3rem;">
            <span style="display: inline-block; width: 20px; height: 14px; background: rgba(179,217,255,0.8); border: 1px solid #999; border-radius: 2px;"></span>
            <span style="font-size: 0.8rem;">0-0.3m</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.3rem;">
            <span style="display: inline-block; width: 20px; height: 14px; background: rgba(102,179,255,0.8); border: 1px solid #999; border-radius: 2px;"></span>
            <span style="font-size: 0.8rem;">0.3-0.5m</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.3rem;">
            <span style="display: inline-block; width: 20px; height: 14px; background: rgba(51,153,255,0.8); border: 1px solid #999; border-radius: 2px;"></span>
            <span style="font-size: 0.8rem;">0.5-1m</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.3rem;">
            <span style="display: inline-block; width: 20px; height: 14px; background: rgba(0,102,204,0.9); border: 1px solid #999; border-radius: 2px;"></span>
            <span style="font-size: 0.8rem;">1-2m</span>
        </span>
        <span style="display: flex; align-items: center; gap: 0.3rem;">
            <span style="display: inline-block; width: 20px; height: 14px; background: rgba(0,51,102,0.95); border: 1px solid #999; border-radius: 2px;"></span>
            <span style="font-size: 0.8rem;">&gt;2m</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def display_metrics(result: SimulationResult, frame_idx: int):
    """Display simulation metrics in columns."""
    if not result.stats_history:
        return
    
    stats = result.stats_history[min(frame_idx, len(result.stats_history) - 1)]
    
    col1, col3, col4 = st.columns(3)
    
    with col1:
        st.metric(
            label="⏱️ Time",
            value=f"{stats['time']:.1f} hrs"
        )
    
    with col3:
        st.metric(
            label="🌊 Flooded Area",
            value=f"{stats['flooded_area_km2']:.2f} km²"
        )
    
    with col4:
        st.metric(
            label="📊 Water Volume",
            value=f"{stats['total_volume_m3']/1e6:.2f} M m³"
        )


def display_mass_balance(result: SimulationResult):
    """Display mass balance information."""
    if result.final_mass_balance is None:
        return
    
    mb = result.final_mass_balance
    
    st.subheader("⚖️ Water Balance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Inputs:**")
        st.write(f"🌧️ Total Rainfall: {mb.total_rainfall/1e6:.3f} million m³")
        
        st.markdown("**Outputs:**")
        st.write(f"🌱 Infiltration: {mb.total_infiltration/1e6:.3f} million m³")
        st.write(f"☀️ Evaporation: {mb.total_evaporation/1e6:.3f} million m³")
        st.write(f"🚿 Edge Drainage: {mb.total_edge_drainage/1e6:.3f} million m³")
    
    with col2:
        st.markdown("**Balance:**")
        st.write(f"💧 Current Volume: {mb.current_volume/1e6:.3f} million m³")
        st.write(f"📊 Expected Volume: {mb.expected_volume/1e6:.3f} million m³")
        
        error_pct = mb.relative_error * 100
        if error_pct < 1:
            st.success(f"✅ Balance Error: {error_pct:.2f}%")
        elif error_pct < 5:
            st.warning(f"⚠️ Balance Error: {error_pct:.2f}%")
        else:
            st.error(f"❌ Balance Error: {error_pct:.2f}%")
    
    # Show convergence info
    if result.stopped_early:
        st.info(f"🎯 {result.stop_reason}")


def plot_stats_chart(result: SimulationResult):
    """Create a chart showing flood statistics over time."""
    if not result.stats_history:
        return None
    
    times = [s['time'] for s in result.stats_history]
    depths = [s['max_depth'] for s in result.stats_history]
    areas = [s['flooded_area_km2'] for s in result.stats_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # Max depth over time
    ax1.fill_between(times, depths, alpha=0.3, color='#3498db')
    ax1.plot(times, depths, color='#2980b9', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Max Water Depth (m)')
    ax1.set_title('Maximum Flood Depth')
    ax1.grid(True, alpha=0.3)
    
    # Mark rainfall duration
    if result.params.rainfall_duration < max(times):
        ax1.axvline(x=result.params.rainfall_duration, color='#e74c3c', 
                   linestyle='--', alpha=0.7, label='Rain stops')
        ax1.legend(loc='upper right')
    
    # Flooded area over time
    ax2.fill_between(times, areas, alpha=0.3, color='#e74c3c')
    ax2.plot(times, areas, color='#c0392b', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Flooded Area (km²)')
    ax2.set_title('Flooded Area')
    ax2.grid(True, alpha=0.3)
    
    if result.params.rainfall_duration < max(times):
        ax2.axvline(x=result.params.rainfall_duration, color='#e74c3c', 
                   linestyle='--', alpha=0.7, label='Rain stops')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_water_balance_chart(result: SimulationResult):
    """Create a chart showing water balance over time."""
    if not result.stats_history:
        return None
    
    times = [s['time'] for s in result.stats_history]
    volumes = [s['total_volume_m3']/1e6 for s in result.stats_history]
    
    # Calculate cumulative values
    rain_cum = []
    infil_cum = []
    drain_cum = []
    
    rain_total = 0
    infil_total = 0
    drain_total = 0
    
    for s in result.stats_history:
        rain_total += s.get('rainfall_volume_m3', 0) / 1e6
        infil_total += s.get('infiltration_volume_m3', 0) / 1e6
        drain_total += s.get('edge_drainage_volume_m3', 0) / 1e6
        rain_cum.append(rain_total)
        infil_cum.append(infil_total)
        drain_cum.append(drain_total)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.plot(times, rain_cum, label='Cumulative Rainfall', color='#3498db', linewidth=2)
    ax.plot(times, volumes, label='Water in Domain', color='#2ecc71', linewidth=2)
    ax.plot(times, infil_cum, label='Infiltration Loss', color='#9b59b6', linewidth=2, linestyle='--')
    ax.plot(times, drain_cum, label='Edge Drainage', color='#e67e22', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Volume (million m³)')
    ax.set_title('Water Balance Over Time')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    # Header with modern styling
    st.markdown('''
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 class="main-header">🌊 FloodSim Sandbox</h1>
        <p class="sub-header">Interactive flood simulation using real-world terrain data</p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                🔬 Physics-based simulation
            </span>
            <span style="background: #dcfce7; color: #166534; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                🗺️ Real terrain data
            </span>
            <span style="background: #fef3c7; color: #92400e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                🎬 Animated results
            </span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar - Controls
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        
        # Data source selection
        st.subheader("📂 Terrain Data")
        
        data_source = st.radio(
            "Select data source:",
            ["Sample DEMs", "Upload File"],
            horizontal=True
        )
        
        terrain_loaded = False
        
        if data_source == "Sample DEMs":
            sample_files = get_sample_files()
            if sample_files:
                selected_file = st.selectbox(
                    "Choose a sample DEM:",
                    sample_files,
                    format_func=lambda x: x.name
                )
                
                max_size = st.slider(
                    "Max resolution (for performance)",
                    min_value=100,
                    max_value=1000,
                    value=300,
                    step=50
                )
                
                if st.button("📥 Load Terrain", use_container_width=True):
                    with st.spinner("Loading DEM..."):
                        try:
                            st.session_state.terrain = load_dem(
                                str(selected_file),
                                max_size=max_size
                            )
                            st.session_state.simulation_result = None
                            terrain_loaded = True
                            st.success("✅ Terrain loaded successfully!")
                        except Exception as e:
                            st.error(f"Error loading DEM: {e}")
            else:
                st.warning("No sample DEMs found in ./data folder")
        
        else:  # Upload file
            uploaded_file = st.file_uploader(
                "Upload GeoTIFF",
                type=['tif', 'tiff'],
                help="Upload a DEM in GeoTIFF format"
            )
            
            if uploaded_file:
                max_size = st.slider(
                    "Max resolution",
                    min_value=100,
                    max_value=500,
                    value=200,
                    step=50
                )
                
                if st.button("📥 Load Uploaded DEM", use_container_width=True):
                    with st.spinner("Processing uploaded file..."):
                        try:
                            # Save to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = tmp.name
                            
                            st.session_state.terrain = load_dem(tmp_path, max_size=max_size)
                            st.session_state.simulation_result = None
                            os.unlink(tmp_path)
                            st.success("✅ Terrain loaded!")
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        st.divider()
        
        # Solver selection
        st.subheader("🔧 Solver")
        solver_type = st.radio(
            "Simulation backend:",
            ["Fast (NumPy)", "Realistic (Shallow Water)"],
            index=0,
            help="**Fast (NumPy):** Simplified diffusion model for quick iteration.\n\n"
                 "**Realistic (Shallow Water):** Full 2D shallow water equations with "
                 "Manning's friction. More accurate but slower."
        )
        
        if solver_type == "Realistic (Shallow Water)":
            st.info("ℹ️ Using shallow water equations solver (slower but more realistic)")
        
        st.divider()
        
        # Simulation parameters
        st.subheader("🌧️ Rainfall Settings")
        
        rainfall_intensity = st.slider(
            "Rainfall Intensity (mm/hr)",
            min_value=5,
            max_value=200,
            value=50,
            step=5,
            help="Typical values:\n"
                 "• Light rain: 5-10 mm/hr\n"
                 "• Moderate rain: 10-25 mm/hr\n"
                 "• Heavy rain: 25-50 mm/hr\n"
                 "• Very heavy: 50-100 mm/hr\n"
                 "• Extreme: >100 mm/hr"
        )
        
        rainfall_duration = st.slider(
            "Rainfall Duration (hours)",
            min_value=0.5,
            max_value=12.0,
            value=2.0,
            step=0.5,
            help="How long the rain event lasts"
        )
        
        st.subheader("⚡ Simulation Settings")
        
        sim_duration = st.slider(
            "Total Simulation Time (hours)",
            min_value=1.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
            help="Total time to simulate (including after rain stops)"
        )
        
        diffusion_coef = st.slider(
            "Flow Speed (m²/s)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Controls how fast water spreads across terrain.\n"
                 "Higher = faster flow, lower = slower pooling.\n"
                 "Typical: 0.5-2.0 m²/s"
        )
        
        st.subheader("🌱 Soil & Drainage")
        
        infiltration_rate = st.slider(
            "Soil Infiltration Rate (mm/hr)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Rate at which water drains into soil:\n"
                 "• Clay soil: 1-5 mm/hr\n"
                 "• Loam soil: 10-25 mm/hr\n"
                 "• Sandy soil: 25-50 mm/hr\n"
                 "• Urban/paved: 0-2 mm/hr"
        )
        
        evaporation_rate = st.slider(
            "Evaporation Rate (mm/hr)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Water loss due to evaporation.\n"
                 "Typically 0-5 mm/hr depending on temperature."
        )
        
        # Advanced settings
        with st.expander("🔬 Advanced Settings"):
            timestep = st.select_slider(
                "Time Step (hours)",
                options=[0.01, 0.02, 0.05, 0.1, 0.2],
                value=0.05,
                help="Smaller = more accurate but slower.\n"
                     "0.05 hrs (3 min) is a good balance."
            )
            
            st.markdown("**Edge Drainage**")
            enable_edge_drainage = st.checkbox(
                "Enable edge drainage",
                value=True,
                help="Allow water to exit at domain boundaries.\n"
                     "Prevents unrealistic water buildup."
            )
            
            edge_drainage_rate = st.slider(
                "Edge drain rate (fraction/step)",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Fraction of water removed at edges per step.\n"
                     "Higher = faster drainage to outside areas.",
                disabled=not enable_edge_drainage
            )
            
            st.markdown("**Convergence**")
            enable_early_stop = st.checkbox(
                "Enable early stopping",
                value=True,
                help="Stop simulation when flood stabilizes"
            )
            
            convergence_steps = st.slider(
                "Stability steps",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Stop if max depth stable for this many steps",
                disabled=not enable_early_stop
            )
        
        st.divider()
        
        # Run simulation button
        run_disabled = st.session_state.terrain is None
        
        if st.button(
            "🚀 Run Simulation",
            use_container_width=True,
            disabled=run_disabled,
            type="primary"
        ):
            terrain = st.session_state.terrain
            
            params = SimulationParams(
                rainfall_intensity=rainfall_intensity,
                rainfall_duration=rainfall_duration,
                timestep=timestep,
                diffusion_coef=diffusion_coef,
                infiltration_rate=infiltration_rate,
                evaporation_rate=evaporation_rate,
                edge_drainage_rate=edge_drainage_rate if enable_edge_drainage else 0.0,
                enable_edge_drainage=enable_edge_drainage,
                convergence_steps=convergence_steps if enable_early_stop else 0
            )
            
            # Select solver based on user choice
            if solver_type == "Realistic (Shallow Water)":
                from simulation_core import RealisticFloodModel
                simulator = RealisticFloodModel(terrain, params)
            else:
                simulator = FloodSimulator(terrain, params)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(step, total, stats):
                progress = step / total
                progress_bar.progress(progress)
                status_text.text(
                    f"Step {step}/{total} | "
                    f"Time: {stats['time']:.1f}h | "
                    f"Max depth: {stats['max_depth']:.2f}m | "
                    f"Area: {stats['flooded_area_km2']:.2f} km²"
                )
            
            result = simulator.run(
                duration=sim_duration,
                record_interval=max(1, int(0.1 / timestep)),  # Record ~every 0.1 hour
                progress_callback=update_progress
            )
            
            st.session_state.simulation_result = result
            st.session_state.current_frame = len(result.depth_history) - 1
            
            # Auto-generate video after simulation
            progress_bar.progress(1.0)
            status_text.text("🎬 Generating animation...")
            
            video_bytes = create_simulation_video(
                terrain,
                result.depth_history,
                result.time_history,
                fps=8,
                use_discrete_colors=True
            )
            st.session_state.video_bytes = video_bytes
            
            progress_bar.empty()
            status_text.empty()
            
            if result.stopped_early:
                st.success(f"✅ Simulation converged! {result.stop_reason}")
            else:
                st.success("✅ Simulation complete! Video generated automatically.")
        
        if run_disabled:
            st.info("👆 Load terrain data first")
    
    # Main content area
    if st.session_state.terrain is not None:
        terrain = st.session_state.terrain
        result = st.session_state.simulation_result
        
        # Terrain info
        stats = get_terrain_stats(terrain)
        
        with st.expander("📊 Terrain Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Grid Size", f"{stats['shape'][0]} × {stats['shape'][1]}")
                st.metric("Min Elevation", f"{stats['min_elevation']:.1f} m")
            with col2:
                st.metric("Resolution", f"{stats['resolution'][0]:.1f} m")
                st.metric("Max Elevation", f"{stats['max_elevation']:.1f} m")
            with col3:
                st.metric("Area", f"{stats['area_km2']:.1f} km²")
                st.metric("CRS", stats['crs'][:20])
        
        # Show video prominently if simulation has been run
        if result is not None and 'video_bytes' in st.session_state and st.session_state.video_bytes is not None:
            # Display metrics at the top
            display_metrics(result, len(result.depth_history) - 1)
            
            # Show the animation video
            display_video_inline(st.session_state.video_bytes)
            
            # Download button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.download_button(
                    label="⬇️ Download Animation (GIF)",
                    data=st.session_state.video_bytes,
                    file_name="flood_simulation.gif",
                    mime="image/gif",
                    use_container_width=True
                )
            
            st.divider()
        
        # Tabs for different views
        if result is not None:
            tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map", "📈 Statistics", "💾 Export"])
        else:
            tab1 = st.container()
            tab2 = None
            tab3 = None
        
        # Tab 1: Interactive Map
        with tab1 if result else tab1:
            st.markdown('<p class="section-header">🗺️ Interactive Map</p>', unsafe_allow_html=True)
            
            # Time slider for animation (if simulation exists)
            water_depth = None
            if result is not None and len(result.depth_history) > 1:
                frame_idx = st.slider(
                    "⏱️ Explore Time Steps",
                    min_value=0,
                    max_value=len(result.depth_history) - 1,
                    value=st.session_state.current_frame,
                    key="frame_slider",
                    help="Drag to see flood at different times"
                )
                st.session_state.current_frame = frame_idx
                water_depth = result.depth_history[frame_idx]
                
                # Show time info
                if frame_idx < len(result.time_history):
                    st.caption(f"⏰ Time: {result.time_history[frame_idx]:.1f} hours | Max Depth: {np.max(water_depth):.2f}m")
            
            # Display options in a cleaner layout
            with st.expander("🎨 Map Display Options", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    show_terrain = st.checkbox("Show Terrain Overlay", value=False)
                    show_water = st.checkbox("Show Flood Water", value=True)
                    use_contour = st.checkbox("Terrain Contour Lines", value=False)
                with col2:
                    base_map = st.selectbox(
                        "Base Map Style",
                        ["OpenStreetMap", "CartoDB positron", "CartoDB dark"],
                        index=0
                    )
                    water_style = st.selectbox(
                        "Water Visualization",
                        ["Discrete Levels", "Contour Lines", "Continuous Gradient"],
                        index=0,
                        help="Discrete: 5 depth levels | Contour: Isolines | Continuous: Gradient"
                    )
            
            # Show depth legend for discrete colors
            if water_style == "Discrete Levels" and show_water and water_depth is not None:
                create_depth_legend()
            
            # Determine visualization settings
            use_water_contour = (water_style == "Contour Lines")
            use_discrete_colors = (water_style == "Discrete Levels")
            
            # Create and display map
            m = create_folium_map(
                terrain,
                water_depth=water_depth,
                show_terrain=show_terrain,
                show_water=show_water,
                use_contour=use_contour,
                use_water_contour=use_water_contour,
                use_discrete_colors=use_discrete_colors,
                base_map=base_map
            )
            
            st_folium(m, width=None, height=500, returned_objects=[])
        
        # Tab 2: Statistics (only shown when result exists)
        if result is not None and tab2 is not None:
            with tab2:
                st.markdown('<p class="section-header">📈 Flood Statistics</p>', unsafe_allow_html=True)
                
                if result.stats_history:
                    # Statistics charts
                    fig = plot_stats_chart(result)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Water balance chart
                    fig2 = plot_water_balance_chart(result)
                    if fig2:
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    # Mass balance summary
                    display_mass_balance(result)
        
        # Tab 3: Export (only shown when result exists)
        if result is not None and tab3 is not None:
            with tab3:
                st.markdown('<p class="section-header">💾 Export Results</p>', unsafe_allow_html=True)
                
                st.markdown("### 📥 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🖼️ Static Image**")
                    # Export as PNG
                    if st.button("📸 Generate Map Image", use_container_width=True):
                        with st.spinner("Creating image..."):
                            fig, ax = plt.subplots(figsize=(12, 10))
                            
                            terrain_img = get_elevation_colormap(terrain.elevation, colormap='YlOrBr', add_hillshade=True)
                            ax.imshow(terrain_img, extent=[terrain.bounds[0], terrain.bounds[2],
                                                            terrain.bounds[1], terrain.bounds[3]])
                            final_depth = result.final_depth
                            if final_depth is not None and np.max(final_depth) > 0.001:
                                water_img = create_water_overlay_image(final_depth)
                                ax.imshow(water_img, extent=[terrain.bounds[0], terrain.bounds[2],
                                                              terrain.bounds[1], terrain.bounds[3]])
                            
                            ax.set_title('FloodSim - Final Flood Extent', fontsize=14, fontweight='bold')
                            ax.axis('off')
                            
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            
                            st.download_button(
                                label="⬇️ Download PNG",
                                data=buf.getvalue(),
                                file_name="flood_map.png",
                                mime="image/png",
                                use_container_width=True
                            )
                
                with col2:
                    st.markdown("**📊 Depth Data**")
                    # Export depth data
                    if st.button("📊 Generate CSV Data", use_container_width=True):
                        with st.spinner("Creating CSV..."):
                            depth_flat = result.final_depth.flatten()
                            csv_data = "row,col,depth_m,lat,lon\n"
                            
                            bounds = terrain.bounds
                            rows, cols = result.final_depth.shape
                            
                            for i, depth in enumerate(depth_flat):
                                if depth > 0.001:
                                    row = i // cols
                                    col = i % cols
                                    lat = bounds[3] - (row / rows) * (bounds[3] - bounds[1])
                                    lon = bounds[0] + (col / cols) * (bounds[2] - bounds[0])
                                    csv_data += f"{row},{col},{depth:.4f},{lat:.6f},{lon:.6f}\n"
                            
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_data,
                                file_name="flood_depths.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                
                # Animation is already at the top, but add regenerate option
                st.markdown("---")
                st.markdown("### 🎬 Animation")
                st.info("💡 Animation is automatically generated after each simulation and displayed at the top of the page.")
                
                if st.button("🔄 Regenerate Animation", use_container_width=True):
                    with st.spinner("Generating new animation..."):
                        video_bytes = create_simulation_video(
                            terrain,
                            result.depth_history,
                            result.time_history,
                            fps=8,
                            use_discrete_colors=True
                        )
                        st.session_state.video_bytes = video_bytes
                        st.success("✅ Animation regenerated! Scroll to top to view.")
                        st.rerun()
    
    else:
        # Welcome screen when no terrain is loaded
        st.info("👈 Use the sidebar to load terrain data and configure the simulation.")
        
        st.markdown("""
        ### 🎯 How to Use
        
        1. **Load Terrain**: Select a sample DEM or upload your own GeoTIFF file
        2. **Configure Rainfall**: Set intensity (mm/hr) and duration (hours)
        3. **Adjust Soil Properties**: Set infiltration rate based on soil type
        4. **Run Simulation**: Click the "Run Simulation" button
        5. **Explore Results**: Use the time slider to see flood progression
        6. **Export**: Download maps and data for further analysis
        
        ### 📖 About v2.0
        
        FloodSim Sandbox v2.0 includes major stability and realism improvements:
        
        - **🔬 Physical Flow Model**: Diffusion scales with timestep and grid resolution
        - **🚿 Edge Drainage**: Water can exit at domain boundaries (realistic outflow)
        - **🌱 Soil Infiltration**: Configurable absorption into ground
        - **⚖️ Mass Balance Tracking**: Monitor water inputs and outputs
        - **🎯 Convergence Detection**: Automatic early stopping when flood stabilizes
        - **📊 Improved Diagnostics**: Charts and metrics for water balance
        
        ### 📐 Parameter Guide
        
        | Parameter | Description | Typical Values |
        |-----------|-------------|----------------|
        | Rainfall Intensity | Rain rate | 10-100 mm/hr |
        | Infiltration Rate | Soil absorption | Clay: 1-5, Sand: 25-50 mm/hr |
        | Flow Speed | Water spreading rate | 0.5-2.0 m²/s |
        | Edge Drainage | Boundary outflow | 5-20% per step |
        """)


if __name__ == "__main__":
    main()
