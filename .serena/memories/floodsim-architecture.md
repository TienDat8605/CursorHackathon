# FloodSim Sandbox Architecture

## Overview
Streamlit-based flood simulation app using real-world GeoTIFF DEM data.

## Key Files
- `app.py` - Main Streamlit UI with folium map integration
- `terrain_loader.py` - DEM loading, normalization, visualization (earth tone colors)
- `simulation_core.py` - Flood simulation engine with rainfall, infiltration, evaporation

## Key Functions (app.py)
- `create_water_overlay_image()` - Deep blue water visualization
- `create_terrain_overlay()` - Earth tone terrain (YlOrBr colormap), supports contour view
- `create_contour_overlay()` - Contour line visualization
- `create_folium_map()` - Interactive map with OpenStreetMap base option
- `create_simulation_video()` - GIF animation export

## SimulationParams (simulation_core.py)
- rainfall_intensity (mm/hr)
- rainfall_duration (hours)
- timestep, diffusion_coef
- infiltration_rate (soil drainage)
- evaporation_rate

## UI Features
- Sample DEM selection or file upload
- Rainfall + soil drainage controls
- Base map selection (OpenStreetMap, CartoDB)
- Contour view toggle
- Time slider animation
- Export: PNG, CSV, GIF video
