# 🌊 **FloodSim Sandbox — Implementation Plan**

1. **Input:** Use a real-world topographic map in **GeoTIFF (.tif)** format.
2. **Output:** Draw results directly **on the map** — overlay flood depth, contours, and affected zones interactively.

---

## 🧭 Overview

This improved version transforms the project from a synthetic demo into a **semi-realistic flood visualization platform**:

* Load **Digital Elevation Model (DEM)** data from `.tif` files.
* Simulate rainfall-driven flooding dynamically over real terrain.
* Render results as interactive overlays on geographic maps.
* Optionally let users draw on the map (e.g., dam locations or rainfall zones).

---

## 🧱 System Architecture (Enhanced)

### 1. **Terrain Loader**

| Feature       | Description                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------- |
| **Input**     | GeoTIFF file (.tif) representing elevation (DEM).                                           |
| **Libraries** | `rasterio`, `numpy`, `matplotlib`, `streamlit-folium` or `folium`.                          |
| **Output**    | 2D NumPy array of elevations, with geographic bounds and coordinate reference system (CRS). |

**Key Improvements**

* Normalize elevation data for numerical stability.
* Handle missing data (NaN fill/interpolation).
* Downsample large DEMs for performance.

**Optional Add-ons**

* Provide a few sample `.tif` DEMs (e.g., SRTM or USGS data).
* Enable clipping to a region of interest using bounding boxes.

---

### 2. **Simulation Core**

Uses the same “height-difference” approach but now works in **real-world units (meters)**.

| Step                      | Description                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Initialization**        | Load terrain grid; create zero-initialized water layer.                                                     |
| **Rainfall Input**        | Add rainfall (mm/hour) over selected or all cells.                                                          |
| **Flow Model**            | Move water between neighboring cells based on terrain + water surface difference.                           |
| **Dams & Drawn Features** | Let user *draw* polygons or lines on the map to define barriers/dams. These cells get temporarily elevated. |
| **Update Loop**           | Iterate timesteps, compute inflows/outflows, recalculate equilibrium.                                       |
| **Output Layers**         | Water depth, flood extent (binary mask), total flooded area.                                                |

**Algorithm Enhancements**

* Add a **flow diffusion coefficient** (tunable).
* Use an **explicit timestep limiter** to prevent instability.
* Apply **smoothing filter** periodically for visual realism.

---

### 3. **Visualization & Map Output**

| Goal              | Description                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **Base Map**      | Display original terrain using color shading (elevation-based colormap).                                    |
| **Flood Overlay** | Semi-transparent blue overlay for water depth, dynamically updated.                                         |
| **Drawing Tools** | Users can **draw** on the map (e.g., dam line, rainfall region) using Streamlit’s `st_folium` or `leafmap`. |
| **Animation**     | Animate flood spread over time (slider or auto-play).                                                       |
| **Metrics Panel** | Show flood stats (area, max depth, duration).                                                               |

**Libraries**

* `folium` or `leafmap` for geospatial interactivity.
* `matplotlib` or `plotly` for dynamic graphs.
* `streamlit-folium` for embedding interactive maps in Streamlit.

---

### 4. **User Interface (Streamlit-based)**

| Component            | Function                                                       |
| -------------------- | -------------------------------------------------------------- |
| File uploader        | Upload `.tif` DEM file.                                        |
| Rainfall controls    | Sliders for intensity (mm/hr) and duration (hours).            |
| Dam toggle           | Checkbox or map-drawn feature to create dam barrier.           |
| Simulation control   | “Run Simulation” button + timestep slider.                     |
| Visualization toggle | Switch between elevation, flood depth, and flood extent views. |
| Export options       | Download flooded map as GeoTIFF or PNG.                        |

---

## ⚙️ Implementation Timeline (Revised 4-Hour Breakdown)

### **Hour 1: Setup & DEM Integration**

**Objectives**

* Environment setup and basic DEM handling.

**Tasks**

1. Initialize project and dependencies:

   ```bash
   pip install numpy matplotlib rasterio streamlit folium streamlit-folium
   ```
2. Create `.py` modules:

   * `terrain_loader.py` — handles DEM read, normalization, and visualization.
   * `simulation_core.py` — manages flooding algorithm.
   * `app.py` — Streamlit UI integration.
3. Load sample `.tif` and visualize elevation as a color map overlay on a map.

**Deliverable**

* Interactive map showing terrain elevation loaded from `.tif`.

---

### **Hour 2: Simulation Engine + Dam Logic**

**Objectives**

* Implement rainfall and water flow simulation.

**Tasks**

1. Implement core loop:

   ```python
   water = np.zeros_like(terrain)
   for step in range(steps):
       water += rainfall_rate
       flow = compute_flow(terrain, water)
       water += flow
   ```
2. Add dam logic:

   * When a dam polygon is drawn, locally increase terrain elevation.
   * Allow timed “dam break” to simulate breach.
3. Output intermediate flood layers.

**Deliverable**

* Working simulation on real DEM with basic rainfall and dam interaction.

---

### **Hour 3: Visualization Layer & Interactivity**

**Objectives**

* Display flood spread directly on the map.

**Tasks**

1. Overlay water depth on base map:

   ```python
   folium.raster_layers.ImageOverlay(...)
   ```
2. Add Streamlit-Folium integration for live controls:

   * Sliders update rainfall and timestep.
   * Map updates dynamically with flood progression.
3. Enable drawing tools for user-defined barriers/dams.

**Deliverable**

* Interactive Streamlit dashboard showing flood evolution.

---

### **Hour 4: Refinement & Output**

**Objectives**

* Polish visuals, export results, and prepare for presentation.

**Tasks**

1. Smooth water animation (interpolate between timesteps).
2. Add legend, colorbar, and metrics panel.
3. Export final flood extent as `.png` and `.tif`.
4. Add summary card (terrain name, rain params, flood area).

**Deliverable**

* Presentation-ready interactive flood simulator using real-world topography.

---

## 📊 Outputs

| Type                    | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| **Animated map**        | Flood propagation over real terrain (blue overlay).      |
| **Flood extent mask**   | Binary flooded/non-flooded raster.                       |
| **Statistical summary** | Total flooded area, max water depth, dam breach effects. |
| **Export files**        | GeoTIFF or PNG of final flood map.                       |

---

## 🔬 Version 2.0 - Stability & Realism Improvements

The following improvements address issues with unrealistic flood depths, water accumulation, and simulation behavior:

### Physical Flow Model
- **Timestep-scaled diffusion**: Flow rate now computed as `effective_k = D * dt / dx²`
- **Grid-aware scaling**: Automatically adapts to DEM resolution
- **Stability checks**: Warns if parameters exceed numerical stability limits

### Water Balance & Drainage
- **Edge drainage**: Water exits at domain boundaries (configurable rate)
- **Soil infiltration**: Non-zero default (5 mm/hr) for realistic absorption
- **Evaporation support**: Optional atmospheric water loss
- **Mass balance tracking**: Monitors inputs vs outputs with error reporting

### Convergence & Stability
- **Early stopping**: Simulation ends when max depth stabilizes
- **Convergence detection**: Configurable threshold and stability window
- **Per-cell dam heights**: Dam reset preserves individual heights

### Pluggable Architecture
- **Backend interface**: `BaseFloodModel` abstract class for solver plugins
- **Fast solver**: NumPy-based diffusion (default)
- **Realistic solver**: Placeholder for LISFLOOD-FP/ANUGA integration

### UI Improvements
- **Clear units**: All parameters labeled with units (mm/hr, hours, m²/s)
- **Advanced settings**: Edge drainage, convergence controls
- **Water balance display**: Charts and metrics for mass balance
- **Soil type guide**: Help text with infiltration rates by soil type

---

## 🚀 Optional Enhancements (Post-hackathon)

1. **3D Terrain Visualization** — integrate with PyVista or Plotly 3D.
2. **Real Rainfall Integration** — pull rainfall data from OpenWeather or NOAA APIs.
3. **GPU Acceleration** — use CuPy for faster diffusion steps.
4. **Hydrological Accuracy** — incorporate Manning's equation or slope-dependent flow.
5. **Multi-dam or levee system modeling.**
6. **LISFLOOD-FP/ANUGA Integration** — full hydrodynamic solver backend.

---

## 📦 Final Deliverables

1. **Streamlit App:** Full interactive flood simulator.
2. **Animated Map:** Water overlay evolving over time.
3. **User Controls:** Rainfall, terrain selection, and dam manipulation.
4. **Exportable Outputs:** Flood depth and extent maps.
5. **Demo Deck:** Screenshots, parameter table, and visual summary.

---

## ✅ Assumptions & Constraints

* DEM units are in meters and GeoTIFFs provide a valid CRS (e.g., EPSG:4326 or a suitable projected CRS).
* Time is discretized into fixed timesteps (e.g., \(\Delta t = 1\)–5 minutes) with a simple, explicit update rule.
* Flow is local (4- or 8-neighborhood) and ignores full hydrodynamic effects (no momentum equation, no channel routing).
* Performance target: runs interactively on a laptop CPU on DEMs up to ~1000×1000 cells (downsample if larger).
* Numerical safety: apply a global cap on per-step water movement to avoid creating negative depths.

---

## 🧩 Minimal Viable Version (MVP)

1. Load a single DEM `.tif` (no clipping, no multi-file mosaics).
2. Uniform rainfall over the whole domain (no spatially varying storms yet).
3. Single dam line drawn by the user; elevation boost is constant along the dam.
4. Simple nearest-neighbor flow using a diffusion-like scheme with a single tunable coefficient.
5. One main visualization mode: elevation + semi-transparent water depth overlay.
6. Basic metrics: max depth, flooded area, and total volume only.

---

## 📝 Implementation Notes

* Keep all core logic in pure functions (`terrain_loader.py`, `simulation_core.py`) so `app.py` stays thin.
* Add a small configuration section or file (e.g., timestep, diffusion coefficient, max iterations) to avoid magic numbers.
* Start with a small cropped DEM (e.g., a 200×200 window from `daklak.tif` or `lamdong.tif`) for fast iteration.
* Log or cache intermediate water-depth fields for quick replay during visualization and debugging.

