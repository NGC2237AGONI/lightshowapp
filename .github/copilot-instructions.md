# Copilot Instructions for FBX 3D Model Processing

## Project Overview
This is a **3D model data extraction and visualization pipeline** for FBX files. It extracts point clouds from rigged 3D models, applies sampling algorithms, samples texture colors, and visualizes results with animation support.

**Key workflow:** FBX file → Extract vertices & colors → Sampling/filtering → Visualization or CSV export

---

## Architecture & Data Flow

### Core Components

1. **FBX Data Extraction** (`parse.py`, `pointwithcolor.py`, `message.py`)
   - Load FBX files using Autodesk FBX SDK (via `fbx` Python package)
   - Traverse node hierarchy to find mesh nodes (filter by `eMesh` attribute)
   - Extract vertex positions, colors, and animations frame-by-frame
   - Apply global transformation matrices to convert local → world coordinates

2. **Data Structures** (`main.py: SampledData class`)
   - **rest_positions**: Vertex positions at rest pose (N, 3) array
   - **colors**: RGBA colors (N, 4) array
   - **animation_frames**: Time-series vertex data (Frames, N, 3)
   - **weights_info**: Skeleton bone weights (simplified in current code)

3. **Sampling & Filtering** (`pointwithcolorset.py`)
   - **Geometric feature extraction**: Uses k-nearest neighbors (KDTree) to detect curvature/edges
   - **Color-based filtering**: Identifies color discontinuities at boundaries
   - **Percentile-based ranking**: Keeps only top N% most distinctive points (avoids hard threshold guessing)

4. **Color Sources** (`texture_sampler.py`, `pointwithcolor.py`)
   - **Priority 1**: Texture colors sampled from UV-mapped diffuse/base color textures
   - **Priority 2**: Per-vertex colors from FBX vertex color layer
   - **Priority 3**: Material diffuse color (fallback)

5. **Output Formats**
   - **NPZ** (`model_data.npz`): Compressed NumPy arrays for fast reload
   - **CSV** (`drone_data.csv`): Frame-by-frame animation data for external tools

---

## Critical Implementation Details

### FBX SDK Compatibility Layer
The FBX Python package has **breaking API changes** between versions. Always use defensive enum access:
```python
try:
    eMesh = fbx.FbxNodeAttribute.eMesh  # Old API
except AttributeError:
    eMesh = fbx.FbxNodeAttribute.EType.eMesh  # New API (2020+)
```
**See:** `parse.py:get_mesh_enum_id()` and `get_skeleton_enum_id()`

### Coordinate Transformation
All positions must be transformed from local → global space using the node's transform matrix:
```python
global_transform = node.EvaluateGlobalTransform(fbx_time)
world_pos = global_transform.MultT(local_pos)  # MultT applies transformation
```
This is critical for animated meshes where nodes move relative to parents.

### Animation Timeline
- FBX files store animation in **stacks** (animation groups)
- Extract time span: `anim_stack.GetLocalTimeSpan().GetStart/Stop()`
- All time queries must use `fbx.FbxTime` objects, not raw floats
- Frame sampling: `fbx_time.SetSecondDouble(current_seconds)`

### Sampling Strategy
The `pointwithcolorset.py` pipeline:
1. Computes k-NN (k=16 default) for each point
2. Calculates **geometric curvature**: distance from point to neighbor centroid
3. Calculates **color edge strength**: color difference from neighbor average
4. Normalizes both scores to 0-1 range
5. **Keeps top P% by percentile** (not fixed threshold) → prevents over-sensitivity to outliers

**Parameter tuning:**
- `k`: Neighborhood size (12-24 recommended; larger = smoother edges)
- `keep_percentage`: Only keep top X% most distinctive points (5-20% typical)

---

## Project-Specific Conventions

### File Naming & Organization
- **Input files**: `*.fbx` (multiple test files: `1.fbx`, `2.fbx`)
- **Texture folders**: `*.fbm/` (FBX Media folders, contain referenced textures)
- **Output data**: `model_data.npz`, `drone_data.csv`
- **Processing scripts**: Standalone files, each handles one pipeline stage

### Configuration Areas
Every script has a **configuration block** at the top (marked with `# =============== 配置区域 ===============`). Always modify these instead of hardcoding values:
- `TARGET_FILE` / `INPUT_FILE`: FBX file path
- `OUTPUT_FILE`: CSV destination
- `SCALE`: Unit scaling factor
- `k`, `keep_percentage`: Sampling parameters
- `FPS`: Frame sampling rate

### Chinese Comments
The codebase uses **Chinese variable names and comments** (e.g., `提取网格数据` = "extract mesh data"). This is intentional for domain expert readability. When extending:
- Maintain bilingual comments for new functions
- Keep Chinese for domain terms (顶点 = vertex, 骨骼 = bone, 贴图 = texture)

### Visualization
- Uses `matplotlib` (3D scatter plots)
- Uses `pyqtgraph.opengl` for real-time GL rendering in `main.py`
- Default: black background, marker size 1.5-2.0 for clarity with large point clouds

---

## Common Integration Patterns

### Pattern 1: Add Custom Sampling Algorithm
- Inherit the feature extraction logic from `pointwithcolorset.py:extract_outline_points_robust()`
- Call `cKDTree.query(points, k=...)` for k-NN queries
- Normalize custom scores to 0-1 before combining
- Use percentile cutoff instead of fixed thresholds

### Pattern 2: Multi-mesh Processing
- Both `message.py` and `pointwithcolor.py` loop through all nodes
- `process_node()` recursively traverses hierarchy
- Filtering by attribute type: `attr.GetAttributeType() == eMesh`
- All meshes contribute to single output arrays

### Pattern 3: Animation Export
- Store time-indexed data in CSV with columns: Frame, Time, Object, VertexID, X, Y, Z, R, G, B
- Iterate frames: `for f in range(total_frames): current_sec = start_time + (f / FPS)`
- Always apply `EvaluateGlobalTransform(fbx_time)` for each frame

---

## Debugging & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Error: 无法打开文件" | Wrong filename or path | Check `TARGET_FILE` config; use absolute path if needed |
| Mesh not loading | FBX node has no mesh attribute | Verify FBX structure with `parse.py` → checks all attributes |
| Colors are all gray | No texture/vertex color layer | Check `texture_sampler.py` fallback logic; may need color layer in FBX |
| Black texture sampling | Texture not found in same folder | Put `*.fbm/` folder contents in working directory |
| Positions are wrong | Skipped global transformation | Verify `EvaluateGlobalTransform()` is being called |
| Too few/many outline points | `keep_percentage` wrong value | Adjust in config (try 2%-20% range) |

---

## Development Workflow

1. **Test FBX loading**: Run `parse.py` → prints node hierarchy
2. **Extract static geometry**: Run `pointwithcolor.py` → generates visualization
3. **Extract animation**: Run `message.py` → generates CSV with frame data
4. **Filter outline**: Run `pointwithcolorset.py` → applies smart sampling
5. **Advanced visualization**: Edit `main.py` for real-time GL rendering

For new features, start by extending `parse.py` (simple tree traversal) before adding complex logic.

---

## Dependencies & Environment

- **fbx**: Python FBX SDK wrapper (may need manual installation)
- **numpy**: Array operations, KDTree queries
- **scipy**: `cKDTree` for spatial indexing
- **matplotlib**: 3D visualization
- **PyQt5**: GUI framework (for `main.py`)
- **Pillow**: Texture image loading
- **pyqtgraph**: OpenGL-based visualization

See import statements for version hints (no strict pinning in current code).
