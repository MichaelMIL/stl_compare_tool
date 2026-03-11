## STL Compare Tool

Command-line utility to compare 3D meshes in two directories of STL files. It loads each STL, computes basic mesh statistics, and approximates geometric similarity using sampled surface point distances. The tool then finds best matches between directories, reports detailed metrics, and can export results as JSON and CSV.

### Features

- **Recursive STL discovery**: Scans two directories for `.stl` / `.STL` files.
- **Mesh cleanup and stats**: Computes watertightness, vertex/face counts, area, volume, bounds, extents, and centroid.
- **Approximate geometric distance**: Uses sampled surface points and nearest-neighbor distances (via `scipy.spatial.cKDTree`) to approximate symmetric Hausdorff-like metrics.
- **Automatic matching**: Pairs meshes in directory A with the closest meshes in directory B based on mean distance.
- **Filtering by size/shape**: Quickly rejects obviously different meshes using volume checks and optional max-distance threshold.
- **JSON / CSV outputs**: Human-readable console output, JSON for scripts/CI, and CSV summary of matches.

### Requirements

Python 3.10+ is recommended.

Dependencies (installable via `pip`):

- `numpy`
- `trimesh`
- `scipy`

### Installation

Clone this repository and install Python dependencies:

```bash
git clone <your-repo-url> stl_compare_tool
cd stl_compare_tool

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install numpy trimesh scipy
```

You can run the script directly:

```bash
python compare.py --help
```

Or make it executable on Unix-like systems:

```bash
chmod +x compare.py
./compare.py --help
```

### Usage

Basic usage compares all STL files in two directories:

```bash
python compare.py /path/to/dir_a /path/to/dir_b
```

Key arguments:

- `dir_a` / `dir_b`: **Required.** Directories containing STL files (scanned recursively).
- `--samples`: Number of surface sample points per mesh (default: `50000`). Higher values are slower but more accurate.
- `--seed`: Random seed for repeatable sampling (default: `123`).
- `--center`: Center meshes at their centroids before computing distances.
- `--max-distance`: Maximum allowed mean distance for a match. Pairs with larger distances are discarded.
- `--json`: Output all results as JSON (printed to stdout).
- `--csv <path>`: Write a CSV summary of matched and unmatched files to the given file.

Example with JSON output and CSV summary:

```bash
python compare.py /data/stl_old /data/stl_new \
  --samples 75000 \
  --center \
  --max-distance 0.5 \
  --json \
  --csv compare_results.csv
```

### Output

- **Console (default)**: For each matched pair, prints:
  - mesh A stats (JSON)
  - mesh B stats (JSON)
  - stat differences (areas, volumes, extents, centroid differences, etc.)
  - shape distance metrics (`mean`, `rms`, `max`, `p95`, `p99`)
- **JSON (`--json`)**: Single JSON object (one pair) or array of objects (multiple pairs), suitable for automated checks or CI.
- **CSV (`--csv`)**: Rows with `File A`, `File B`, and `Distance` for matches, and `None` for unmatched files from directory A.

Distances are reported in the STL units (often millimeters); exact units depend on how the source meshes were authored.

### Notes and Limitations

- This tool approximates geometric differences via sampling; it is not an exact Hausdorff distance.
- Volume-based pre-filtering is only applied when both meshes are watertight and have valid volumes.
- Very large meshes or high `--samples` values will increase computation time and memory usage.

### License

Add your preferred license information here.

