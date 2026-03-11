#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class MeshStats:
    path: str
    watertight: bool
    vertices: int
    faces: int
    area: float
    volume: float | None
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    extents: Tuple[float, float, float]
    centroid: Tuple[float, float, float]


def load_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load_mesh(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        # trimesh may return a Scene if STL contains multiple parts
        if hasattr(m, "dump"):
            parts = m.dump()
            if len(parts) == 0:
                raise ValueError(f"No mesh geometry found in: {path}")
            m = trimesh.util.concatenate(parts)
        else:
            raise ValueError(f"Unsupported mesh type from file: {path}")
    # Clean-up (lightweight, safe)
    m.update_faces(m.unique_faces())
    # Remove degenerate faces (zero or near-zero area)
    face_areas = m.area_faces
    valid_faces = face_areas > 1e-10
    m.update_faces(valid_faces)
    m.remove_unreferenced_vertices()
    return m


def mesh_stats(mesh: trimesh.Trimesh, path: str) -> MeshStats:
    bounds = mesh.bounds
    watertight = bool(mesh.is_watertight)
    # Volume is only meaningful if watertight
    volume = float(mesh.volume) if watertight else None

    c = mesh.centroid
    return MeshStats(
        path=path,
        watertight=watertight,
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        area=float(mesh.area),
        volume=volume,
        bounds_min=(float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[0, 2])),
        bounds_max=(float(bounds[1, 0]), float(bounds[1, 1]), float(bounds[1, 2])),
        extents=(float(mesh.extents[0]), float(mesh.extents[1]), float(mesh.extents[2])),
        centroid=(float(c[0]), float(c[1]), float(c[2])),
    )


def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    return m


def sample_points(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # trimesh.sample.sample_surface uses random internally; we can shuffle faces by seed via RNG
    # easiest: call sample_surface and then jitter order deterministically
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    # deterministic shuffle for repeatability
    idx = rng.permutation(len(pts))
    return pts[idx]


def nn_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    For each point in a, compute distance to nearest point in b.
    """
    tree = cKDTree(b)
    dists, _ = tree.query(a, k=1, workers=-1)
    return dists


def diff_metrics(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, samples: int, seed: int) -> Dict[str, Any]:
    """
    Symmetric point-to-point distance metrics using surface sampling.
    Approximates Hausdorff via percentile/max on sampled points.
    """
    a_pts = sample_points(mesh_a, samples, seed=seed)
    b_pts = sample_points(mesh_b, samples, seed=seed + 1)

    a_to_b = nn_distances(a_pts, b_pts)
    b_to_a = nn_distances(b_pts, a_pts)

    sym = np.concatenate([a_to_b, b_to_a])

    def summarize(x: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(x)),
            "rms": float(np.sqrt(np.mean(x * x))),
            "max": float(np.max(x)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
        }

    return {
        "a_to_b": summarize(a_to_b),
        "b_to_a": summarize(b_to_a),
        "symmetric": summarize(sym),
        "samples": int(samples),
        "note": "Distances are in the STL's units (often mm). Metrics are approximate due to sampling.",
    }


def find_stl_files(directory: str) -> Dict[str, Path]:
    """
    Find all STL files in a directory (recursively) and return a dict mapping
    filename (without path) to full path.
    """
    stl_files = {}
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Find both .stl and .STL files (case-insensitive)
    for stl_path in dir_path.rglob("*.stl"):
        filename = stl_path.name
        if filename in stl_files:
            # If duplicate filenames exist, keep the first one found
            # Could also raise an error or use full relative path as key
            continue
        stl_files[filename] = stl_path
    for stl_path in dir_path.rglob("*.STL"):
        filename = stl_path.name
        if filename in stl_files:
            # If duplicate filenames exist, keep the first one found
            continue
        stl_files[filename] = stl_path
    
    return stl_files


def find_matching_pairs_by_geometry(dir_a: str, dir_b: str, samples: int, seed: int, center: bool, max_distance: float = None) -> Tuple[List[Tuple[Path, Path, float]], List[Path]]:
    """
    Find matching STL file pairs between two directories based on geometric similarity.
    Compares each file in dir_a with all files in dir_b and finds best matches.
    Returns tuple of:
    - List of matched pairs: (path_in_dir_a, path_in_dir_b, distance_score)
    - List of unmatched files from dir_a: (path_in_dir_a,)
    """
    stl_a = find_stl_files(dir_a)
    stl_b = find_stl_files(dir_b)
    
    if len(stl_a) == 0:
        raise ValueError(f"No STL files found in directory A: {dir_a}")
    if len(stl_b) == 0:
        raise ValueError(f"No STL files found in directory B: {dir_b}")
    
    # Load all meshes from dir_a
    meshes_a = {}
    stats_a = {}
    for filename, path in stl_a.items():
        try:
            mesh = load_mesh(str(path))
            meshes_a[path] = center_mesh(mesh) if center else mesh
            stats_a[path] = mesh_stats(mesh, str(path))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
            continue
    
    # Load all meshes from dir_b
    meshes_b = {}
    stats_b = {}
    for filename, path in stl_b.items():
        try:
            mesh = load_mesh(str(path))
            meshes_b[path] = center_mesh(mesh) if center else mesh
            stats_b[path] = mesh_stats(mesh, str(path))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
            continue
    
    matches = []
    matched_files_a = set()
    
    # For each mesh in dir_a, find the best match in dir_b
    for path_a, mesh_a in meshes_a.items():
        best_match = None
        best_distance = float('inf')
        
        for path_b, mesh_b in meshes_b.items():
            try:
                # Quick pre-filter: check if volumes/areas are very different
                sa = stats_a[path_a]
                sb = stats_b[path_b]
                
                # Skip if volumes differ by more than 50% (if both are watertight)
                if sa.watertight and sb.watertight and sa.volume and sb.volume:
                    vol_ratio = min(sa.volume, sb.volume) / max(sa.volume, sb.volume)
                    if vol_ratio < 0.5:
                        continue
                
                # Compute shape distance
                shape_diff = diff_metrics(mesh_a, mesh_b, samples=samples, seed=seed)
                distance = shape_diff["symmetric"]["mean"]
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = path_b
                    
            except Exception as e:
                print(f"Warning: Failed to compare {path_a} with {path_b}: {e}", file=sys.stderr)
                continue
        
        if best_match is not None:
            if max_distance is None or best_distance <= max_distance:
                matches.append((path_a, best_match, best_distance))
                matched_files_a.add(path_a)
    
    # Find unmatched files from dir_a
    unmatched = [path_a for path_a in meshes_a.keys() if path_a not in matched_files_a]
    
    # Sort matches by distance (best matches first)
    matches.sort(key=lambda x: x[2])
    
    return matches, unmatched


def compare_stats(sa: MeshStats, sb: MeshStats) -> Dict[str, Any]:
    def rel(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        denom = max(abs(a), abs(b), 1e-12)
        return float((a - b) / denom)

    def absdiff(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return float(a - b)

    return {
        "area_abs": absdiff(sa.area, sb.area),
        "area_rel": rel(sa.area, sb.area),
        "volume_abs": absdiff(sa.volume, sb.volume),
        "volume_rel": rel(sa.volume, sb.volume),
        "extents_abs": tuple(float(a - b) for a, b in zip(sa.extents, sb.extents)),
        "centroid_abs": tuple(float(a - b) for a, b in zip(sa.centroid, sb.centroid)),
        "watertight_mismatch": bool(sa.watertight != sb.watertight),
        "vertex_diff": int(sa.vertices - sb.vertices),
        "face_diff": int(sa.faces - sb.faces),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare STL files between two directories (stats + approximate shape distance).")
    ap.add_argument("dir_a", help="Path to directory A (containing STL files)")
    ap.add_argument("dir_b", help="Path to directory B (containing STL files)")
    ap.add_argument("--samples", type=int, default=50000, help="Surface samples per mesh (default: 50000)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for repeatability")
    ap.add_argument("--center", action="store_true", help="Center both meshes at their centroids before comparing distances")
    ap.add_argument("--max-distance", type=float, default=None, help="Maximum mean distance threshold for matching (default: no limit)")
    ap.add_argument("--json", action="store_true", help="Output JSON (useful for CI/scripts)")
    ap.add_argument("--csv", type=str, default=None, help="Output CSV file with matching pairs and distances")
    args = ap.parse_args()

    try:
        # Find matching pairs by geometry
        if not args.json:
            print("Comparing STL files by geometry (this may take a while)...\n")
        
        pairs, unmatched = find_matching_pairs_by_geometry(
            args.dir_a, args.dir_b, 
            samples=args.samples, 
            seed=args.seed, 
            center=args.center,
            max_distance=args.max_distance
        )
        
        if len(pairs) == 0 and len(unmatched) == 0:
            print(f"No STL files found to compare between {args.dir_a} and {args.dir_b}", file=sys.stderr)
            return 1
        
        if not args.json:
            print(f"Found {len(pairs)} matching STL file pair(s)")
            if len(unmatched) > 0:
                print(f"Found {len(unmatched)} unmatched file(s) in directory A\n")
            else:
                print()
        
        all_results = []
        
        for path_a, path_b, distance in pairs:
            if not args.json:
                print(f"{'='*60}")
                print(f"Match (distance: {distance:.6f})")
                print(f"  A: {path_a}")
                print(f"  B: {path_b}")
                print(f"{'='*60}\n")
            
            # Reload meshes for detailed comparison (they were already loaded but we need fresh stats)
            ma = load_mesh(str(path_a))
            mb = load_mesh(str(path_b))

            sa = mesh_stats(ma, str(path_a))
            sb = mesh_stats(mb, str(path_b))

            ma_cmp = center_mesh(ma) if args.center else ma
            mb_cmp = center_mesh(mb) if args.center else mb

            stats_diff = compare_stats(sa, sb)
            shape_diff = diff_metrics(ma_cmp, mb_cmp, samples=args.samples, seed=args.seed)

            report = {
                "file_a": str(path_a),
                "file_b": str(path_b),
                "match_distance": float(distance),
                "mesh_a": asdict(sa),
                "mesh_b": asdict(sb),
                "stats_diff": stats_diff,
                "shape_diff": shape_diff,
                "centered_for_distance": bool(args.center),
            }
            
            all_results.append(report)

            if not args.json:
                print("=== Mesh A ===")
                print(json.dumps(asdict(sa), indent=2))
                print("\n=== Mesh B ===")
                print(json.dumps(asdict(sb), indent=2))
                print("\n=== Stat Differences (A - B) ===")
                print(json.dumps(stats_diff, indent=2))
                print("\n=== Shape Distance (approx, sampled) ===")
                print(json.dumps(shape_diff, indent=2))
                print("\n")
        
        # Output JSON once at the end (single object if one pair, array if multiple)
        if args.json:
            if len(all_results) == 1:
                print(json.dumps(all_results[0], indent=2))
            else:
                print(json.dumps(all_results, indent=2))
        
        # Output CSV file with matching pairs and distances
        if args.csv:
            with open(args.csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['File A', 'File B', 'Distance'])
                # Write matched pairs
                for path_a, path_b, distance in pairs:
                    writer.writerow([str(path_a), str(path_b), f'{distance:.6f}'])
                # Write unmatched files
                for path_a in unmatched:
                    writer.writerow([str(path_a), 'None', 'None'])
            if not args.json:
                print(f"CSV output written to: {args.csv}\n")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
