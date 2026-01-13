import igl
from line_profiler import profile
import numpy as np
import meshio
import pandas as pd
import json
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

use_pymeme = True
# try to import pymeme
if use_pymeme:
    try:
        import pymeme

        use_pymeme = True
        print("Using pymeme for metric computation.")
    except ImportError:
        use_pymeme = False
        print("WARNING: pymeme not found, using Python implementation.")
        pass
else:
    print("Using Python implementation for metric computation.")

# import cProfile
# import re

"""
Get mesh quality metrics for triangular meshes stored as .vtu files.

Input: path to a .vtu file or a folder with .vtu files
Output: CSV file with mesh quality metrics for each mesh. It can be stored as .csv or .zip to save disk space.

Example usage:
    python get_metrics.py -i path/to/mesh_or_folder -o output_metrics.csv -j 4

Arguments:
    -i: path to the mesh file or folder with mesh files (default: current directory)
    -o: output CSV file to save the metrics (default: mesh_quality_metrics.csv)
    -j: number of parallel jobs to use (default: 1)
    --extension: file extension of the mesh files (default: .vtu). Ignored if a single file is provided.

For running on Greene Cluster:
    module load anaconda3/2024.02
    conda create -p ./penv python=3.10
    conda activate ./penv
    conda install pip
    python3 -m pip install libigl
    pip install pandas meshio

Installing pymeme (optional):
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    copy app/pymeme... to the project folder

"""


def get_metric_names():
    if use_pymeme:
        return pymeme.get_metric_names()
    else:
        return [
            "min_angle",
            "max_angle",
            "avg_angle",
            "min_radius_ratio",
            "max_radius_ratio",
            "avg_radius_ratio",
            "min_shape_quality",
            "max_shape_quality",
            "avg_shape_quality",
            "min_edge_length",
            "max_edge_length",
            "avg_edge_length",
            "#F",
            "#V",
        ]


metrics_names = get_metric_names()


def law_of_cosines(a, b, c):
    x = (b**2 + c**2 - a**2) / (2 * b * c)
    x = np.clip(x, -1.0, 1.0)  # numerical stability
    return np.arccos(x) * (180.0 / np.pi)


def compute_metrics_detailed(V, F):
    if F.shape[0] == 0:
        return []

    all_angles = []
    radius_ratios = []
    shape_qualities = []
    for face in F:
        v0 = V[face[0]]
        v1 = V[face[1]]
        v2 = V[face[2]]
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v0 - v2)
        c = np.linalg.norm(v0 - v1)

        # handle degenerate triangles
        if a == 0 or b == 0 or c == 0:
            # print("Degenerate triangle found:", face, "v0, v1, v2 =", v0, v1, v2)
            continue

        angles = [
            law_of_cosines(a, b, c),
            law_of_cosines(b, a, c),
            law_of_cosines(c, a, b),
        ]
        all_angles.extend(angles)

        s = (a + b + c) / 2.0
        area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), a_min=0, a_max=None))

        if area == 0:
            radius_ratios.append(0.0)
            continue

        inradius = area / s
        circumradius = (a * b * c) / (4.0 * area)
        radius_ratio = 2.0 * inradius / circumradius
        radius_ratios.append(radius_ratio)
        shape_quality = (4.0 * np.sqrt(3) * area) / (a**2 + b**2 + c**2)
        shape_qualities.append(shape_quality)

    all_angles = np.array(all_angles)
    all_angles.sort()
    radius_ratios = np.array(radius_ratios)
    radius_ratios.sort()
    shape_qualities = np.array(shape_qualities)
    shape_qualities.sort()

    # get bounding box
    bbox_min = np.min(V, axis=0)
    bbox_max = np.max(V, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = np.linalg.norm(bbox_size)
    # get edges
    edges = igl.edges(F)
    edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    edge_lengths.sort()
    edge_lengths /= bbox_diag  # normalize by bounding box diagonal

    # write all metrics in dictionary
    metrics = {
        "angle": all_angles.tolist(),
        "radius_ratio": radius_ratios.tolist(),
        "shape_quality": shape_qualities.tolist(),
        "edge_length": edge_lengths.tolist(),
        "#F": F.shape[0],
        "#V": V.shape[0],
    }

    return metrics


def compute_metrics_compact(V, F):
    if F.shape[0] == 0:
        return []

    all_angles = []
    radius_ratios = []
    shape_qualities = []
    for face in F:
        v0 = V[face[0]]
        v1 = V[face[1]]
        v2 = V[face[2]]
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v0 - v2)
        c = np.linalg.norm(v0 - v1)

        # handle degenerate triangles
        if a == 0 or b == 0 or c == 0:
            # print("Degenerate triangle found:", face, "v0, v1, v2 =", v0, v1, v2)
            continue

        angles = [
            law_of_cosines(a, b, c),
            law_of_cosines(b, a, c),
            law_of_cosines(c, a, b),
        ]
        all_angles.extend(angles)

        s = (a + b + c) / 2.0
        area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), a_min=0, a_max=None))

        if area == 0:
            radius_ratios.append(0.0)
            continue

        inradius = area / s
        circumradius = (a * b * c) / (4.0 * area)
        radius_ratio = 2.0 * inradius / circumradius
        radius_ratios.append(radius_ratio)
        shape_quality = (4.0 * np.sqrt(3) * area) / (a**2 + b**2 + c**2)
        shape_qualities.append(shape_quality)

    all_angles = np.array(all_angles)
    all_angles.sort()
    radius_ratios = np.array(radius_ratios)
    radius_ratios.sort()
    shape_qualities = np.array(shape_qualities)
    shape_qualities.sort()

    # get bounding box
    bbox_min = np.min(V, axis=0)
    bbox_max = np.max(V, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = np.linalg.norm(bbox_size)
    # get edges
    edges = igl.edges(F)
    edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    edge_lengths.sort()
    edge_lengths /= bbox_diag  # normalize by bounding box diagonal

    metrics = [
        all_angles[0],
        all_angles[-1],
        np.mean(all_angles),
        radius_ratios[0],
        radius_ratios[-1],
        np.mean(radius_ratios),
        shape_qualities[0],
        shape_qualities[-1],
        np.mean(shape_qualities),
        edge_lengths[0],
        edge_lengths[-1],
        np.mean(edge_lengths),
        F.shape[0],
        V.shape[0],
    ]

    return metrics


def load_mesh(mesh_path):
    mesh = meshio.read(mesh_path)

    if "triangle" not in mesh.cells_dict:
        V = np.zeros((0, 3))
        F = np.zeros((0, 3), dtype=int)
        return V, F

    V = mesh.points
    F = mesh.cells_dict["triangle"]
    return V, F


def mesh_metrics_compact(mesh_path):
    V, F = load_mesh(mesh_path)
    if F.shape[0] == 0:
        return []
    if F.shape[1] != 3:
        return []

    if use_pymeme:
        return pymeme.get_metrics(V, F)
    else:
        return compute_metrics_compact(V, F)


def mesh_metrics_detailed(mesh_path):
    V, F = load_mesh(mesh_path)
    return compute_metrics_detailed(V, F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mesh quality metrics for one or multiple triangular meshes stored as .vtu files."
    )
    # optional argument for path
    parser.add_argument(
        "-i",
        type=str,
        default=".",
        required=False,
        help="Path to the mesh file or folder with mesh files.",
    )
    parser.add_argument(
        "-o",
        type=str,
        default="mesh_quality_metrics.csv",
        required=False,
        help="Output CSV file to save the metrics.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".vtu",
        required=False,
        help="File extension of the mesh files (default: .vtu). Ignored if a single file is provided.",
    )
    parser.add_argument(
        "-j",
        type=int,
        default=1,
        required=False,
        help="Number of parallel jobs to use.",
    )
    args = parser.parse_args()

    path = args.i
    output_csv = args.o
    num_jobs = args.j
    extension = args.extension

    # record runtime
    start_time = time.time()

    if os.path.isfile(path):
        print(f"Evaluating mesh: {path}")

        metrics = mesh_metrics_compact(path)
        for name, value in zip(metrics_names, metrics):
            print(f"{name}: {value}")

        metrics = mesh_metrics_detailed(path)
        output_json = output_csv.replace(".csv", ".json")
        with open(output_json, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved results to {output_json}")
    else:
        # find all files recursively in path
        mesh_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(extension) and not f.endswith(f"_edges{extension}"):
                    mesh_files.append(os.path.join(root, f))

        print(f"Found {len(mesh_files)} mesh files.")

        non_empty_mesh_files = 0
        all_results = []

        if num_jobs > 1:
            print(f"Using {num_jobs} parallel jobs.")
            with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                future_to_mesh = {
                    executor.submit(mesh_metrics_compact, mesh_file): mesh_file
                    for mesh_file in mesh_files
                }
                for future in as_completed(future_to_mesh):
                    mesh_file = future_to_mesh[future]
                    try:
                        metrics = future.result()
                    except Exception as exc:  # keep failures visible
                        print(f"Failed on {mesh_file}: {exc}")
                        continue
                    if not metrics:
                        continue
                    non_empty_mesh_files += 1
                    all_results.append(metrics)
        else:
            print("Using single thread.")
            for mesh_file in mesh_files:
                try:
                    metrics = mesh_metrics_compact(mesh_file)
                except Exception as exc:
                    print(f"Failed on {mesh_file}: {exc}")
                    continue
                if not metrics:
                    continue
                non_empty_mesh_files += 1
                all_results.append(metrics)

        print(f"Evaluated {non_empty_mesh_files} non-empty mesh files.")
        df = pd.DataFrame(all_results, columns=metrics_names)
        # print first 5 rows of dataframe
        print(df.head())
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print("===== Finished =====")
