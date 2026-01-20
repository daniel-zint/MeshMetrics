import random
import sys
import igl

# from line_profiler import profile
import numpy as np
import meshio
import pandas as pd
import json
import argparse
import os
import time
import traceback
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

use_pymeme = True
import pymeme

# # try to import pymeme
# if use_pymeme:
#     try:
#         import pymeme

#         use_pymeme = True
#         print("Using pymeme for metric computation.")
#     except ImportError:
#         use_pymeme = False
#         print("ERROR: pymeme not found, using Python implementation.")
#         pass
# else:
#     print("Using Python implementation for metric computation.")

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
    pip install -v 'libigl==2.5.1'
    pip install pandas meshio

Installing pymeme:
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    copy app/pymeme... to the project folder

"""


def get_metric_names():
    names = pymeme.get_metric_names_per_tri()
    # names.append("mesh_file")
    return names


metrics_names = get_metric_names()


def load_mesh(mesh_path):
    mesh = meshio.read(mesh_path)

    if "triangle" not in mesh.cells_dict:
        V = np.zeros((0, 3))
        F = np.zeros((0, 3), dtype=int)
        return V, F

    V = mesh.points
    F = mesh.cells_dict["triangle"]
    return V, F


def mesh_metrics_per_tri(results_path, mesh_id):
    out_mesh_path = os.path.join(results_path, mesh_id, "remeshed.vtu")
    in_mesh_path = os.path.join(results_path, mesh_id, "mesh.obj")

    if not os.path.exists(out_mesh_path) or not os.path.exists(in_mesh_path):
        raise FileNotFoundError(
            f"Mesh files not found for mesh ID {mesh_id} in {results_path}."
        )

    # metrics for input mesh
    V_in, F_in = load_mesh(in_mesh_path)
    metrics_in = pymeme.get_metrics_per_tri(V_in, F_in)
    metrics_in = metrics_in.astype(np.float16)

    # metrics for remeshed mesh
    V_out, F_out = load_mesh(out_mesh_path)
    metrics_out = pymeme.get_metrics_per_tri(V_out, F_out)
    metrics_out = metrics_out.astype(np.float16)
    return metrics_in, metrics_out


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
        "-j",
        type=int,
        default=1,
        required=False,
        help="Number of parallel jobs to use.",
    )
    args = parser.parse_args()

    results_path = args.i
    num_jobs = args.j
    # record runtime
    start_time = time.time()

    # find all folders in path
    mesh_ids = [
        d
        for d in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, d))
    ]

    print(f"Found {len(mesh_ids)} folders in {results_path}.")

    n_samples = 10
    if len(mesh_ids) > n_samples:
        print(f"Evaluating runtimes on {n_samples} random meshes...")
        # pick random 10 meshes for testing (use pseudo-random for reproducibility)
        random.seed(42)
        mesh_ids = random.sample(mesh_ids, n_samples)
    else:
        print(f"Evaluating runtimes on all {len(mesh_ids)} meshes...")

    all_results_in = []
    all_results_out = []

    if len(mesh_ids) == 0:
        print("No meshes to process. Exiting.")
        exit(0)

    non_empty_mesh_files = 0

    output_pickle = "metrics_per_tri.pkl"

    if num_jobs > 1:
        print(f"Using {num_jobs} parallel jobs.")
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            future_to_mesh = {
                executor.submit(
                    mesh_metrics_per_tri,
                    results_path,
                    mesh_id,
                ): mesh_id
                for mesh_id in mesh_ids
            }
            for future in as_completed(future_to_mesh):
                mesh_id = future_to_mesh[future]
                try:
                    metrics_in, metrics_out = future.result()
                except Exception as exc:  # keep failures visible
                    print(f"Failed on {mesh_id}: {exc}")
                    continue
                non_empty_mesh_files += 1
                all_results_in.append(metrics_in)
                all_results_out.append(metrics_out)
                print(
                    f"{non_empty_mesh_files}/{len(mesh_ids)} Processed mesh: {mesh_id}"
                )
                sys.stdout.flush()
    else:
        print("Using single thread.")
        for mesh_id in mesh_ids:
            try:
                metrics_in, metrics_out = mesh_metrics_per_tri(results_path, mesh_id)
            except Exception as exc:
                print(f"Failed on {mesh_id}: {exc}")
                # traceback.print_exc()
                continue
            non_empty_mesh_files += 1
            all_results_in.append(metrics_in)
            all_results_out.append(metrics_out)
            print(f"{non_empty_mesh_files}/{len(mesh_ids)} Processed mesh: {mesh_id}")
            sys.stdout.flush()

    print(f"Evaluated {non_empty_mesh_files} non-empty mesh files.")

    # Convert lists to numpy arrays
    if all_results_in:
        all_results_in = np.vstack(all_results_in)
        all_results_out = np.vstack(all_results_out)

        df_in = pd.DataFrame(all_results_in, columns=metrics_names)
        print(df_in.head())  # print first 5 rows of dataframe

        df_out = pd.DataFrame(all_results_out, columns=metrics_names)
        print(df_out.head())  # print first 5 rows of dataframe

        # Store both arrays in a single pickle file
        with open(output_pickle, "wb") as f:
            pickle.dump(
                {
                    "names": metrics_names,
                    "in": all_results_in,
                    "out": all_results_out,
                },
                f,
            )

        print(f"Saved results to {output_pickle}")

    stop_time = time.time()
    print(f"Total runtime: {stop_time - start_time:.2f} seconds")
