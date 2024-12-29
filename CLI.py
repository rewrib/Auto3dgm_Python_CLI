#!/usr/bin/env python3

"""
CLI script that uses the 'downloads' table directly:
1) Reuses the same SQLite database that has 'downloads' (from Script 2).
2) Collects mesh file paths + taxonomy from 'downloads'.
3) Performs alignment using the Auto3dgm logic (based on Script 1).
4) Computes pairwise Procrustes distances for each pair of meshes.
5) Stores the results in 'auto3dgm_relationships'.
6) Computes cross-species median distance and stores in 'auto3dgm_species_pair_medians'.
7) Prints a summary to the console.

Usage:
  python auto3dgm_cli.py --db /path/to/database.db [--mesh-dir /path/to/meshes]
"""

import argparse
import os
import sqlite3
import statistics
import sys
import time
from itertools import combinations_with_replacement

import numpy as np
import numpy.matlib
import trimesh

# Auto3dgm
import auto3dgm_nazar
from auto3dgm_nazar.mesh.meshfactory import MeshFactory

###############################################################################
# CONFIGURATION / ARGUMENT PARSING
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI script to compare 3D meshes (talus bones) using Auto3dgm + Procrustes distance."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite database (same used in Script 2).",
    )
    parser.add_argument(
        "--mesh-dir",
        required=False,
        default="",
        help=(
            "Optional directory path for raw mesh files. "
            "If omitted, we assume 'downloads.file_name' is already a valid path."
        ),
    )
    parser.add_argument(
        "--num-subsample-low",
        type=int,
        default=100,
        help="Number of subsampled points for low-resolution alignment.",
    )
    parser.add_argument(
        "--num-subsample-high",
        type=int,
        default=200,
        help="Number of subsampled points for high-resolution alignment.",
    )
    parser.add_argument(
        "--reflection",
        action="store_true",
        help="Enable reflection during alignment (mirror=True).",
    )
    return parser.parse_args()


###############################################################################
# DATABASE SETUP
###############################################################################


def setup_database(db_file: str):
    """
    Create or update the SQLite database for storing Procrustes results.
    We assume the 'downloads' table already exists.
    We'll create 'auto3dgm_relationships' and 'auto3dgm_species_pair_medians'.
    """
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Store pairwise Procrustes distances in a new table.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auto3dgm_relationships (
            id1 INTEGER,  -- references downloads.id
            id2 INTEGER,  -- references downloads.id
            procrustes_distance REAL,
            UNIQUE(id1, id2)
        )
        """
    )

    # Cross-species median distances (like in script 2, but for 3D meshes).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auto3dgm_species_pair_medians (
            taxonomy1 TEXT,
            taxonomy2 TEXT,
            median_distance REAL,
            UNIQUE(taxonomy1, taxonomy2)
        )
        """
    )

    conn.commit()
    return conn, cur


###############################################################################
# HELPER FUNCTIONS (Adapted from Script 1)
###############################################################################


def Centralize(mesh, scale=None):
    """
    Centers a mesh at the origin and optionally normalizes its surface area to 1.
    Same logic as 'Centralize' in Script 1.
    """
    center = np.mean(mesh.vertices, axis=0).reshape(1, 3)
    foo = np.matlib.repmat(center, len(mesh.vertices), 1)
    mesh.vertices -= foo
    if scale is not None:
        mesh.vertices = mesh.vertices * np.sqrt(1.0 / mesh.area)
    return mesh, center


def procrustes_distance(vertsA, vertsB):
    """
    Computes the Procrustes distance between two sets of points that already have
    established correspondence and (optionally) scaling/alignment done.
    We'll just use the Frobenius norm of the difference.
    """
    diff = vertsA - vertsB
    return np.linalg.norm(diff, "fro")


###############################################################################
# MAIN LOGIC: LOADING FROM "downloads", ALIGNMENT, AND STORING
###############################################################################


def collect_meshes_from_downloads(cur, mesh_dir: str):
    """
    Fetch entries from 'downloads' where file_name is a known 3D format,
    build a data structure: { download_id : (full_mesh_path, taxonomy) }.
    If mesh_dir is provided, prepend it to file_name if file_name is not absolute.
    """
    allowed_exts = {".ply", ".obj", ".off", ".stl"}
    # Grab id, file_name, taxonomy from downloads
    cur.execute("SELECT id, file_name, taxonomy FROM downloads")
    rows = cur.fetchall()

    result = {}
    for dl_id, fname, tax in rows:
        if not fname:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in allowed_exts:
            continue

        # Construct possible full path
        if mesh_dir and not os.path.isabs(fname):
            full_path = os.path.join(mesh_dir, fname)
        else:
            full_path = fname

        # Store
        result[dl_id] = (full_path, tax if tax else "")
    return result


def align_meshes_and_get_highres_points(
    mesh_map,
    num_subsample_low,
    num_subsample_high,
    reflection=False,
):
    """
    - mesh_map: { download_id : (full_path, taxonomy) }
    - Load all meshes into memory, convert them to auto3dgm.
    - Perform alignment (low-res -> high-res).
    - Return final_data { id : { "original_vertices", "original_faces", "aligned_highres_points" } }
    """

    if not mesh_map:
        print("No usable meshes from 'downloads'.")
        return {}

    # 1) Load into trimesh and then into auto3dgm
    #    We'll store an auto3dgm mesh with name "dl_{id}" to keep track
    dataset_list = []
    tri_map = {}  # store raw trimesh objects if we need them later
    for dl_id, (path_str, _) in mesh_map.items():
        if not os.path.exists(path_str):
            print(f"Warning: file not found => {path_str}")
            continue
        tri = trimesh.load(path_str)
        tri_map[dl_id] = tri
        auto_m = MeshFactory.mesh_from_data(
            vertices=tri.vertices,
            faces=tri.faces,
            name=f"dl_{dl_id}",
            center_scale=False,
        )
        dataset_list.append(auto_m)

    if not dataset_list:
        return {}

    dataset_obj = auto3dgm_nazar.dataset.dataset.Dataset(dataset_list)

    # 2) Subsample for low-res alignment
    print("Subsampling (low-res) ...")
    ss_low = auto3dgm_nazar.mesh.subsample.Subsample(
        pointNumber=(num_subsample_low,),
        meshes=dataset_obj,
        center_scale=False,
        seed=None,
    )
    low_res_out = ss_low.ret[num_subsample_low]["output"]
    low_res_meshes = []
    for meshname, meshdata in low_res_out.items():
        newMesh = MeshFactory.mesh_from_data(
            meshdata.koodinimi,
            center_scale=True,
            name=meshname,
        )
        low_res_meshes.append(newMesh)

    # 3) Subsample for high-res
    print("Subsampling (high-res) ...")
    ss_high = auto3dgm_nazar.mesh.subsample.Subsample(
        pointNumber=(num_subsample_high,),
        meshes=dataset_obj,
        center_scale=False,
        seed=None,
    )
    high_res_out = ss_high.ret[num_subsample_high]["output"]
    high_res_meshes = []
    for meshname, meshdata in high_res_out.items():
        newMesh = MeshFactory.mesh_from_data(
            meshdata.koodinimi,
            center_scale=True,
            name=meshname,
        )
        high_res_meshes.append(newMesh)

    # 4) Align low-res
    print("Aligning low-res ...")
    corr_low = auto3dgm_nazar.analysis.correspondence.Correspondence(
        meshes=low_res_meshes,
        mirror=reflection,
    )
    ga = corr_low.globalized_alignment

    # 5) Align high-res using the low-res alignment
    print("Aligning high-res ...")
    corr_high = auto3dgm_nazar.analysis.correspondence.Correspondence(
        meshes=high_res_meshes,
        mirror=reflection,
        initial_alignment=ga,
    )

    # 6) Extract final aligned points
    final_data = {}
    # The order of final aligned meshes in corr_high should match
    # the order of high_res_meshes. We'll map "dl_{id}" -> index.
    name_to_idx = {m.name: i for i, m in enumerate(high_res_meshes)}

    for dl_id, tri_obj in tri_map.items():
        mesh_name = f"dl_{dl_id}"
        if mesh_name not in name_to_idx:
            continue
        idx = name_to_idx[mesh_name]
        aligned_verts = high_res_meshes[idx].vertices  # post-center_scale
        final_data[dl_id] = {
            "original_vertices": tri_obj.vertices,
            "original_faces": tri_obj.faces,
            "aligned_highres_points": aligned_verts,
        }

    return final_data


def store_similarity_scores(conn, cur, final_data):
    """
    For each pair (id1, id2), compute Procrustes distance from 'aligned_highres_points'
    and store in 'auto3dgm_relationships'.
    """
    all_ids = sorted(final_data.keys())

    for i, id1 in enumerate(all_ids):
        for id2 in all_ids[i + 1 :]:
            pts1 = final_data[id1]["aligned_highres_points"]
            pts2 = final_data[id2]["aligned_highres_points"]
            dist = procrustes_distance(pts1, pts2)
            cur.execute(
                """
                INSERT OR IGNORE INTO auto3dgm_relationships (id1, id2, procrustes_distance)
                VALUES (?, ?, ?)
                """,
                (id1, id2, dist),
            )

    conn.commit()


def display_relationships(cur):
    """
    Print pairwise distances from auto3dgm_relationships.
    We'll also join to 'downloads' so we can see file names or taxonomies if desired.
    """
    cur.execute(
        """
        SELECT d1.file_name, d2.file_name, r.procrustes_distance
        FROM auto3dgm_relationships r
        JOIN downloads d1 ON r.id1 = d1.id
        JOIN downloads d2 ON r.id2 = d2.id
        ORDER BY r.procrustes_distance ASC
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("No pairwise distances found in 'auto3dgm_relationships'.")
        return

    print("\nPairwise Procrustes Distances (lowest = most similar):")
    for f1, f2, dist in rows:
        print(f" - {os.path.basename(f1)} vs. {os.path.basename(f2)}: {dist:.4f}")


def compute_species_pair_medians(conn, cur):
    """
    For each distinct pair of species from 'downloads',
    compute the median Procrustes distance of all their pairs in auto3dgm_relationships.
    Then store in 'auto3dgm_species_pair_medians'.
    """
    # Collect distinct non-empty species from downloads
    cur.execute(
        """
        SELECT DISTINCT taxonomy
        FROM downloads
        WHERE taxonomy IS NOT NULL AND taxonomy != ''
        """
    )
    species_list = [row[0] for row in cur.fetchall()]

    species_pairs = list(combinations_with_replacement(sorted(species_list), 2))

    for specA, specB in species_pairs:
        # We'll gather all pairs (id1, id2) where one is specA, the other is specB.
        # Then gather their distances from auto3dgm_relationships.
        cur.execute(
            """
            SELECT r.procrustes_distance
            FROM auto3dgm_relationships r
            JOIN downloads d1 ON r.id1 = d1.id
            JOIN downloads d2 ON r.id2 = d2.id
            WHERE d1.taxonomy = ? AND d2.taxonomy = ?
            UNION
            SELECT r.procrustes_distance
            FROM auto3dgm_relationships r
            JOIN downloads d1 ON r.id1 = d1.id
            JOIN downloads d2 ON r.id2 = d2.id
            WHERE d1.taxonomy = ? AND d2.taxonomy = ?
            """,
            (specA, specB, specB, specA),
        )
        distances = [row[0] for row in cur.fetchall()]
        if not distances:
            continue
        median_val = statistics.median(distances)
        cur.execute(
            """
            INSERT INTO auto3dgm_species_pair_medians (taxonomy1, taxonomy2, median_distance)
            VALUES (?, ?, ?)
            ON CONFLICT(taxonomy1, taxonomy2)
            DO UPDATE SET median_distance = excluded.median_distance
            """,
            (specA, specB, median_val),
        )

    conn.commit()


def display_species_pair_medians(cur):
    """
    Show cross-species median distances from auto3dgm_species_pair_medians.
    """
    cur.execute(
        """
        SELECT taxonomy1, taxonomy2, median_distance
        FROM auto3dgm_species_pair_medians
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("\nNo cross-species median distances found.")
        return

    print("\nCross-Species Median Procrustes Distances:")
    for t1, t2, dist in rows:
        print(f" - {t1} vs. {t2}: {dist:.4f}")


###############################################################################
# MAIN
###############################################################################


def main():
    args = parse_args()
    conn, cur = setup_database(args.db)

    # 1) Collect mesh info from 'downloads'
    mesh_map = collect_meshes_from_downloads(cur, args.mesh_dir)
    if not mesh_map:
        print("No 3D mesh files found in 'downloads'. Exiting.")
        sys.exit(0)

    # 2) Subsample & align
    start_align_time = time.time()
    final_data = align_meshes_and_get_highres_points(
        mesh_map=mesh_map,
        num_subsample_low=args.num_subsample_low,
        num_subsample_high=args.num_subsample_high,
        reflection=args.reflection,
    )
    print(f"Alignment completed in {time.time() - start_align_time:.2f} seconds.")

    if not final_data:
        print("No aligned data found. Exiting.")
        sys.exit(0)

    # 3) Calculate Procrustes distances & store
    store_similarity_scores(conn, cur, final_data)

    # 4) Display pairwise results
    display_relationships(cur)

    # 5) Compute cross-species median distances
    compute_species_pair_medians(conn, cur)

    # 6) Display cross-species medians
    display_species_pair_medians(cur)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
