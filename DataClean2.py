import os

import pymeshlab as pml

###############################################################################
# Configuration
###############################################################################
MESH_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2"
OUTPUT_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2_cleaned"

NOT_SIMPLY_CONNECTED_DIR = os.path.join(OUTPUT_DIR, "NotSimplyConnected")
DISC_DIR = os.path.join(OUTPUT_DIR, "DiscTopology")
SPHERE_DIR = os.path.join(OUTPUT_DIR, "SphereTopology")
BAD_DIR = os.path.join(OUTPUT_DIR, "BadMeshes")

# Number of smoothing iterations
NUM_SMOOTH = 2

# Maximum repair attempts before giving up
MAX_MANIFOLD_REPAIR_ATTEMPTS = 10

# Face count threshold
TARGET_FACE_COUNT = 10000

# Max times we subdivide if the mesh is too small
MAX_SUBDIV_ATTEMPTS = 20

# Hole closure sizes
MAX_HOLE_SIZE = 30


###############################################################################
# Utilities
###############################################################################
def touch(new_dir):
    """Create a directory if it doesn't exist."""
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)


def keep_largest_component(ms):
    """
    If multiple connected components exist, keep only the largest by bounding-box volume.
    """
    out_dict = ms.get_topological_measures()
    if out_dict["connected_components_number"] > 1:
        ms.generate_splitting_by_connected_components()
        best_vol = 0
        best_ind = 1
        for idx in range(out_dict["connected_components_number"]):
            mesh_id = idx + 1
            m = ms.mesh(mesh_id)
            bb = m.bounding_box()
            cur_vol = bb.dim_x() * bb.dim_y() * bb.dim_z()
            if cur_vol > best_vol:
                best_vol = cur_vol
                best_ind = mesh_id
        ms.set_current_mesh(best_ind)

    # Clean up to have only one mesh in the MeshSet
    ms_temp = pml.MeshSet()
    ms_temp.add_mesh(ms.current_mesh())
    return ms_temp


def try_manifold_repairs(ms, max_attempts=MAX_MANIFOLD_REPAIR_ATTEMPTS):
    """
    Attempt to repair non-manifold edges/vertices, remove duplicates, etc.
    Stop early if the mesh becomes two-manifold.
    """
    out_dict = ms.get_topological_measures()
    attempt = 0
    while not out_dict["is_mesh_two_manifold"] and attempt < max_attempts:
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        try:
            ms.meshing_close_holes(
                maxholesize=50, newfaceselected=True, selfintersection=True
            )
        except Exception:
            pass

        # Keep only the largest component if there are multiple
        ms = keep_largest_component(ms)
        out_dict = ms.get_topological_measures()
        attempt += 1

    return ms


def subdivide_if_needed(
    ms, target_faces=TARGET_FACE_COUNT, max_subdiv=MAX_SUBDIV_ATTEMPTS
):
    """
    If the mesh has fewer than target_faces, subdivide it (Loop subdivision)
    up to max_subdiv times, then decimate back to target_faces.
    """
    face_count = ms.current_mesh().face_number()
    subdiv_attempts = 0
    while face_count < target_faces and subdiv_attempts < max_subdiv:
        ms.meshing_surface_subdivision_loop(
            loopweight=1, iterations=1, threshold=pml.Percentage(0)
        )
        face_count = ms.current_mesh().face_number()
        subdiv_attempts += 1

    # Now decimate to target_faces (if we have more than that)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces, autoclean=True
    )
    return ms


def final_topology_classification(ms, mesh_name):
    """
    Save the mesh into the appropriate directory based on topological measures.
    """
    # Remove any older copies in each classification directory
    for directory in [BAD_DIR, NOT_SIMPLY_CONNECTED_DIR, DISC_DIR, SPHERE_DIR]:
        old_path = os.path.join(directory, mesh_name)
        if os.path.isfile(old_path):
            os.remove(old_path)

    # Try re-orienting faces
    try:
        ms.meshing_re_orient_faces_coherentely()
        out_dict = ms.get_topological_measures()

        if out_dict["connected_components_number"] > 1:
            ms.save_current_mesh(os.path.join(BAD_DIR, mesh_name))
            print(f"{mesh_name}: ConnectedComponentIssue")
        elif out_dict["genus"] > 0:
            ms.save_current_mesh(os.path.join(NOT_SIMPLY_CONNECTED_DIR, mesh_name))
            print(f"{mesh_name}: NotSimplyConnected")
        elif out_dict["boundary_edges"] > 0:
            ms.save_current_mesh(os.path.join(DISC_DIR, mesh_name))
            print(f"{mesh_name}: Disc")
        else:
            ms.save_current_mesh(os.path.join(SPHERE_DIR, mesh_name))
            print(f"{mesh_name}: Sphere")

    except Exception:
        ms.save_current_mesh(os.path.join(BAD_DIR, mesh_name))
        print(f"{mesh_name}: BadMesh")


###############################################################################
# Main script
###############################################################################
if __name__ == "__main__":
    touch(OUTPUT_DIR)
    touch(NOT_SIMPLY_CONNECTED_DIR)
    touch(DISC_DIR)
    touch(SPHERE_DIR)
    touch(BAD_DIR)

    mesh_list = os.listdir(MESH_DIR)

    for mesh_name in mesh_list:
        mesh_path = os.path.join(MESH_DIR, mesh_name)
        if not os.path.isfile(mesh_path):
            continue

        try:
            print(mesh_name, flush=True)
            ms = pml.MeshSet()
            ms.load_new_mesh(mesh_path)

            # 1) Attempt manifold repairs
            ms = try_manifold_repairs(ms)
            out_dict = ms.get_topological_measures()

            # 2) If there's more than 1 connected component, keep only the largest
            ms = keep_largest_component(ms)
            out_dict = ms.get_topological_measures()

            # 3) Subdivide if needed, then decimate
            ms = subdivide_if_needed(ms)

            # 4) Apply smoothing
            for _ in range(NUM_SMOOTH):
                ms.apply_coord_hc_laplacian_smoothing()

            # 5) Repair manifold issues again if needed (less attempts)
            ms = try_manifold_repairs(ms, max_attempts=5)

            # 6) Close holes and remove small components
            ms.meshing_close_holes(
                maxholesize=MAX_HOLE_SIZE, newfaceselected=True, selfintersection=True
            )
            ms.meshing_surface_subdivision_loop(
                loopweight=1, iterations=2, selected=True
            )
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pml.Percentage(20)
            )
            ms.meshing_close_holes(
                maxholesize=MAX_HOLE_SIZE, newfaceselected=True, selfintersection=True
            )

            # 7) Keep only the largest component again
            ms = keep_largest_component(ms)

            # 8) Final classification and save
            final_topology_classification(ms, mesh_name)

        except Exception as e:
            print(f"Error loading/processing mesh {mesh_name}: {e}", flush=True)
            continue
