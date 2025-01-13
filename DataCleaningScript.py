import logging
import os

import pymeshlab as pml

# --- Configure Logging ---
# You can customize the format to your preference. Below includes:
# - Timestamp
# - Log level
# - Message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths to meshes
MESH_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2"
OUTPUT_DIR = r"D:\Uni\BA\output\Morphosource\Meshes3_cleaned"

NOT_SIMPLY_CONNECTED_DIR = os.path.join(OUTPUT_DIR, "NotSimplyConnected")
DISC_DIR = os.path.join(OUTPUT_DIR, "DiscTopology")
SPHERE_DIR = os.path.join(OUTPUT_DIR, "SphereTopology")
BAD_DIR = os.path.join(OUTPUT_DIR, "BadMeshes")

# Number of smoothing iterations
NUM_SMOOTH = 2


def touch(newDir):
    """Create directory if it doesn't already exist."""
    if not os.path.isdir(newDir):
        logger.info(f"Creating directory: {newDir}")
        os.mkdir(newDir)


def repair_mesh(ms):
    """Attempt to repair the mesh by removing various manifold issues."""
    logger.info("Repairing mesh (non-manifold edges, vertices, duplicates, etc.)")
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()


def try_manifold_repairs(ms, out_dict):
    """
    Repeatedly attempt manifold repairs and close holes
    until it becomes two-manifold or we reach a limit.
    """
    logger.info("Entering try_manifold_repairs() loop.")
    cnt = 0
    while not out_dict["is_mesh_two_manifold"]:
        logger.info("Mesh is not two-manifold. Attempting repairs.")
        repair_mesh(ms)
        try:
            logger.info("Closing holes with max holesize = 50.")
            ms.meshing_close_holes(
                maxholesize=50, newfaceselected=True, selfintersection=True
            )
        except Exception as exc:
            logger.warning(f"Exception while closing holes: {exc}")
            continue

        out_dict = ms.get_topological_measures()
        cnt += 1
        if cnt == 9999:
            logger.warning(
                "Unable to clean without deleting components. Breaking loop."
            )
            break
    logger.info("Exiting try_manifold_repairs() loop.")
    return out_dict


def keep_largest_component(ms, out_dict):
    """If there are multiple connected components, keep only the largest."""
    logger.info("Checking for multiple connected components...")
    if out_dict["connected_components_number"] > 1:
        logger.info(
            f"Found {out_dict['connected_components_number']} connected components. Splitting..."
        )
        ms.generate_splitting_by_connected_components()
        bestVol = 0
        bestInd = 0
        for j in range(out_dict["connected_components_number"]):
            k = j + 1
            curVol = (
                ms.mesh(k).bounding_box().dim_x()
                * ms.mesh(k).bounding_box().dim_y()
                * ms.mesh(k).bounding_box().dim_z()
            )
            if curVol > bestVol:
                bestInd = j + 1
                bestVol = curVol
        logger.info(f"Keeping largest component: #{bestInd}, volume={bestVol}")
        ms.set_current_mesh(bestInd)


def subdivide_if_needed(ms):
    """
    Subdivide until face_number exceeds a threshold or we have done it
    enough times. Then decimate to a target face count.
    """
    logger.info("Checking if subdivision is needed...")
    cnt = 20
    while ms.current_mesh().face_number() < 10000:
        logger.info(
            f"Current face count = {ms.current_mesh().face_number()}, subdividing..."
        )
        ms.meshing_surface_subdivision_loop(
            loopweight=1, iterations=1, threshold=pml.Percentage(0)
        )
        cnt -= 1
        if cnt == 0:
            logger.warning("Reached max subdivision attempts. Breaking out.")
            break

    logger.info("Decimating to target face count = 10000.")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000, autoclean=True)


def try_manifold_repairs2(ms, cnt):
    """
    Another attempt at manifold repairs using keep_largest_component
    after each repair iteration.
    """
    logger.info("Entering try_manifold_repairs2() loop.")
    out_dict = ms.get_topological_measures()
    while not out_dict["is_mesh_two_manifold"]:
        logger.info(
            "Mesh is not two-manifold. Attempting repairs with largest component check."
        )
        repair_mesh(ms)
        keep_largest_component(ms, out_dict)
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        out_dict = ms.get_topological_measures()

        cnt += 1
        if cnt == 9999:
            logger.warning(
                "Max attempts reached in try_manifold_repairs2(). Breaking loop."
            )
            break
    logger.info("Exiting try_manifold_repairs2() loop.")
    return ms


def final_topology_classification(
    notSimplyConnectedDir, discDir, sphereDir, badDir, mesh, ms
):
    """
    Final classification based on genus and boundary edges.
    """
    logger.info("Reorienting faces coherently.")
    ms.meshing_re_orient_faces_coherentely()
    out_dict = ms.get_topological_measures()

    logger.info("Final topology classification check.")
    if out_dict["connected_components_number"] > 1:
        logger.info(f"{mesh}: ConnectedComponentIssue -> Saving to {badDir}.")
        ms.save_current_mesh(os.path.join(badDir, mesh))
    elif out_dict["genus"] > 0:
        logger.info(f"{mesh}: NotSimplyConnected -> Saving to {notSimplyConnectedDir}.")
        ms.save_current_mesh(os.path.join(notSimplyConnectedDir, mesh))
    elif out_dict["boundary_edges"] > 0:
        logger.info(f"{mesh}: Disc -> Saving to {discDir}.")
        ms.save_current_mesh(os.path.join(discDir, mesh))
    else:
        logger.info(f"{mesh}: Sphere -> Saving to {sphereDir}.")
        ms.save_current_mesh(os.path.join(sphereDir, mesh))


def process_mesh(mesh):
    """
    Main routine to clean, subdivide, repair, and classify a single mesh.
    """
    logger.info(f"Processing mesh: {mesh}")
    ms = pml.MeshSet()

    meshPath = os.path.join(MESH_DIR, mesh)
    logger.info(f"Loading mesh from: {meshPath}")
    ms.load_new_mesh(meshPath)

    ms.set_current_mesh(0)

    # Initial topological measures
    out_dict = ms.get_topological_measures()

    # Try manifold repairs (first pass)
    out_dict = try_manifold_repairs(ms, out_dict)

    # Keep largest component if multiple
    keep_largest_component(ms, out_dict)
    msTemp = pml.MeshSet()
    msTemp.add_mesh(ms.current_mesh())
    ms = msTemp

    # Subdivide if needed
    subdivide_if_needed(ms)

    # Smooth the mesh a given number of times
    logger.info(f"Smoothing the mesh {NUM_SMOOTH} time(s).")
    for j in range(NUM_SMOOTH):
        ms.apply_coord_hc_laplacian_smoothing()

    # Second pass: repair + largest component
    out_dict = ms.get_topological_measures()
    keep_largest_component(ms, out_dict)
    msTemp = pml.MeshSet()
    msTemp.add_mesh(ms.current_mesh())
    ms = msTemp
    out_dict = try_manifold_repairs(ms, out_dict)

    # Additional cleaning steps
    logger.info("Closing holes with max size=30 on selected faces.")
    ms.meshing_close_holes(maxholesize=30, newfaceselected=True, selfintersection=True)

    logger.info("Surface subdivision on selected faces (2 iterations).")
    ms.meshing_surface_subdivision_loop(loopweight=1, iterations=2, selected=True)

    logger.info("Removing unreferenced vertices.")
    ms.meshing_remove_unreferenced_vertices()

    logger.info("Removing small connected components by diameter (20%).")
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pml.Percentage(20)
    )

    logger.info("Closing holes again (max size=30).")
    ms.meshing_close_holes(maxholesize=30, newfaceselected=True, selfintersection=True)

    # Keep largest after final cleaning
    out_dict = ms.get_topological_measures()
    keep_largest_component(ms, out_dict)

    # Final classification
    try:
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        final_topology_classification(
            NOT_SIMPLY_CONNECTED_DIR, DISC_DIR, SPHERE_DIR, BAD_DIR, mesh, ms
        )
    except Exception as exc:
        logger.error(f"Exception during final classification: {exc}")
        logger.info(f"{mesh}: BadMesh -> Saving to {BAD_DIR}.")
        ms.save_current_mesh(os.path.join(BAD_DIR, mesh))


if __name__ == "__main__":
    logger.info("Script started.")
    meshList = os.listdir(MESH_DIR)
    touch(OUTPUT_DIR)
    touch(NOT_SIMPLY_CONNECTED_DIR)
    touch(DISC_DIR)
    touch(SPHERE_DIR)
    touch(BAD_DIR)

    for mesh in meshList:
        try:
            process_mesh(mesh)
        except Exception as e:
            logger.error(f"Error loading/processing mesh {mesh}: {e}")
            continue
    logger.info("Script finished.")
