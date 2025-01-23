import logging
import os

import pymeshlab as pml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# INPUT
# path to meshes
# PLEASE ADD SLASHES AS APPROPRIATE FOR OS
#MESH_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2"
#OUTPUT_DIR = r"D:\Uni\BA\output\Morphosource\Meshes3_cleaned"

MESH_DIR = r"/home/batest/Projects/BA/output/Morphosource/Meshes2/"
OUTPUT_DIR = r"/home/batest/Projects/BA/output/Morphosource/Meshes6_cleaned/"


NOT_SIMPLY_CONNECTED_DIR = os.path.join(OUTPUT_DIR, "NotSimplyConnected")
DISC_DIR = os.path.join(OUTPUT_DIR, "DiscTopology")
SPHERE_DIR = os.path.join(OUTPUT_DIR, "SphereTopology")
# path to bad meshes
BAD_DIR = os.path.join(OUTPUT_DIR, "BadMeshes")
# number of smoothing iterations
NUM_SMOOTH = 2


def touch(newDir):
    if not os.path.isdir(newDir):
        logger.info(f"Creating directory: {newDir}")
        os.mkdir(newDir)


def process_mesh(mesh):
    logger.info(f"Processing mesh: {mesh}")
    ms = pml.MeshSet()
    try:
        ms.load_new_mesh(MESH_DIR + mesh)
    except:
        # TODO: check logic
        return
    ms.set_current_mesh(0)
    # ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(20))
    out_dict = ms.get_topological_measures()

    out_dict = try_manifold_repairs(ms, out_dict)
    keep_largest_component(ms, out_dict)
    msTemp = pml.MeshSet()
    msTemp.add_mesh(ms.current_mesh())
    ms = msTemp

    subdivide_if_needed(ms)
    logger.info(f"Smoothing the mesh {NUM_SMOOTH} time(s).")
    for j in range(NUM_SMOOTH):
        ms.apply_coord_hc_laplacian_smoothing()

    out_dict = ms.get_topological_measures()
    keep_largest_component(ms, out_dict)
    msTemp = pml.MeshSet()
    msTemp.add_mesh(ms.current_mesh())
    ms = msTemp
    out_dict = ms.get_topological_measures()
    ms = more_manifold_repairs(ms, out_dict)
    for prefix in [BAD_DIR, NOT_SIMPLY_CONNECTED_DIR, DISC_DIR, SPHERE_DIR]:
        if os.path.isfile(prefix + mesh):
            os.remove(prefix + mesh)

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
    out_dict = ms.get_topological_measures()
    keep_largest_component(ms, out_dict)
    try:
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        ms.meshing_re_orient_faces_coherentely()
        out_dict = ms.get_topological_measures()

        if out_dict["connected_components_number"] > 1:
            ms.save_current_mesh(os.path.join(BAD_DIR, mesh))
            logger.info(mesh + ":ConnectedComponentIssue")
        elif out_dict["genus"] > 0:
            ms.save_current_mesh(os.path.join(NOT_SIMPLY_CONNECTED_DIR, mesh))
            logger.info(mesh + ":NotSimplyConnected")
        elif out_dict["boundary_edges"] > 0:
            ms.save_current_mesh(os.path.join(DISC_DIR, mesh))
            logger.info(mesh + ":Disc")
        else:
            ms.save_current_mesh(os.path.join(SPHERE_DIR, mesh))
            logger.info(mesh + ":Sphere")
    except Exception as exc:
        logger.error(f"Exception during final classification: {exc}")
        logger.info(f"{mesh}: BadMesh -> Saving to {BAD_DIR}.")
        ms.save_current_mesh(os.path.join(BAD_DIR, mesh))


def more_manifold_repairs(ms, out_dict):
    cnt = 0
    while not out_dict["is_mesh_two_manifold"]:
        repair_mesh(ms)
        if out_dict["connected_components_number"] > 1:
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
            ms.set_current_mesh(bestInd)
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        out_dict = ms.get_topological_measures()

        cnt = cnt + 1
        if cnt == 10:
            out_dict
            break
    return ms

def subdivide_if_needed(ms):
    cnt = 20
    while ms.current_mesh().face_number() < 10000:
        ms.meshing_surface_subdivision_loop(
            loopweight=1, iterations=1, threshold=pml.Percentage(0)
        )
        cnt = cnt - 1
        if cnt == 0:
            break
    logger.info("Decimating to target face count = 10000.")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000, autoclean=True)

def keep_largest_component(ms, out_dict):
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

def try_manifold_repairs(ms, out_dict):
    cnt = 0
    while not out_dict["is_mesh_two_manifold"]:
        logger.info("Mesh is not two-manifold. Attempting repairs.")
        repair_mesh(ms)
        try:
            ms.meshing_close_holes(
                maxholesize=50, newfaceselected=True, selfintersection=True
            )
        except Exception as ex:
            logger.warning(f"Exception while closing holes: {ex}")
            continue

        out_dict = ms.get_topological_measures()
        cnt = cnt + 1
        if cnt == 30:
            logger.warning(
                "Unable to clean without deleting some connected components, attempting..."
            )
            break
    return out_dict

def repair_mesh(ms):
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()


if __name__ == "__main__":
    logger.info("Script started.")
    meshList = os.listdir(MESH_DIR)
    touch(OUTPUT_DIR)
    touch(NOT_SIMPLY_CONNECTED_DIR)
    touch(DISC_DIR)
    touch(SPHERE_DIR)
    touch(BAD_DIR)

    for mesh in meshList:
        process_mesh(mesh)
