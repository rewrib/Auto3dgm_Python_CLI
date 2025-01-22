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
MESH_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2"
OUTPUT_DIR = r"D:\Uni\BA\output\Morphosource\Meshes3_cleaned"


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
    print(meshList[mesh], flush=True)
    ms = pml.MeshSet()
    try:
        ms.load_new_mesh(MESH_DIR + meshList[mesh])
    except:
        continue
    ms.set_current_mesh(0)
    # ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(20))
    out_dict = ms.get_topological_measures()

    cnt = 0
    while not out_dict["is_mesh_two_manifold"]:
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        try:
            ms.meshing_close_holes(
                maxholesize=50, newfaceselected=True, selfintersection=True
            )
        except:
            continue

        out_dict = ms.get_topological_measures()
        cnt = cnt + 1
        if cnt == 30:
            print(
                "Unable to clean without deleting some connected components, attempting..."
            )
            break
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

    cnt = 20
    while ms.current_mesh().face_number() < 10000:
        ms.meshing_surface_subdivision_loop(
            loopweight=1, iterations=1, threshold=pml.Percentage(0)
        )
        cnt = cnt - 1
        if cnt == 0:
            break
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000, autoclean=True)

    for j in range(NUM_SMOOTH):
        ms.apply_coord_hc_laplacian_smoothing()

    cnt = 0
    out_dict = ms.get_topological_measures()
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
    while not out_dict["is_mesh_two_manifold"]:
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
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
    for prefix in [BAD_DIR, NOT_SIMPLY_CONNECTED_DIR, DISC_DIR, SPHERE_DIR]:
        if os.path.isfile(prefix + meshList[mesh]):
            os.remove(prefix + meshList[mesh])

    ms.meshing_close_holes(maxholesize=30, newfaceselected=True, selfintersection=True)
    ms.meshing_surface_subdivision_loop(loopweight=1, iterations=2, selected=True)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pml.Percentage(20)
    )
    ms.meshing_close_holes(maxholesize=30, newfaceselected=True, selfintersection=True)
    out_dict = ms.get_topological_measures()
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
    try:
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        ms.meshing_re_orient_faces_coherentely()
        out_dict = ms.get_topological_measures()

        if out_dict["connected_components_number"] > 1:
            ms.save_current_mesh(BAD_DIR + meshList[mesh])
            print(meshList[mesh] + ":ConnectedComponentIssue", flush=True)
        elif out_dict["genus"] > 0:
            ms.save_current_mesh(NOT_SIMPLY_CONNECTED_DIR + meshList[mesh])
            print(meshList[mesh] + ":NotSimplyConnected", flush=True)
        elif out_dict["boundary_edges"] > 0:
            ms.save_current_mesh(DISC_DIR + meshList[mesh])
            print(meshList[mesh] + ":Disc", flush=True)
        else:
            ms.save_current_mesh(SPHERE_DIR + meshList[mesh])
            print(meshList[mesh] + ":Sphere", flush=True)
    except:
        ms.save_current_mesh(BAD_DIR + meshList[mesh])
        print(meshList[mesh] + ":BadMesh", flush=True)


if __name__ == "__main__":
    logger.info("Script started.")
    meshList = os.listdir(MESH_DIR)
    touch(OUTPUT_DIR)
    touch(NOT_SIMPLY_CONNECTED_DIR)
    touch(DISC_DIR)
    touch(SPHERE_DIR)
    touch(BAD_DIR)

    for mesh in meshList:
        process_mesh(
            MESH_DIR,
            NOT_SIMPLY_CONNECTED_DIR,
            DISC_DIR,
            SPHERE_DIR,
            BAD_DIR,
            NUM_SMOOTH,
            meshList,
            mesh,
        )
