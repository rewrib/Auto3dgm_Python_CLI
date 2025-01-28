import logging
import os
import time

import pymeshlab as pml

MESH_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2"
OUTPUT_DIR = r"D:\Uni\BA\output\Morphosource\Meshes4_cleaned_logging"


NOT_SIMPLY_CONNECTED_DIR = os.path.join(OUTPUT_DIR, "NotSimplyConnected")
DISC_DIR = os.path.join(OUTPUT_DIR, "DiscTopology")
SPHERE_DIR = os.path.join(OUTPUT_DIR, "SphereTopology")
# path to bad meshes
BAD_DIR = os.path.join(OUTPUT_DIR, "BadMeshes")
# number of smoothing iterations
NUM_SMOOTH = 2
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "processing.log")


def touch(newDir):
    if not os.path.isdir(newDir):
        os.mkdir(newDir)


meshList = os.listdir(MESH_DIR)
touch(OUTPUT_DIR)
touch(NOT_SIMPLY_CONNECTED_DIR)
touch(DISC_DIR)
touch(SPHERE_DIR)
touch(BAD_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

for mesh in meshList:
    start_time = time.time()
    logger.info(mesh)
    try:
        ms = pml.MeshSet()
        try:
            ms.load_new_mesh(os.path.join(MESH_DIR, mesh))
        except Exception as e:
            logger.error(f"{mesh}: LoadError ({e})")
            continue
        ms.set_current_mesh(0)
        # ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(20))
        logger.info("Checking topology...")
        out_dict = ms.get_topological_measures()

        cnt = 0
        while not out_dict["is_mesh_two_manifold"]:
            logger.info(
                f"Attempting to repair non-manifold issues (iteration {cnt + 1})..."
            )
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()
            try:
                ms.meshing_close_holes(
                    maxholesize=50, newfaceselected=True, selfintersection=True
                )
            except Exception as e:
                logger.warning(f"{mesh}: Hole closing error ({e}), continuing.")
                continue

            out_dict = ms.get_topological_measures()
            cnt = cnt + 1
            if cnt == 30:
                logger.warning(
                    f"Stopping repairs after {cnt} iterations for mesh {mesh}."
                )
                break
        if out_dict["connected_components_number"] > 1:
            logger.info(
                f"Round 1: Mesh has multiple connected components: {out_dict['connected_components_number']}"
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
            ms.set_current_mesh(bestInd)
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp

        logger.info("Subdividing mesh to meet face count threshold...")
        cnt = 20
        while ms.current_mesh().face_number() < 10000:
            ms.meshing_surface_subdivision_loop(
                loopweight=1, iterations=1, threshold=pml.Percentage(0)
            )
            cnt = cnt - 1
            if cnt == 0:
                break
        logger.info("Applying decimation and smoothing...")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000, autoclean=True)

        for j in range(NUM_SMOOTH):
            ms.apply_coord_hc_laplacian_smoothing()

        logger.info("Performing final topology fixes...")
        cnt = 0
        out_dict = ms.get_topological_measures()
        if out_dict["connected_components_number"] > 1:
            logger.info(
                f"Round 2: Mesh has multiple connected components: {out_dict['connected_components_number']}"
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
            ms.set_current_mesh(bestInd)
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        out_dict = ms.get_topological_measures()
        while not out_dict["is_mesh_two_manifold"]:
            logger.info(f"Fixing non-manifold issues (iteration {cnt + 1})...")
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()
            if out_dict["connected_components_number"] > 1:
                logger.info(
                    f"Round 3: Mesh has multiple connected components: {out_dict['connected_components_number']}"
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
                ms.set_current_mesh(bestInd)
            msTemp = pml.MeshSet()
            msTemp.add_mesh(ms.current_mesh())
            ms = msTemp
            out_dict = ms.get_topological_measures()

            cnt = cnt + 1
            if cnt == 10:
                logger.warning(
                    f"{mesh}: Stopping manifold fixes after {cnt} iterations."
                )
                out_dict
                break

        """ for prefix in [BAD_DIR,NOT_SIMPLY_CONNECTED_DIR,DISC_DIR,SPHERE_DIR]:
            if os.path.isfile(prefix+meshList[mesh]):
                os.remove(prefix+meshList[mesh]) """

        ms.meshing_close_holes(
            maxholesize=30, newfaceselected=True, selfintersection=True
        )
        ms.meshing_surface_subdivision_loop(loopweight=1, iterations=2, selected=True)
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.Percentage(20)
        )
        ms.meshing_close_holes(
            maxholesize=30, newfaceselected=True, selfintersection=True
        )
        out_dict = ms.get_topological_measures()
        if out_dict["connected_components_number"] > 1:
            logger.info(
                f"Round 4: Mesh has multiple connected components: {out_dict['connected_components_number']}"
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
            ms.set_current_mesh(bestInd)
        try:
            logger.info("Classifying and saving mesh...")
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
        except Exception as e:
            ms.save_current_mesh(os.path.join(BAD_DIR, mesh))
            logger.error(f"{mesh}:BadMesh \n {e}")
    except Exception as e:
        logger.error(f"{mesh}:Error: \n {e}")
        continue

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Finished processing mesh: {mesh} in {elapsed:.2f} seconds.\n")
