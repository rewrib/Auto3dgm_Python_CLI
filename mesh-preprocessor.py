import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

import pymeshlab as pml

# Initialize the logger
logger = logging.getLogger("mesh_preprocessing")
logger.setLevel(logging.INFO)  # or logging.DEBUG for more detailed logs

# Configure file handler to use the same log file as main.py
LOG_FILE = r"C:\zeug\BA2\master_pipeline.log"
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Console output handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Attach both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def setup_directories():
    """
    Parses command-line arguments, sets up directories for output,
    and retrieves the list of mesh files from the input directory.
    """
    parser = argparse.ArgumentParser(description="Mesh Preprocessing Script")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory with input mesh files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed meshes",
    )
    parser.add_argument(
        "--num_smooth", type=int, default=2, help="Number of smoothing iterations"
    )

    args = parser.parse_args()

    # Define input, output directories and smoothing iterations
    mesh_dir = args.input_dir
    output_dir = args.output_dir
    num_smooth = args.num_smooth

    # Define subdirectories for different mesh types
    notSimplyConnectedDir = os.path.join(output_dir, "NotSimplyConnected")
    discDir = os.path.join(output_dir, "DiscTopology")
    sphereDir = os.path.join(output_dir, "SphereTopology")
    badDir = os.path.join(output_dir, "BadMeshes")

    # Create directories if they don't exist
    for dir_path in [output_dir, notSimplyConnectedDir, discDir, sphereDir, badDir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of mesh files in the input directory
    supported_extensions = (".obj", ".stl", ".ply", ".fbx", ".dae", ".3ds")
    mesh_list = [
        os.path.join(root, f)
        for root, _, files in os.walk(mesh_dir)
        for f in files
        if f.lower().endswith(supported_extensions)
    ]

    logger.info(f"Found {len(mesh_list)} mesh files to process.")

    return (
        mesh_dir,
        output_dir,
        num_smooth,
        mesh_list,
        notSimplyConnectedDir,
        discDir,
        sphereDir,
        badDir,
    )


# Create directories if they do not exist
def touch(newDir):
    if not os.path.isdir(newDir):
        os.mkdir(newDir)


# Main processing function containing the original script
def main_preprocessing(
    meshDir,
    numSmooth,
    meshList,
    notSimplyConnectedDir,
    discDir,
    sphereDir,
    badDir,
):
    """
    Main function for mesh preprocessing. Uses pymeshlab for various mesh operations.
    """
    for i in range(len(meshList)):
        logger.info(f"Processing mesh: {meshList[i]}")
        ms = pml.MeshSet()
        try:
            ms.load_new_mesh(os.path.join(meshDir, meshList[i]))
        except Exception:
            logger.exception(f"Failed to load mesh {meshList[i]}")
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
                logger.exception(f"Failed to close holes in {meshList[i]}")
                continue

            out_dict = ms.get_topological_measures()
            cnt = cnt + 1
            if cnt == 30:
                logger.warning(
                    "Unable to clean without deleting some connected components, attempting..."
                )
                break
        if out_dict["connected_components_number"] > 1:
            logger.warning(f"{meshList[i]} has multiple connected components.")
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

        for j in range(numSmooth):
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
        if out_dict["is_mesh_two_manifold"] is False:
            logger.warning(
                f"{meshList[i]} is not two-manifold and could not be fully repaired after 10 attempts."
            )
            ms.save_current_mesh(badDir + os.path.basename(meshList[i]))
            continue

        for prefix in [badDir, notSimplyConnectedDir, discDir, sphereDir]:
            if os.path.isfile(prefix + meshList[i]):
                os.remove(prefix + meshList[i])

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
                ms.save_current_mesh(badDir + os.path.basename(meshList[i]))
                logger.info(meshList[i] + ":ConnectedComponentIssue", flush=True)
            elif out_dict["genus"] > 0:
                ms.save_current_mesh(
                    notSimplyConnectedDir + os.path.basename(meshList[i])
                )
                logger.info(meshList[i] + ":NotSimplyConnected", flush=True)
            elif out_dict["boundary_edges"] > 0:
                ms.save_current_mesh(discDir + os.path.basename(meshList[i]))
                logger.info(meshList[i] + ":Disc", flush=True)
            else:
                ms.save_current_mesh(sphereDir + os.path.basename(meshList[i]))
                logger.info(meshList[i] + ":Sphere", flush=True)
        except Exception as e:
            ms.save_current_mesh(badDir + os.path.basename(meshList[i]))
            logger.exception(
                f"Error processing {os.path.basename(meshList[i])}:BadMesh {e}"
            )


def main():
    (
        mesh_dir,
        output_dir,
        num_smooth,
        mesh_list,
        notSimplyConnectedDir,
        discDir,
        sphereDir,
        badDir,
    ) = setup_directories()

    # Run main preprocessing function
    main_preprocessing(
        mesh_dir,
        num_smooth,
        mesh_list,
        notSimplyConnectedDir,
        discDir,
        sphereDir,
        badDir,
    )


if __name__ == "__main__":
    main()
