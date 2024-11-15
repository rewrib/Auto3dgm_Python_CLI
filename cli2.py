import argparse
import csv
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import numpy as np
import numpy.matlib
import pandas as pd
import psutil
import scipy.spatial
import trimesh

import auto3dgm_nazar

# Initialize the logger
logger = logging.getLogger("auto3dgm_cli")
logger.setLevel(logging.DEBUG)  # or logging.DEBUG for more detailed logs

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


class Auto3DGM_CLI:
    def __init__(self, args):
        self.args = args
        self.alignData = None
        self.sampledMeshes = None
        self.originalMeshes = None

        # Read settings from JSON file if provided
        if args.settings_json:
            with open(args.settings_json, "r") as f:
                self.settings = json.load(f)
        else:
            self.settings = vars(args)

    def validate_settings(self):
        mesh_dir = self.settings["mesh_dir"]
        output_dir = self.settings["output_dir"]

        if not os.path.isdir(mesh_dir):
            logger.error(f"Mesh directory '{mesh_dir}' does not exist.")
            sys.exit(1)

        if not os.path.isdir(output_dir):
            logger.info(f"Output directory '{output_dir}' does not exist. Creating it.")
            os.makedirs(output_dir)

        # Add trailing slash if needed
        if not mesh_dir.endswith("/") and not mesh_dir.endswith("\\"):
            mesh_dir += "/"
        if not output_dir.endswith("/") and not output_dir.endswith("\\"):
            output_dir += "/"

        self.settings["mesh_dir"] = mesh_dir
        self.settings["output_dir"] = output_dir

    def convert(self):
        mesh_dir = self.settings["mesh_dir"]
        output_dir = os.path.join(self.settings["output_dir"], "originalMeshes/")
        self.touch(output_dir)

        logger.info("Converting mesh files to .ply format if necessary...")
        for file in os.listdir(mesh_dir):
            if file.endswith((".off", ".obj", ".ply")):
                filename = os.path.splitext(file)[0]
                mesh = trimesh.load(os.path.join(mesh_dir, file), process=False)
                mesh.export(os.path.join(output_dir, filename + ".ply"))

        self.settings["mesh_dir"] = output_dir

    def Centralize(self, mesh, scale=None):
        Center = np.mean(mesh.vertices, 0).reshape(1, 3)
        foo = np.matlib.repmat(Center, len(mesh.vertices), 1)
        mesh.vertices -= foo

        if scale is not None:
            mesh.vertices = mesh.vertices * np.sqrt(1 / mesh.area)

        return mesh, Center

    def alignMesh(self):
        # Convert files to .ply
        self.convert()

        mesh_dir = self.settings["mesh_dir"]
        num_subsample = [
            self.settings["num_subsample_low"],
            self.settings["num_subsample_high"],
        ]
        seed = self.settings["seed"]

        dataset_coll = auto3dgm_nazar.dataset.datasetfactory.DatasetFactory.ds_from_dir(
            mesh_dir
        )
        self.originalMeshes = dataset_coll.datasets[0]

        logger.info("Subsampling meshes...")
        sample_time = time.time()
        if seed:
            ss = auto3dgm_nazar.mesh.subsample.Subsample(
                pointNumber=num_subsample,
                meshes=self.originalMeshes,
                seed=seed,
                center_scale=False,
            )
        else:
            ss = auto3dgm_nazar.mesh.subsample.Subsample(
                pointNumber=num_subsample,
                meshes=self.originalMeshes,
                center_scale=False,
            )
        self.subsample = ss

        ss_res = ss.ret

        low_res_meshes = []
        for name, mesh in ss_res[num_subsample[0]]["output"].items():
            mesh.name = name
            newMesh = auto3dgm_nazar.mesh.meshfactory.MeshFactory.mesh_from_data(
                mesh.koodinimi, center_scale=True, name=mesh.name
            )
            low_res_meshes.append(newMesh)

        self.sampledMeshes = []
        for name, mesh in ss_res[num_subsample[1]]["output"].items():
            mesh.name = name
            newMesh = auto3dgm_nazar.mesh.meshfactory.MeshFactory.mesh_from_data(
                mesh.koodinimi, center_scale=True, name=mesh.name
            )
            self.sampledMeshes.append(newMesh)
        sample_time = time.time() - sample_time
        logger.info(f"Subsampling completed in {sample_time:.2f} seconds.")

        # Align low resolution meshes
        logger.info("Aligning low-resolution meshes...")
        low_res_time = time.time()
        mirror = self.settings["reflection"]
        corr = auto3dgm_nazar.analysis.correspondence.Correspondence(
            meshes=low_res_meshes, mirror=mirror
        )
        low_res_time = time.time() - low_res_time
        logger.info(
            f"Low-resolution alignment completed in {low_res_time:.2f} seconds."
        )

        # Align high resolution meshes
        logger.info("Aligning high-resolution meshes...")
        high_res_time = time.time()
        ga = corr.globalized_alignment
        self.alignData = auto3dgm_nazar.analysis.correspondence.Correspondence(
            meshes=self.sampledMeshes, mirror=mirror, initial_alignment=ga
        )
        high_res_time = time.time() - high_res_time
        logger.info(
            f"High-resolution alignment completed in {high_res_time:.2f} seconds."
        )

        # Save aligned meshes
        logger.info("Saving aligned meshes...")
        output_dir = os.path.join(self.settings["output_dir"], "alignedMeshes/")
        self.touch(output_dir)

        # Viewer directory
        viewer_dir = os.path.join(self.settings["output_dir"], "viewer/aligned_meshes/")
        self.touch(viewer_dir)

        # Normalize and export meshes
        time_save = time.time()
        for t in range(len(self.originalMeshes)):
            R = self.alignData.globalized_alignment["r"][t]

            verts = self.originalMeshes[t].vertices
            faces = self.originalMeshes[t].faces
            name = self.originalMeshes[t].name

            vertices = np.transpose(np.matmul(R, np.transpose(verts)))
            faces = faces.astype("int64")

            aligned_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False
            )
            if np.linalg.det(R) < 0:
                aligned_mesh.faces = aligned_mesh.faces[:, [0, 2, 1]]
            aligned_mesh, _ = self.Centralize(aligned_mesh, scale=None)
            aligned_mesh.export(os.path.join(output_dir, name + ".ply"))
            name = name.replace("_", "-")
            aligned_mesh, _ = self.Centralize(aligned_mesh, scale=1)
            aligned_mesh.export(os.path.join(viewer_dir, name + ".obj"))

        time_save = time.time() - time_save
        logger.info(f"Aligned meshes saved in {time_save:.2f} seconds.")

    def landmarksFromPseudoLandmarks(
        self, subsampledMeshes, permutations, rotations, origScale=False
    ):
        meshes = []
        for i in range(len(subsampledMeshes)):
            mesh = self.sampledMeshes[i]
            perm = permutations[i]
            rot = rotations[i]
            V = mesh.vertices
            lmtranspose = V.T @ perm
            landmarks = np.transpose(np.matmul(rot, lmtranspose))
            unscaledLandmarks = landmarks * mesh.initial_scale
            if not origScale:
                mesh = auto3dgm_nazar.mesh.meshfactory.MeshFactory.mesh_from_data(
                    vertices=landmarks, name=mesh.name, center_scale=False, deep=True
                )
                meshes.append(mesh)
            else:
                mesh = auto3dgm_nazar.mesh.meshfactory.MeshFactory.mesh_from_data(
                    vertices=unscaledLandmarks,
                    name=mesh.name,
                    center_scale=False,
                    deep=True,
                )
                meshes.append(mesh)

        return meshes

    def exportAlignedLandmarksNew(self):
        output = self.settings["output_dir"]
        exportFolder = os.path.join(output, "scaled_landmarks/")
        unscaleOutput = os.path.join(output, "unscaled_landmarks/")
        self.touch(unscaleOutput)
        self.touch(exportFolder)
        m = self.sampledMeshes
        r = self.alignData.globalized_alignment["r"]
        p = self.alignData.globalized_alignment["p"]
        landmarks = self.landmarksFromPseudoLandmarks(m, p, r, origScale=False)
        unscaledLandmarks = self.landmarksFromPseudoLandmarks(m, p, r, origScale=True)

        # Create Pandas Dataframe
        colNames = ["Name"]
        for i in range(1, len(landmarks[0].vertices) + 1):
            idx = str(i)
            vals = ["X" + idx, "Y" + idx, "Z" + idx]
            colNames.extend(vals)

        dfLandmarks = pd.DataFrame(columns=colNames)

        for l in landmarks:
            self.saveNumpyArrayToFcsv(l.vertices, os.path.join(exportFolder, l.name))
            data = [l.name]
            verts = np.array(l.vertices)
            data.extend(verts.flatten())
            dfLandmarks.loc[len(dfLandmarks)] = data

        # Save landmarks
        dfLandmarks.to_csv(os.path.join(output, "landmarks_scaled.csv"), index=False)

        dfUnscaledLandmarks = pd.DataFrame(columns=colNames)

        for l in unscaledLandmarks:
            self.saveNumpyArrayToFcsv(l.vertices, os.path.join(unscaleOutput, l.name))
            data = [l.name]
            verts = np.array(l.vertices)
            data.extend(verts.flatten())
            dfUnscaledLandmarks.loc[len(dfUnscaledLandmarks)] = data

        dfUnscaledLandmarks.to_csv(
            os.path.join(output, "landmarks_unscaled.csv"), index=False
        )

        # Write morphologika files
        self.writeMorphologikaFile(
            landmarks, dfLandmarks, os.path.join(output, "morphologika_scaled.txt")
        )
        self.writeMorphologikaFile(
            unscaledLandmarks,
            dfUnscaledLandmarks,
            os.path.join(output, "morphologika_unscaled.txt"),
        )

    def writeMorphologikaFile(self, landmarks, dfLandmarks, filepath):
        with open(filepath, "w") as fid:
            fid.write("[Individuals]\n")
            fid.write(str(dfLandmarks.shape[0]) + "\n")
            fid.write("[Landmarks]\n")
            fid.write(str(self.settings["num_subsample_high"]) + "\n")
            fid.write("[dimensions]\n")
            fid.write("3\n")
            fid.write("[names]\n")
            for l in landmarks:
                fid.write(l.name + "\n")
            fid.write("\n")
            fid.write("[rawpoints]\n")
            for l in landmarks:
                fid.write("\n")
                fid.write("'" + l.name + "\n")
                fid.write("\n")
                for i in range(l.vertices.shape[0]):
                    fid.write(
                        "{:.7e}".format(l.vertices[i, 0])
                        + " "
                        + "{:.7e}".format(l.vertices[i, 1])
                        + " "
                        + "{:.7e}".format(l.vertices[i, 2])
                        + "\n"
                    )

    def saveNumpyArrayToFcsv(self, array, filename):
        l = np.shape(array)[0]
        fname2 = filename + ".fcsv"
        with open(fname2, "w") as file:
            file.write("# Markups fiducial file version = 4.4 \n")
            file.write("# columns = id,x,y,z,vis,sel \n")
            for i in range(l):
                file.write("p" + str(i) + ",")
                file.write(
                    str(array[i, 0])
                    + ","
                    + str(array[i, 1])
                    + ","
                    + str(array[i, 2])
                    + ",1,1 \n"
                )

    def touch(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def saveNumpyArrayToCsv(self, array, filename):
        np.savetxt(filename + ".csv", array, delimiter=",", fmt="%s")

    def exportRotationsScaleTranslation(self):
        output = self.settings["output_dir"]
        rotations = dict()
        scale_mat = dict()
        translation_mat = dict()
        exportFolder = os.path.join(output, "rotations/")
        self.touch(exportFolder)
        m = self.sampledMeshes
        r = self.alignData.globalized_alignment["r"]

        for idx in range(len(m)):
            mesh = m[idx]
            rot = r[idx]
            rotations[mesh.name] = rot
            self.saveNumpyArrayToCsv(rot, os.path.join(exportFolder, mesh.name))

        exportFolder = os.path.join(output, "scale/")
        self.touch(exportFolder)

        meshes = self.originalMeshes
        for idx in range(len(meshes)):
            filename = os.path.join(exportFolder, meshes[idx].name)
            curMesh = trimesh.Trimesh(
                vertices=meshes[idx].vertices, faces=meshes[idx].faces, process=False
            )
            curArea = np.sqrt(1.0 / (curMesh.area + 0.0)) * np.diag([1, 1, 1])
            scale_mat[meshes[idx].name] = curArea
            self.saveNumpyArrayToCsv(curArea, filename)

        exportFolder = os.path.join(output, "translation/")
        self.touch(exportFolder)
        for idx in range(len(meshes)):
            filename = os.path.join(exportFolder, meshes[idx].name)
            Center = np.mean(meshes[idx].vertices, 0).reshape(1, 3)
            translation_mat[meshes[idx].name] = Center
            self.saveNumpyArrayToCsv(Center, filename)

        # Create scale outputs
        fields = ["Name", "Rotations", "", "", "", "Scale", "", "", "", "Translation"]
        filename = os.path.join(output, "rotation_scale_translation.csv")

        with open(filename, "w+", newline="") as f:
            write = csv.writer(f)
            write.writerow(fields)

            for mesh in rotations.keys():
                row = [mesh]
                row.extend(rotations[mesh][0])
                row.append("")
                row.extend(scale_mat[mesh][0])
                row.append("")
                row.extend(translation_mat[mesh][0])
                write.writerow(row)

                for i in range(1, len(rotations[mesh])):
                    row = [""]
                    row.extend(rotations[mesh][i])
                    row.append("")
                    row.extend(scale_mat[mesh][i])
                    write.writerow(row)

    def compute_procrustes_distance(self, mesh1, mesh2):
        # Ensure both meshes have the same number of vertices
        if mesh1.vertices.shape != mesh2.vertices.shape:
            raise ValueError(
                "Meshes must have the same number of vertices for Procrustes analysis."
            )

        # Center the meshes
        mesh1_centered = mesh1.vertices - np.mean(mesh1.vertices, axis=0)
        mesh2_centered = mesh2.vertices - np.mean(mesh2.vertices, axis=0)

        # Normalize the meshes
        norm1 = np.linalg.norm(mesh1_centered)
        norm2 = np.linalg.norm(mesh2_centered)
        mesh1_normalized = mesh1_centered / norm1
        mesh2_normalized = mesh2_centered / norm2

        # Compute the Procrustes distance
        matrix, disparity = scipy.spatial.procrustes(mesh1_normalized, mesh2_normalized)
        procrustes_distance = np.sqrt(disparity)
        return procrustes_distance

    def compute_similarity_scores(self):
        logger.info("Computing similarity scores between aligned meshes...")
        aligned_meshes_dir = os.path.join(self.settings["output_dir"], "alignedMeshes/")
        mesh_files = [f for f in os.listdir(aligned_meshes_dir) if f.endswith(".ply")]
        mesh_files.sort()  # Ensure consistent ordering

        # Load meshes
        meshes = {}
        for mesh_file in mesh_files:
            mesh_path = os.path.join(aligned_meshes_dir, mesh_file)
            mesh = trimesh.load(mesh_path, process=False)
            meshes[mesh_file] = mesh

        # Compute pairwise Procrustes distances
        results = []
        for i in range(len(mesh_files)):
            for j in range(i + 1, len(mesh_files)):
                mesh_name1 = mesh_files[i]
                mesh_name2 = mesh_files[j]
                mesh1 = meshes[mesh_name1]
                mesh2 = meshes[mesh_name2]
                try:
                    distance = self.compute_procrustes_distance(mesh1, mesh2)
                    results.append(
                        {
                            "Mesh1": mesh_name1,
                            "Mesh2": mesh_name2,
                            "ProcrustesDistance": distance,
                        }
                    )
                    logger.info(
                        f"Computed distance between {mesh_name1} and {mesh_name2}: {distance}"
                    )
                except Exception as e:
                    logger.exception(
                        f"Error computing distance between {mesh_name1} and {mesh_name2}: {e}"
                    )

        # Save results to CSV
        similarity_scores_file = os.path.join(
            self.settings["output_dir"], "auto3dgm_similarity_scores.csv"
        )
        df = pd.DataFrame(results)
        df.to_csv(similarity_scores_file, index=False)
        logger.info(f"Similarity scores saved to {similarity_scores_file}")

    def run(self):
        total_time = time.time()
        try:
            self.validate_settings()
            self.alignMesh()
            self.exportAlignedLandmarksNew()
            self.exportRotationsScaleTranslation()
            self.compute_similarity_scores()
            total_time = time.time() - total_time
            logger.info(f"Total run time: {total_time:.2f} seconds.")
            logger.info("Alignment and similarity computation completed successfully.")
        except Exception as e:
            logger.exception(f"An error occurred during execution: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Auto3DGM Command Line Interface")

    parser.add_argument(
        "--num-subsample-low",
        type=int,
        default=100,
        help="Number of subsampled points (low resolution)",
    )
    parser.add_argument(
        "--num-subsample-high",
        type=int,
        default=200,
        help="Number of subsampled points (high resolution)",
    )
    parser.add_argument("--seed", type=int, default=None, help="FPV Seed (Optional)")
    parser.add_argument(
        "--num-cores",
        type=int,
        default=psutil.cpu_count(logical=False),
        help="Number of cores to use",
    )
    parser.add_argument("--single-core", action="store_true", help="Use single core")
    parser.add_argument(
        "--sample-method",
        type=str,
        default="FPS",
        choices=["FPS", "GPL", "Hybrid"],
        help="Sampling method",
    )
    parser.add_argument("--reflection", action="store_true", help="Use reflection")
    parser.add_argument(
        "--mesh-dir", type=str, required=True, help="Directory containing mesh files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.getcwd(), help="Output directory"
    )
    parser.add_argument("--settings-json", type=str, help="Path to settings JSON file")
    parser.add_argument(
        "--log_file", type=str, default="auto3dgm_cli.log", help="Log file path"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    auto3dgm_cli = Auto3DGM_CLI(args)
    auto3dgm_cli.run()


if __name__ == "__main__":
    main()
