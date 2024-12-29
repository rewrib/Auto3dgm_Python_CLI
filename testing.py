#!/usr/bin/env python3

import os
import shutil

import numpy as np

from auto3dgm_nazar.analysis.correspondence import Correspondence

# auto3dgm_nazar must be installed.
# pip install git+https://github.com/toothandclaw/auto3dgm_nazar.git
from auto3dgm_nazar.dataset.datasetfactory import DatasetFactory
from auto3dgm_nazar.mesh.meshexport import MeshExport
from auto3dgm_nazar.mesh.meshfactory import MeshFactory
from auto3dgm_nazar.mesh.subsample import Subsample


class Auto3dgmData:
    """
    Stores module-level results and data not stored in a GUI.
    Holds references to dataset collection, sampled points, etc.
    """

    def __init__(self):
        self.dataset = None
        self.datasetCollection = None
        self.phase1SampledPoints = None
        self.phase2SampledPoints = None
        self.aligned_meshes = []


class Auto3dgmLogic:
    """
    Core logic class for Auto3dgm functionality,
    stripped of any 3D Slicer dependencies.
    """

    @staticmethod
    def runAll(Auto3dgmData, mirror=False):
        """
        Run all possible analysis steps (subsampling,
        Phase 1 alignment, Phase 2 alignment).
        """
        Auto3dgmLogic.subsample(
            Auto3dgmData,
            [Auto3dgmData.phase1SampledPoints, Auto3dgmData.phase2SampledPoints],
            Auto3dgmData.datasetCollection.datasets[0],
        )
        print("Subsampling complete.")

        # Phase 1
        Auto3dgmData.datasetCollection.add_analysis_set(
            Auto3dgmLogic.correspondence(Auto3dgmData, mirror, phase=1), "Phase 1"
        )
        print("Phase 1 complete.")

        # Phase 2
        Auto3dgmData.datasetCollection.add_analysis_set(
            Auto3dgmLogic.correspondence(Auto3dgmData, mirror, phase=2), "Phase 2"
        )
        print("Phase 2 complete.")

    @staticmethod
    def createDataset(inputdirectory):
        """
        Load meshes from a directory and return
        a single dataset (centered and scaled).
        """
        dataset = DatasetFactory.ds_from_dir(inputdirectory, center_scale=True)
        return dataset

    @staticmethod
    def subsample(Auto3dgmData, list_of_pts, meshes):
        """
        Subsample the given meshes at various resolutions
        (list_of_pts is a list of integer sampling sizes).
        """
        ss = Subsample(
            pointNumber=list_of_pts, meshes=meshes, seed={}, center_scale=True
        )
        for point in list_of_pts:
            new_dataset = {}
            new_dataset[point] = [
                ss.ret[point]["output"]["output"][key]
                for key in ss.ret[point]["output"]["output"]
            ]
            Auto3dgmData.datasetCollection.add_dataset(new_dataset, point)
        return Auto3dgmData

    @staticmethod
    def createDatasetCollection(dataset, name):
        """
        Create a dataset collection from a single dataset.
        This is typically used if not using the 'ds_from_dir' approach.
        """
        from auto3dgm_nazar.dataset.datasetcollection import DatasetCollection

        datasetCollection = DatasetCollection(datasets=[dataset], dataset_names=[name])
        return datasetCollection

    @staticmethod
    def checkMeshQuality(meshes):
        """
        Check whether any mesh contains NaN values;
        prints a warning if any do.
        """
        badMesh = []
        for mesh in meshes:
            if np.isnan(np.sum(mesh.vertices)):
                print(f"Found NaN in mesh: {mesh.name}")
                badMesh.append(mesh.name)

        if len(badMesh) > 0:
            print("Mesh quality check failed. The following meshes have NaNs:")
            for name in badMesh:
                print(f" - {name}")
        else:
            print("Mesh quality check passed.")

    @staticmethod
    def correspondence(Auto3dgmData, mirror=False, phase=1):
        """
        Compute a new Correspondence object for Phase 1 or Phase 2.
        """
        if phase == 1:
            npoints = Auto3dgmData.phase1SampledPoints
            label = "Phase 1"
        else:
            npoints = Auto3dgmData.phase2SampledPoints
            label = "Phase 2"
        meshes = Auto3dgmData.datasetCollection.datasets[npoints][npoints]
        corr = Correspondence(meshes=meshes, mirror=mirror)
        print(f"Correspondence computed for {label}")
        return corr

    @staticmethod
    def landmarksFromPseudoLandmarks(subsampledMeshes, permutations, rotations):
        """
        Convert 'pseudo-landmarks' to real landmarks by applying
        rotations and permutations.
        """
        output_meshes = []
        for i in range(len(subsampledMeshes)):
            mesh = subsampledMeshes[i]
            perm = permutations[i]
            rot = rotations[i]
            # Original subsampled vertices
            V = mesh.vertices
            scaledV = mesh.initial_vertices  # these are the pre-center-scale vertices

            lmtranspose = scaledV.T @ perm
            aligned = np.transpose(np.matmul(rot, lmtranspose))

            newMesh = MeshFactory.mesh_from_data(
                vertices=aligned, name=mesh.name, center_scale=False, deep=True
            )
            output_meshes.append(newMesh)
        return output_meshes

    @staticmethod
    def landmarksOSSFromPseudoLandmarks(subsampledMeshes, permutations):
        """
        Create a set of 'one-step scaled' landmarks by
        applying only permutations (no rotation).
        """
        output_meshes = []
        for i in range(len(subsampledMeshes)):
            mesh = subsampledMeshes[i]
            perm = permutations[i]
            scaledV = mesh.initial_vertices

            lmtranspose = scaledV.T @ perm
            aligned = np.transpose(lmtranspose)

            newMesh = MeshFactory.mesh_from_data(
                vertices=aligned, name=mesh.name, center_scale=False, deep=True
            )
            output_meshes.append(newMesh)
        return output_meshes

    @staticmethod
    def saveNumpyArrayToCsv(array, filename):
        """
        Save a numpy array as CSV file with the given filename (no extension).
        """
        np.savetxt(filename + ".csv", array, delimiter=",", fmt="%s")

    @staticmethod
    def saveNumpyArrayToFcsv(array, filename):
        """
        Save a numpy array as an Slicer Markups-like FCSV file,
        but in a standalone manner (no scene nodes).
        """
        l = np.shape(array)[0]
        fname2 = filename + ".fcsv"
        with open(fname2, "w") as file:
            file.write("# Markups fiducial file version = 4.4 \n")
            file.write("# columns = id,x,y,z,vis,sel \n")
            for i in range(l):
                file.write(f"p{i},{array[i,0]},{array[i,1]},{array[i,2]},1,1\n")

    @staticmethod
    def saveTransform(m, filename):
        """
        Save a 4x4 transform matrix to a .tfm file in a style similar
        to ITK/Slicer transform files — but purely text-based,
        no Slicer dependencies.
        """
        filename += ".tfm"
        invm = np.linalg.inv(m)
        with open(filename, "w") as file:
            file.write("#Insight Transform File V1.0\n")
            file.write("#Transform 0\n")
            file.write("Transform: AffineTransform_double_3_3\n")
            file.write("Parameters: ")
            # Write the 3x3 submatrix
            for i in range(3):
                for j in range(3):
                    file.write(f"{m[i, j]} ")
            # Then translation
            file.write(f"{m[0,3]} {m[1,3]} {m[2,3]}\n")
            file.write("FixedParameters: 0 0 0\n")

    @staticmethod
    def alignOriginalMeshes(Auto3dgmData, phase=2):
        """
        Apply the computed rotations from Phase 1 or Phase 2
        to the original full-resolution meshes (datasetCollection.datasets[0]).
        """
        if phase == 1:
            label = "Phase 1"
        elif phase == 2:
            label = "Phase 2"
        else:
            raise ValueError("Invalid phase number. Must be 1 or 2.")

        corr = Auto3dgmData.datasetCollection.analysis_sets[label]
        meshes = Auto3dgmData.datasetCollection.datasets[0]
        Auto3dgmData.aligned_meshes = []

        for t, mesh in enumerate(meshes):
            R = corr.globalized_alignment["r"][t]
            verts = mesh.vertices
            faces = mesh.faces
            name = mesh.name

            # Apply rotation
            new_verts = np.transpose(np.matmul(R, np.transpose(verts)))
            new_faces = faces.astype("int64")

            aligned_mesh = MeshFactory.mesh_from_data(
                new_verts, faces=new_faces, name=name, center_scale=False, deep=True
            )
            Auto3dgmData.aligned_meshes.append(aligned_mesh)
        return Auto3dgmData

    @staticmethod
    def saveAlignedMeshes(Auto3dgmData, outputFolder):
        """
        Save aligned meshes as PLY files to the specified output folder.
        """
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        for mesh in Auto3dgmData.aligned_meshes:
            print(f"Saving aligned mesh: {mesh.name}")
            MeshExport.writeToFile(outputFolder, mesh, format="ply")
        print("Aligned meshes saved.\n")

    @staticmethod
    def exportData(Auto3dgmData, outputFolder, phases=[1, 2]):
        """
        Export all relevant data: aligned meshes, aligned landmarks,
        rotations, scale info, original meshes, etc., for the requested phases.
        """
        acceptable_phases = [1, 2]
        for p in phases:
            if p not in acceptable_phases:
                raise ValueError("Unacceptable phase number passed to exportData.")

            exportFolder = os.path.join(outputFolder, f"phase{p}")
            subDirs = [
                "aligned_meshes",
                "aligned_landmarks",
                "rotation",
                "scale_info",
                "landmarks_OSS",
                "original_meshes",
            ]
            Auto3dgmLogic.prepareDirs(exportFolder, subDirs)

            Auto3dgmLogic.alignOriginalMeshes(Auto3dgmData, p)
            Auto3dgmLogic.saveAlignedMeshes(
                Auto3dgmData, os.path.join(exportFolder, subDirs[0])
            )
            Auto3dgmLogic.exportAlignedLandmarks(
                Auto3dgmData, os.path.join(exportFolder, subDirs[1]), p
            )
            Auto3dgmLogic.exportRotations(
                Auto3dgmData, os.path.join(exportFolder, subDirs[2]), p
            )
            Auto3dgmLogic.exportScaleInfo(
                Auto3dgmData, os.path.join(exportFolder, subDirs[3])
            )
            Auto3dgmLogic.exportLandmarksOSS(
                Auto3dgmData, os.path.join(exportFolder, subDirs[4]), p
            )
            Auto3dgmLogic.exportOriginalMeshes(
                Auto3dgmData, os.path.join(exportFolder, subDirs[5])
            )

        print("All computation and export complete.\n")

    @staticmethod
    def exportAlignedLandmarks(Auto3dgmData, exportFolder, phase=2):
        """
        Export aligned landmarks as FCSV files.
        """
        if phase == 1:
            n = Auto3dgmData.phase1SampledPoints
            label = "Phase 1"
        elif phase == 2:
            n = Auto3dgmData.phase2SampledPoints
            label = "Phase 2"
        else:
            raise ValueError("Invalid phase number in exportAlignedLandmarks.")

        m = Auto3dgmData.datasetCollection.datasets[n][n]
        r = Auto3dgmData.datasetCollection.analysis_sets[label].globalized_alignment[
            "r"
        ]
        p = Auto3dgmData.datasetCollection.analysis_sets[label].globalized_alignment[
            "p"
        ]
        landmarks = Auto3dgmLogic.landmarksFromPseudoLandmarks(m, p, r)

        if not os.path.exists(exportFolder):
            os.makedirs(exportFolder)

        for lmkMesh in landmarks:
            Auto3dgmLogic.saveNumpyArrayToFcsv(
                lmkMesh.vertices, os.path.join(exportFolder, lmkMesh.name)
            )

    @staticmethod
    def exportLandmarksOSS(Auto3dgmData, exportFolder, phase=2):
        """
        Export 'one-step scaled' landmarks,
        ignoring rotation, just permutations.
        """
        if phase == 1:
            n = Auto3dgmData.phase1SampledPoints
            label = "Phase 1"
        elif phase == 2:
            n = Auto3dgmData.phase2SampledPoints
            label = "Phase 2"
        else:
            raise ValueError("Invalid phase number in exportLandmarksOSS.")

        m = Auto3dgmData.datasetCollection.datasets[n][n]
        p = Auto3dgmData.datasetCollection.analysis_sets[label].globalized_alignment[
            "p"
        ]
        landmarks = Auto3dgmLogic.landmarksOSSFromPseudoLandmarks(m, p)

        if not os.path.exists(exportFolder):
            os.makedirs(exportFolder)

        # Retrieve each mesh's original scale and centroid for final restore
        mmesh = Auto3dgmData.datasetCollection.datasets[0]
        scaleDict = {}
        centerDict = {}
        for mesh in mmesh:
            scaleDict[mesh.name] = mesh.initial_scale
            centerDict[mesh.name] = mesh.initial_centroid

        for lmkMesh in landmarks:
            scaleVal = scaleDict[lmkMesh.name]
            centroidVal = centerDict[lmkMesh.name]
            # Re-scale and re-center
            lmkOSS = scaleVal * lmkMesh.vertices + centroidVal
            Auto3dgmLogic.saveNumpyArrayToFcsv(
                lmkOSS, os.path.join(exportFolder, lmkMesh.name)
            )

    @staticmethod
    def exportRotations(Auto3dgmData, exportFolder, phase=2):
        """
        Export rotation matrices as both CSV and .tfm.
        """
        if phase == 1:
            label = "Phase 1"
            n = Auto3dgmData.phase1SampledPoints
        elif phase == 2:
            label = "Phase 2"
            n = Auto3dgmData.phase2SampledPoints
        else:
            raise ValueError("Invalid phase number in exportRotations.")

        m = Auto3dgmData.datasetCollection.datasets[n][n]
        r = Auto3dgmData.datasetCollection.analysis_sets[label].globalized_alignment[
            "r"
        ]

        if not os.path.exists(exportFolder):
            os.makedirs(exportFolder)

        for idx in range(len(m)):
            mesh = m[idx]
            rot = r[idx].copy()

            # Adjust sign in certain cells if needed
            # (Slicer's logic for flipping coordinate systems).
            rot[2][0] = -1 * rot[2][0]
            rot[2][1] = -1 * rot[2][1]
            rot[0][2] = -1 * rot[0][2]
            rot[1][2] = -1 * rot[1][2]

            # Pad into 4x4
            rot = np.vstack((rot.T, [0, 0, 0]))
            rot = np.vstack((rot.T, [0, 0, 0, 1]))

            # Save as CSV + TFM
            filename = os.path.join(exportFolder, mesh.name)
            Auto3dgmLogic.saveNumpyArrayToCsv(rot, filename)
            Auto3dgmLogic.saveTransform(rot, filename)

    @staticmethod
    def exportScaleInfo(Auto3dgmData, exportFolder):
        """
        Export scale matrices (and transform) for the original meshes.
        """
        mmesh = Auto3dgmData.datasetCollection.datasets[0]

        if not os.path.exists(exportFolder):
            os.makedirs(exportFolder)

        for mesh in mmesh:
            filename = os.path.join(exportFolder, mesh.name)
            # Inverse of mesh.initial_scale is the scaling factor
            scaleMat = (1.0 / mesh.initial_scale) * np.eye(3)
            c = (1.0 / mesh.initial_scale) * mesh.initial_centroid

            # Build a 4x4 matrix
            mat4x4 = np.vstack((scaleMat.T, [c[0], c[1], -c[2]]))
            mat4x4 = np.vstack((mat4x4.T, [0, 0, 0, 1]))

            Auto3dgmLogic.saveNumpyArrayToCsv(mat4x4, filename)
            Auto3dgmLogic.saveTransform(mat4x4, filename)

    @staticmethod
    def exportOriginalMeshes(Auto3dgmData, outputFolder):
        """
        Export the original (un-rotated) meshes as PLY.
        They are simply the initial vertices/faces.
        """
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for mesh in Auto3dgmData.datasetCollection.datasets[0]:
            mesh_orig = MeshFactory.mesh_from_data(
                vertices=mesh.initial_vertices,
                faces=mesh.faces,
                name=mesh.name,
                center_scale=False,
                deep=True,
            )
            MeshExport.writeToFile(outputFolder, mesh_orig, format="ply")

    @staticmethod
    def prepareDirs(exportFolder, subDirs=[]):
        """
        Create exportFolder and any requested subdirectories if they don't exist.
        """
        if not os.path.exists(exportFolder):
            os.makedirs(exportFolder)
        for subdir in subDirs:
            d = os.path.join(exportFolder, subdir)
            if not os.path.exists(d):
                os.makedirs(d)

    @staticmethod
    def removeDir(trashFolder):
        if os.path.exists(trashFolder):
            shutil.rmtree(trashFolder)


if __name__ == "__main__":
    """
    Example usage: 
    This small demo shows how you might run the logic in a
    simple, standalone manner.

    Steps:
        1. Create an Auto3dgmData object.
        2. Load meshes (create dataset).
        3. Create a dataset collection and store it in the data object.
        4. Provide subsampling settings (phase1, phase2).
        5. Run the logic (subsample, alignment, export).
    """

    # 1) Initialize data container
    auto3dgmData = Auto3dgmData()

    # 2) Load from some directory (change "myMeshFolder" to real path)
    myMeshFolder = "/home/batest/Projects/BA/output/Morphosource/Meshes2"
    dataset = Auto3dgmLogic.createDataset(myMeshFolder)

    # 3) Create a dataset collection
    auto3dgmData.datasetCollection = Auto3dgmLogic.createDatasetCollection(
        dataset, "MyDataset"
    )

    # 4) Provide sampling settings
    auto3dgmData.phase1SampledPoints = 100
    auto3dgmData.phase2SampledPoints = 200

    # 5) Run alignment
    Auto3dgmLogic.runAll(auto3dgmData, mirror=False)

    # 6) Export results (e.g., to "myOutputFolder")
    myOutputFolder = "/home/batest/Projects/BA/output/Auto3dgm_Python"
    Auto3dgmLogic.exportData(auto3dgmData, myOutputFolder, phases=[1, 2])

    print("Done.")
