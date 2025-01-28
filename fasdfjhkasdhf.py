def exportAlignedLandmarksNew(self, output):
    """
    Exports two types of landmarks:
    1) 'scaled_landmarks': The landmarks that have been centered and unit-scaled
    2) 'unscaled_landmarks': The landmarks that have been rotated/aligned but preserve original mesh scale

    In addition, exports to:
      - CSV files (landmarks_scaled.csv, landmarks_unscaled.csv)
      - FCSV files (per-mesh)
      - Morphologika format
      - JSON files (one per mesh) that can be used in JavaScript with Three.js
    """
    exportFolder = output + "scaled_landmarks/"
    unscaleOutput = output + "unscaled_landmarks/"
    self.touch(unscaleOutput)
    self.touch(exportFolder)

    # Extract relevant alignment data
    m = self.sampledMeshes
    r = self.alignData.globalized_alignment["r"]
    p = self.alignData.globalized_alignment["p"]

    # Landmarks with and without original scale
    landmarks = self.landmarksFromPseudoLandmarks(m, p, r, origScale=False)
    unscaledLandmarks = self.landmarksFromPseudoLandmarks(m, p, r, origScale=True)

    # --------------------------------------------------------------------------
    # 1) CREATE & SAVE A CSV SUMMARY FOR SCALED LANDMARKS
    # --------------------------------------------------------------------------
    colNames = ["Name"]
    # The number of landmarks is the same for each mesh
    num_landmarks = len(landmarks[0].vertices)
    for i in range(1, num_landmarks + 1):
        idx = str(i)
        vals = ["X" + idx, "Y" + idx, "Z" + idx]
        colNames.extend(vals)

    dfLandmarks = pd.DataFrame(columns=colNames)

    for l in landmarks:
        # FCSV export (Slicer-friendly markup format)
        self.saveNumpyArrayToFcsv(l.vertices, os.path.join(exportFolder, l.name))

        # CSV DataFrame row
        data = [l.name]
        verts = np.array(l.vertices)
        data.extend(verts.flatten())
        dfLandmarks.loc[len(dfLandmarks)] = data

    # Write all scaled landmarks to a single CSV file
    dfLandmarks.to_csv(os.path.join(output, "landmarks_scaled.csv"), index=False)

    # --------------------------------------------------------------------------
    # 2) CREATE & SAVE A CSV SUMMARY FOR UNSCALED LANDMARKS
    # --------------------------------------------------------------------------
    dfUnscaledLandmarks = pd.DataFrame(columns=colNames)

    for l in unscaledLandmarks:
        # FCSV export
        self.saveNumpyArrayToFcsv(l.vertices, os.path.join(unscaleOutput, l.name))

        # CSV DataFrame row
        data = [l.name]
        verts = np.array(l.vertices)
        data.extend(verts.flatten())
        dfUnscaledLandmarks.loc[len(dfUnscaledLandmarks)] = data

    # Write all unscaled landmarks to a single CSV file
    dfUnscaledLandmarks.to_csv(
        os.path.join(output, "landmarks_unscaled.csv"), index=False
    )

    # --------------------------------------------------------------------------
    # 3) WRITE JSON FILES FOR USE WITH THREE.JS (or other JS frameworks)
    #    One JSON file per mesh, containing the landmark coordinates.
    # --------------------------------------------------------------------------
    # SCALED landmarks
    for l in landmarks:
        json_data = {
            "name": l.name,
            # Convert NumPy array to a regular Python list
            # so it can be JSON-serialized.
            "landmarks": l.vertices.tolist(),
        }
        json_filename = os.path.join(exportFolder, l.name + ".json")
        with open(json_filename, "w") as f:
            json.dump(json_data, f, indent=2)

    # UNSCALED landmarks
    for l in unscaledLandmarks:
        json_data = {"name": l.name, "landmarks": l.vertices.tolist()}
        json_filename = os.path.join(unscaleOutput, l.name + ".json")
        with open(json_filename, "w") as f:
            json.dump(json_data, f, indent=2)

    # --------------------------------------------------------------------------
    # 4) WRITE MORPHOLOGIKA FILES FOR BOTH SCALED AND UNSCALED
    #    [Optional; remove if not needed]
    # --------------------------------------------------------------------------
    # Scaled
    fname_scaled = os.path.join(output, "morphologika_scaled.txt")
    with open(fname_scaled, "w") as fid:
        fid.write("[Individuals]\n")
        fid.write(str(dfLandmarks.shape[0]) + "\n")
        fid.write("[Landmarks]\n")
        fid.write(str(num_landmarks) + "\n")
        fid.write("[dimensions]\n")
        fid.write("3\n")
        fid.write("[names]\n")
        for l in landmarks:
            fid.write(l.name + "\n")
        fid.write("\n[rawpoints]\n")

        for l in landmarks:
            fid.write("\n'" + l.name + "\n\n")
            for i in range(l.vertices.shape[0]):
                fid.write(
                    "{:.7e} {:.7e} {:.7e}\n".format(
                        l.vertices[i, 0], l.vertices[i, 1], l.vertices[i, 2]
                    )
                )

    # Unscaled
    fname_unscaled = os.path.join(output, "morphologika_unscaled.txt")
    with open(fname_unscaled, "w") as fid:
        fid.write("[Individuals]\n")
        fid.write(str(dfUnscaledLandmarks.shape[0]) + "\n")
        fid.write("[Landmarks]\n")
        fid.write(str(num_landmarks) + "\n")
        fid.write("[dimensions]\n")
        fid.write("3\n")
        fid.write("[names]\n")
        for l in unscaledLandmarks:
            fid.write(l.name + "\n")
        fid.write("\n[rawpoints]\n")

        for l in unscaledLandmarks:
            fid.write("\n'" + l.name + "\n\n")
            for i in range(l.vertices.shape[0]):
                fid.write(
                    "{:.7e} {:.7e} {:.7e}\n".format(
                        l.vertices[i, 0], l.vertices[i, 1], l.vertices[i, 2]
                    )
                )

    print("Landmarks exported (FCSV, CSV, Morphologika, and JSON).")
