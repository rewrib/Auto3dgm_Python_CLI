import sqlite3
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

DB_FILE = r"/home/batest/Projects/BA/output/Morphosource/Database/downloads_metadata-test8.db"  # make sure this matches your script

def load_auto3dgm_similarity_scores():
    """
    Load (id1, id2, similarity) from auto3dgm_relationships, 
    and also fetch file names from the downloads table for labeling.

    Returns:
       - mesh_ids: a list of download-IDs (the primary keys from downloads.id)
       - file_names: a parallel list of file_name (strings) for labeling
       - distance_matrix: NxN NumPy array of distances = 1 - (similarity/100)
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Query all relationships, along with file_name from downloads
    # so we can label each row in the dendrogram
    cur.execute("""
        SELECT r.id1, r.id2, r.similarity, d1.file_name AS file1, d2.file_name AS file2
        FROM auto3dgm_relationships r
        JOIN downloads d1 ON r.id1 = d1.id
        JOIN downloads d2 ON r.id2 = d2.id
        ORDER BY r.id1, r.id2
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No similarity data found in auto3dgm_relationships.")
        return None, None, None

    # Gather all unique IDs
    id_set = set()
    for (id1, id2, sim, f1, f2) in rows:
        id_set.add(id1)
        id_set.add(id2)
    mesh_ids = sorted(list(id_set))

    # Build an index map for ID -> row/column in the distance matrix
    id_to_index = {mid: i for i, mid in enumerate(mesh_ids)}
    n = len(mesh_ids)

    # Initialize a distance matrix (NxN). We'll fill in below.
    distance_matrix = np.zeros((n, n), dtype=float)

    # We also want a stable label for each ID. 
    # Let's do a second query or build a dictionary from the rows we already have.
    # However, the above rows don't guarantee coverage of every ID's file_name 
    # if an ID only appears in the second half of a pair. We can do a direct fetch:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # build a dictionary: id -> file_name
    id_to_name = {}
    for mid in mesh_ids:
        cur.execute("SELECT file_name FROM downloads WHERE id=?", (mid,))
        result = cur.fetchone()
        if result:
            id_to_name[mid] = result[0]
        else:
            id_to_name[mid] = f"Unknown_{mid}"
    conn.close()

    # Fill in the distance matrix from relationships
    # distance = 1 - (similarity / 100)
    for (id1, id2, similarity, file1, file2) in rows:
        i = id_to_index[id1]
        j = id_to_index[id2]
        dist = 1.0 - (similarity / 100.0)
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

    # Convert id_to_name into a list of labels in index order
    file_names = [id_to_name[mid] for mid in mesh_ids]

    return mesh_ids, file_names, distance_matrix


def hierarchical_clustering_auto3dgm():
    """
    Perform hierarchical clustering on data from auto3dgm_relationships
    and plot a dendrogram labeled by file_name.
    """
    mesh_ids, file_names, distance_matrix = load_auto3dgm_similarity_scores()
    if mesh_ids is None:
        return  # No data

    # SciPy's linkage needs a condensed distance format
    condensed = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(condensed, method="ward")

    # Plot the dendrogram
    plt.figure(figsize=(10, 6))
    sch.dendrogram(
        linkage_matrix,
        labels=file_names,  # Use file names (or mesh_ids) for labeling each leaf
        leaf_rotation=90
    )
    plt.title("Hierarchical Clustering (auto3dgm_relationships)")
    plt.xlabel("File name")
    plt.ylabel("Distance (Ward linkage)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage: run after your DB integration has finished
    # run_auto3dgm_db_integration()  # ensure auto3dgm_relationships is populated
    hierarchical_clustering_auto3dgm()
