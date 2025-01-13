import os

import pymeshlab as pml

# Paths to meshes
MESH_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2"
OUTPUT_DIR = r"D:\Uni\BA\output\Morphosource\Meshes2_cleaned"

NOT_SIMPLY_CONNECTED_DIR = os.path.join(OUTPUT_DIR, "NotSimplyConnected")
DISC_DIR = os.path.join(OUTPUT_DIR, "DiscTopology")
SPHERE_DIR = os.path.join(OUTPUT_DIR, "SphereTopology")
BAD_DIR = os.path.join(OUTPUT_DIR, "BadMeshes")

# number of smoothing iterations
NUM_SMOOTH = 2


def touch(newDir):
    if not os.path.isdir(newDir):
        os.mkdir(newDir)


def try_manifold_repairs(ms, out_dict, cnt):
    while not out_dict["is_mesh_two_manifold"]:
        repair_mesh(ms)
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
    return out_dict


def repair_mesh(ms):
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()


def keep_largest_component(ms, out_dict):
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


def subdivide_if_needed(ms):
    cnt = 20
    while ms.current_mesh().face_number() < 10000:
        ms.meshing_surface_subdivision_loop(
            loopweight=1, iterations=1, threshold=pml.Percentage(0)
        )
        cnt = cnt - 1
        if cnt == 0:
            break
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000, autoclean=True)


def try_manifold_repairs2(ms, cnt):
    out_dict = ms.get_topological_measures()
    while not out_dict["is_mesh_two_manifold"]:
        repair_mesh(ms)
        keep_largest_component(ms, out_dict)
        msTemp = pml.MeshSet()
        msTemp.add_mesh(ms.current_mesh())
        ms = msTemp
        out_dict = ms.get_topological_measures()

        cnt = cnt + 1
        if cnt == 10:
            out_dict
            break
    return ms


def final_topology_classification(
    notSimplyConnectedDir, discDir, sphereDir, badDir, mesh, ms
):
    ms.meshing_re_orient_faces_coherentely()
    out_dict = ms.get_topological_measures()

    if out_dict["connected_components_number"] > 1:
        ms.save_current_mesh(os.path.join(badDir, mesh))
        print(f"{mesh}: ConnectedComponentIssue", flush=True)
    elif out_dict["genus"] > 0:
        ms.save_current_mesh(os.path.join(notSimplyConnectedDir, mesh))
        print(f"{mesh}: NotSimplyConnected", flush=True)
    elif out_dict["boundary_edges"] > 0:
        ms.save_current_mesh(os.path.join(discDir, mesh))
        print(f"{mesh}: Disc", flush=True)
    else:
        ms.save_current_mesh(os.path.join(sphereDir, mesh))
        print(f"{mesh}: Sphere", flush=True)


if __name__ == "__main__":
    meshList = os.listdir(MESH_DIR)
    touch(OUTPUT_DIR)
    touch(NOT_SIMPLY_CONNECTED_DIR)
    touch(DISC_DIR)
    touch(SPHERE_DIR)
    touch(BAD_DIR)

    for mesh in meshList:
        try:
            print(mesh, flush=True)
            ms = pml.MeshSet()

            meshPath = os.path.join(MESH_DIR, mesh)
            ms.load_new_mesh(meshPath)

            ms.set_current_mesh(0)
            # ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(20))
            out_dict = ms.get_topological_measures()

            cnt = 0
            out_dict = try_manifold_repairs(ms, out_dict, cnt)
            keep_largest_component(ms, out_dict)
            msTemp = pml.MeshSet()
            msTemp.add_mesh(ms.current_mesh())
            ms = msTemp

            subdivide_if_needed(ms)

            for j in range(NUM_SMOOTH):
                ms.apply_coord_hc_laplacian_smoothing()

            cnt = 0
            out_dict = ms.get_topological_measures()
            keep_largest_component(ms, out_dict)
            msTemp = pml.MeshSet()
            msTemp.add_mesh(ms.current_mesh())
            ms = msTemp
            ms = try_manifold_repairs(ms, cnt)
            for prefix in [BAD_DIR, NOT_SIMPLY_CONNECTED_DIR, DISC_DIR, SPHERE_DIR]:
                meshPath = os.path.join(prefix, mesh)
                if os.path.isfile(meshPath):
                    os.remove(meshPath)

            ms.meshing_close_holes(
                maxholesize=30, newfaceselected=True, selfintersection=True
            )
            ms.meshing_surface_subdivision_loop(
                loopweight=1, iterations=2, selected=True
            )
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pml.Percentage(20)
            )
            ms.meshing_close_holes(
                maxholesize=30, newfaceselected=True, selfintersection=True
            )
            out_dict = ms.get_topological_measures()
            keep_largest_component(ms, out_dict)
            try:
                msTemp = pml.MeshSet()
                msTemp.add_mesh(ms.current_mesh())
                ms = msTemp
                final_topology_classification(
                    NOT_SIMPLY_CONNECTED_DIR, DISC_DIR, SPHERE_DIR, BAD_DIR, mesh, ms
                )
            except:
                ms.save_current_mesh(os.path.join(BAD_DIR, mesh))
                print(f"{mesh}: BadMesh", flush=True)
        except Exception as e:
            print(f"Error loading mesh: {e}", flush=True)
            continue
