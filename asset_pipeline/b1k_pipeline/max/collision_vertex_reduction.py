import trimesh
import numpy as np

import pymxs
rt = pymxs.runtime

MAX_VERTEX_COUNT = 60
REDUCTION_STEP = 5

def reduce_mesh(mesh):   
    # Check its vertex count and reduce as necessary
    reduction_target_vertex_count = MAX_VERTEX_COUNT + REDUCTION_STEP
    reduced_mesh = mesh
    while len(reduced_mesh.vertices) > MAX_VERTEX_COUNT:
        # Reduction function takes a number of faces as its input. We estimate this, if it doesn't
        # work out, we keep trying with a smaller estimate.
        reduction_target_vertex_count -= REDUCTION_STEP
        if reduction_target_vertex_count < MAX_VERTEX_COUNT / 2:
            # Don't want to reduce too far
            raise ValueError("Vertex reduction failed.")

        reduction_target_face_count = int(reduction_target_vertex_count / len(mesh.vertices) * len(mesh.faces))
        reduced_mesh = mesh.simplify_quadratic_decimation(reduction_target_face_count)

    return reduced_mesh.convex_hull


def convert_to_trimesh(obj):
    # Get vertices and faces into numpy arrays for conversion
    verts = np.array([rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))])
    faces = np.array(rt.polyop.getFacesVerts(obj, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj)))) - 1
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    # Split the faces into elements
    elems = {tuple(rt.polyop.GetElementsUsingFace(obj, i + 1)) for i in range(rt.polyop.GetNumFaces(obj))}
    assert len(elems) <= 32, f"{obj.name} should not have more than 32 elements."
    elems = np.array(list(elems))
    assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"
    
    # Iterate through the elements
    meshes = []
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        assert m.is_volume, f"{obj.name} element {i} is not a volume"
        # assert m.is_convex, f"{obj.name} element {i} is not convex"
        assert len(m.split()) == 1, f"{obj.name} element {i} has elements trimesh still finds splittable"
        meshes.append(m)

    return meshes


def process_collision_obj(obj):
    # Get the collision meshes into a set of trimeshes
    collision_meshes = convert_to_trimesh(obj)

    # Check if any of the meshes have too many verts
    if not any(len(m.vertices) > MAX_VERTEX_COUNT for m in collision_meshes):
        return
    
    print("Reducing", obj.name)

    # Individually reduce each of the trimeshes
    reduced_trimeshes = [reduce_mesh(m) for m in collision_meshes]

    # Get a flattened list of vertices and faces
    all_vertices = []
    all_faces = []
    for split in reduced_trimeshes:
        vertices = [rt.Point3(*v.tolist()) for v in split.vertices]
        # Offsetting here by the past vertex count
        faces = [[v + len(all_vertices) + 1 for v in f.tolist()] for f in split.faces]
        all_vertices.extend(vertices)
        all_faces.extend(faces)

    # Delete the original node
    name = obj.name
    parent = obj.parent
    position = obj.position
    rotation = obj.rotation
    rt.delete(obj)
    del obj

    # Create a new node for the collision mesh
    new_obj = rt.Editable_Mesh()
    rt.ConvertToPoly(new_obj)
    new_obj.name = name
    new_obj.position = position
    new_obj.rotation = rotation
    
    # Add the vertices
    for v in all_vertices:
        rt.polyop.createVert(new_obj, v)

    # Add the faces
    for f in all_faces:
        rt.polyop.createPolygon(new_obj, f)

    # Optionally set its wire color
    new_obj.wirecolor = rt.yellow

    # Update the mesh to reflect changes
    rt.update(new_obj)

    # Parent the mesh
    new_obj.parent = parent

    # Check that the new element count is the same as the split count
    elems = {tuple(rt.polyop.GetElementsUsingFace(new_obj, i + 1)) for i in range(rt.polyop.GetNumFaces(new_obj))}
    assert len(elems) == len(collision_meshes), f"{name} has different number of faces in collision mesh than in splits"
    elems = np.array(list(elems))
    assert not np.any(np.sum(elems, axis=0) > 1), f"{name} has same face appear in multiple elements"

    # Hide the mesh
    new_obj.isHidden = True

def process_all_collision_objs():
    for obj in rt.objects:
        if "Mcollision" not in obj.name:
            continue
        
        process_collision_obj(obj)

def main():
    # obj, = rt.selection
    # assert "Mcollision" in obj.name
    # process_collision_obj(obj)

    process_all_collision_objs()

if __name__ == "__main__":
    main()