import collections
from concurrent import futures
import copy
import io
import json
import logging
import os
import traceback
import xml.etree.ElementTree as ET
from xml.dom import minidom

import fs.copy
from fs.tempfs import TempFS
from fs.osfs import OSFS
import networkx as nx
import numpy as np
import tqdm
import trimesh
import trimesh.voxel.creation
from scipy.spatial.transform import Rotation as R
from PIL import Image

from b1k_pipeline import mesh_tree
import b1k_pipeline.utils
from b1k_pipeline.utils import parse_name, get_targets, SUBDIVIDE_CLOTH_CATEGORIES, save_mesh

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

VRAY_MAPPING = {
    "VRayRawDiffuseFilterMap": "albedo",
    "VRayNormalsMap": "normal",
    "VRayMtlReflectGlossinessBake": "roughness",
    "VRayMetalnessMap": "metalness",
    "VRayRawRefractionFilterMap": "opacity",
    "VRaySelfIlluminationMap": "emission",
    "VRayAOMap": "ao",
}

MTL_MAPPING = {
    "map_Kd": "albedo",
    "map_bump": "normal",
    "map_Pr": "roughness",
    "map_Pm": "metalness",
    "map_Tf": "opacity",
    # "map_Ke": "emission",
    # "map_Ks": "ao",
}

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}

CLOTH_SUBDIVISION_THRESHOLD = 0.05

LOG_SURFACE_AREA_RANGE = (-6, 4)
LOG_TEXTURE_RANGE = (4, 11)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_mesh_unit_bbox(mesh, *args, **kwargs):
    mesh_copy = mesh.copy()

    # Find how much the mesh would need to be scaled to fit into a unit cube
    bounding_box = mesh_copy.bounding_box.extents
    assert np.all(bounding_box > 0), f"Bounding box extents are not all positive: {bounding_box}"
    scale = 1 / bounding_box

    # Scale the mesh
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] = np.diag(scale)
    mesh_copy.apply_transform(scale_matrix)

    # Save the scaled mesh
    save_mesh(mesh_copy, *args, **kwargs)

    # Return the inverse scale that needs to be applied for the mesh
    return 1 / scale


def transform_mesh(orig_mesh, translation, rotation):
    mesh = orig_mesh.copy()

    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end

    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(rotation).as_matrix()
    transform[:3, 3] = translation
    inv_transform = trimesh.transformations.inverse_matrix(transform)
    mesh.apply_transform(inv_transform)

    return mesh


def transform_points(points, translation, rotation):
    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(rotation).as_matrix()
    transform[:3, 3] = translation
    inv_transform = trimesh.transformations.inverse_matrix(transform)
    return trimesh.transformations.transform_points(points, inv_transform)


def transform_meta_links(orig_meta_links, translation, rotation):
    meta_links = copy.deepcopy(orig_meta_links)
    rotation_inv = R.from_quat(rotation).inv()
    for meta_link_id_to_subid in meta_links.values():
        for meta_link_subid_to_link in meta_link_id_to_subid.values():
            for meta_link in meta_link_subid_to_link:
                meta_link["position"] = meta_link["position"]
                meta_link["position"] -= translation
                meta_link["position"] = np.dot(rotation_inv.as_matrix(), meta_link["position"])
                # meta_link["position"] += translation
                meta_link["orientation"] = (rotation_inv * R.from_quat(meta_link["orientation"])).as_quat()

    return meta_links


def normalize_meta_links(orig_meta_links, offset):
    meta_links = copy.deepcopy(orig_meta_links)
    for meta_link_id_to_subid in meta_links.values():
        for meta_link_subid_to_link in meta_link_id_to_subid.values():
            for meta_link in meta_link_subid_to_link:
                meta_link["position"] += offset

    return meta_links


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def get_part_nodes(G, root_node):
    root_node_metadata = G.nodes[root_node]["metadata"]
    part_nodes = []
    for part_name in root_node_metadata["parts"]:
        # Find the part node
        part_name_parsed = parse_name(part_name)
        part_cat = part_name_parsed.group("category")
        part_model = part_name_parsed.group("model_id")
        part_inst_id = part_name_parsed.group("instance_id")
        part_link_name = part_name_parsed.group("link_name")
        part_link_name = "base_link" if part_link_name is None else part_link_name
        if part_link_name != "base_link":
            continue
        part_node_key = (part_cat, part_model, part_inst_id, part_link_name)
        if part_node_key not in G.nodes:
            print(list(G.nodes))
        assert part_node_key in G.nodes, f"Could not find part node {part_node_key}"
        part_nodes.append(part_node_key)

    return part_nodes


def get_bbox_data_for_mesh(mesh):
    axis_aligned_bbox = mesh.bounding_box
    axis_aligned_bbox_dict = {
        "extent": np.array(axis_aligned_bbox.primitive.extents).tolist(),
        "transform": np.array(axis_aligned_bbox.primitive.transform).tolist(),
    }

    oriented_bbox = mesh.bounding_box_oriented
    oriented_bbox_dict = {
        "extent": np.array(oriented_bbox.primitive.extents).tolist(),
        "transform": np.array(oriented_bbox.primitive.transform).tolist(),
    }

    return {"axis_aligned": axis_aligned_bbox_dict, "oriented": oriented_bbox_dict}


def compute_link_aligned_bounding_boxes(G, root_node):
    link_bounding_boxes = collections.defaultdict(dict)
    for link_node in nx.dfs_preorder_nodes(G, root_node):
        obj_cat, obj_model, obj_inst_id, link_name = link_node
       
        # Get the pose and transform it
        for key in ["collision", "visual"]:
            try:
                mesh = G.nodes[link_node]["visual_mesh"] if key == "visual" else G.nodes[link_node]["canonical_collision_mesh"]
                link_bounding_boxes[link_name][key] = get_bbox_data_for_mesh(
                    transform_mesh(mesh, -G.nodes[link_node]["mesh_in_link_frame"], [0, 0, 0, 1]))
            except Exception as e:
                print(f"Problem with {obj_cat}-{obj_model} link {link_name}: {str(e)}")

    return link_bounding_boxes


def compute_object_bounding_box(root_node_data):
    combined_mesh = root_node_data["combined_mesh"]
    lower_mesh_center = get_mesh_center(root_node_data["lower_mesh"])
    mesh_orientation = root_node_data["canonical_orientation"]
    canonical_combined_mesh = transform_mesh(combined_mesh, lower_mesh_center, mesh_orientation)
    base_link_offset = canonical_combined_mesh.bounding_box.centroid
    bbox_size = canonical_combined_mesh.bounding_box.extents

    # Compute the bbox world centroid
    bbox_world_rotation = R.from_quat(mesh_orientation)
    bbox_world_centroid = lower_mesh_center + bbox_world_rotation.apply(base_link_offset)

    return bbox_size, base_link_offset, bbox_world_centroid, bbox_world_rotation


def process_link(G, link_node, base_link_center, canonical_orientation, obj_name, output_fs, tree_root, out_metadata):
    category_name, _, _, link_name = link_node
    raw_meta_links = G.nodes[link_node]["meta_links"]

    # Create a canonicalized copy of the lower and upper meshes.
    mesh_center = get_mesh_center(G.nodes[link_node]["lower_mesh"])
    canonical_mesh = transform_mesh(G.nodes[link_node]["lower_mesh"], mesh_center, canonical_orientation)
    meta_links = transform_meta_links(raw_meta_links, mesh_center, canonical_orientation)

    # Somehow we need to manually write the vertex normals to cache
    canonical_mesh._cache.cache["vertex_normals"] = canonical_mesh.vertex_normals

    in_edges = list(G.in_edges(link_node))
    assert len(in_edges) <= 1, f"Something's wrong: there's more than 1 in-edge to node {link_node}"

    # Compute the texture resolution needed
    mesh_surface_area = canonical_mesh.area
    log_area = np.clip(np.log(mesh_surface_area) / np.log(10), *LOG_SURFACE_AREA_RANGE)
    log_area_range_fraction = (log_area - LOG_SURFACE_AREA_RANGE[0]) / (LOG_SURFACE_AREA_RANGE[1] - LOG_SURFACE_AREA_RANGE[0])
    log2_area = LOG_TEXTURE_RANGE[0] + log_area_range_fraction * (LOG_TEXTURE_RANGE[1] - LOG_TEXTURE_RANGE[0])
    log2_area = int(np.clip(np.round(log2_area), *LOG_TEXTURE_RANGE))
    texture_res = int(2 ** log2_area)

    # Save the mesh
    with TempFS() as tfs:      
        obj_relative_path = f"{obj_name}-{link_name}.obj"
        save_mesh(canonical_mesh, tfs, obj_relative_path)

        # Check that a material got exported.
        material_files = [x for x in tfs.listdir("/") if x.endswith(".mtl")]
        assert len(material_files) == 1, f"Something's wrong: there's more than 1 material file in {tfs.listdir('/')}"
        original_material_filename = material_files[0]

        # Move the mesh to the correct path
        obj_link_mesh_folder_fs = output_fs.makedir("shape", recreate=True)
        obj_link_visual_mesh_folder_fs = obj_link_mesh_folder_fs.makedir("visual", recreate=True)
        obj_link_collision_mesh_folder_fs = obj_link_mesh_folder_fs.makedir("collision", recreate=True)
        obj_link_material_folder_fs = output_fs.makedir("material", recreate=True)

        # Fix texture file paths if necessary.
        original_material_fs = G.nodes[link_node]["material_dir"]
        if original_material_fs:
            for src_texture_file in original_material_fs.listdir("/"):
                fname = src_texture_file
                # fname is in the same format as room_light-0-0_VRayAOMap.png
                vray_name = fname[fname.index("VRay") : -4] if "VRay" in fname else None
                if vray_name in VRAY_MAPPING:
                    dst_fname = VRAY_MAPPING[vray_name]
                else:
                    raise ValueError(f"Unknown texture map: {fname}")

                dst_texture_file = f"{obj_name}-{link_name}-{dst_fname}.png"

                # Load the image
                # TODO: Re-enable this after tuning it.
                # texture = Image.open(original_material_fs.open(src_texture_file, "rb"), formats=("png",))
                # existing_texture_res = texture.size[0]
                # if existing_texture_res > texture_res:
                #     texture = texture.resize((texture_res, texture_res), Image.BILINEAR)
                # texture.save(obj_link_material_folder_fs.open(dst_texture_file, "wb"), format="png")
                fs.copy.copy_file(original_material_fs, src_texture_file, obj_link_material_folder_fs, dst_texture_file)

        # Copy the OBJ into the right spot
        fs.copy.copy_file(tfs, obj_relative_path, obj_link_visual_mesh_folder_fs, obj_relative_path)
        
        # Save and merge precomputed collision mesh
        canonical_collision_meshes = []
        collision_filenames_and_scales = []
        for i, collision_mesh in enumerate(G.nodes[link_node]["collision_mesh"]):
            canonical_collision_mesh = transform_mesh(collision_mesh, mesh_center, canonical_orientation)
            canonical_collision_mesh._cache.cache["vertex_normals"] = canonical_collision_mesh.vertex_normals
            collision_filename = obj_relative_path.replace(".obj", f"-{i}.obj")
            collision_scale = save_mesh_unit_bbox(canonical_collision_mesh, obj_link_collision_mesh_folder_fs, collision_filename)
            collision_filenames_and_scales.append((collision_filename, collision_scale))
            canonical_collision_meshes.append(canonical_collision_mesh)

        # Store the final meshes
        G.nodes[link_node]["visual_mesh"] = canonical_mesh.copy()
        G.nodes[link_node]["canonical_collision_mesh"] = trimesh.util.concatenate(canonical_collision_meshes)

        # Modify MTL reference in OBJ file
        mtl_name = f"{obj_name}-{link_name}.mtl"
        with obj_link_visual_mesh_folder_fs.open(obj_relative_path, "r") as f:
            new_lines = []
            for line in f.readlines():
                if f"mtllib {original_material_filename}" in line:
                    line = f"mtllib {mtl_name}\n"
                new_lines.append(line)

        with obj_link_visual_mesh_folder_fs.open(obj_relative_path, "w") as f:
            for line in new_lines:
                f.write(line)

        # Modify texture reference in MTL file
        with tfs.open(original_material_filename, "r") as f:
            new_lines = []
            for line in f.readlines():
                if "map_Kd material_0.png" in line:
                    line = ""
                    for key in MTL_MAPPING:
                        line += f"{key} ../../material/{obj_name}-{link_name}-{MTL_MAPPING[key]}.png\n"
                new_lines.append(line)

        with obj_link_visual_mesh_folder_fs.open(mtl_name, "w") as f:
            for line in new_lines:
                f.write(line)

    # Create the link in URDF
    link_xml = ET.SubElement(tree_root, "link")
    link_xml.attrib = {"name": link_name}
    visual_xml = ET.SubElement(link_xml, "visual")
    visual_origin_xml = ET.SubElement(visual_xml, "origin")
    visual_origin_xml.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
    visual_geometry_xml = ET.SubElement(visual_xml, "geometry")
    visual_mesh_xml = ET.SubElement(visual_geometry_xml, "mesh")
    visual_mesh_xml.attrib = {
        "filename": os.path.join("shape", "visual", obj_relative_path).replace("\\", "/"),
        "scale": " ".join([str(item) for item in np.ones(3)])
    }

    collision_origin_xmls = []
    for collision_filename, collision_scale in collision_filenames_and_scales:
        collision_xml = ET.SubElement(link_xml, "collision")
        collision_xml.attrib = {"name": collision_filename.replace(".obj", "")}
        collision_origin_xml = ET.SubElement(collision_xml, "origin")
        collision_origin_xml.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
        collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
        collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
        collision_mesh_xml.attrib = {
            "filename": os.path.join("shape", "collision", collision_filename).replace("\\", "/"),
            "scale": " ".join([str(item) for item in collision_scale])
        }
        collision_origin_xmls.append(collision_origin_xml)

    # This object might be a base link and thus without an in-edge. Nothing to do then.
    if len(in_edges) == 0:
        G.nodes[link_node]["link_frame_in_base"] = np.zeros(3)
        G.nodes[link_node]["mesh_in_link_frame"] = np.zeros(3)
    else:
        # Grab the lone edge to the parent.
        edge, = in_edges
        parent_node, child_node = edge
        assert child_node == link_node, f"Something's wrong: the child node of the edge is not the link node {link_node}"
        joint_type = G.edges[edge]["joint_type"]
        parent_frame = G.nodes[parent_node]["link_frame_in_base"]

        rotated_parent_frame = R.from_quat(canonical_orientation).apply(parent_frame)

        # Load the meshes.
        lower_canonical_points = transform_points(G.nodes[child_node]["lower_points"], base_link_center + rotated_parent_frame, canonical_orientation)

        # Get the center of mass of the child link in the parent frame.
        child_center = transform_points(np.array([mesh_center]), base_link_center + rotated_parent_frame, canonical_orientation)[0]

        # Create the joint in the URDF
        joint_xml = ET.SubElement(tree_root, "joint")
        joint_xml.attrib = {
            "name": f"j_{child_node[3]}",
            "type": {"P": "prismatic", "R": "revolute", "F": "fixed"}[joint_type]
        }

        joint_parent_xml = ET.SubElement(joint_xml, "parent")
        joint_parent_xml.attrib = {"link": parent_node[3]}
        joint_child_xml = ET.SubElement(joint_xml, "child")
        joint_child_xml.attrib = {"link": child_node[3]}

        mesh_offset = np.zeros(3)
        if joint_type in ("P", "R"):
            upper_canonical_points = transform_points(G.nodes[child_node]["upper_points"], base_link_center + rotated_parent_frame, canonical_orientation)
            
            if joint_type == "R":
                # Revolute joint
                num_v_lower = lower_canonical_points.shape[0]
                num_v_upper = upper_canonical_points.shape[0]
                assert num_v_lower == num_v_upper, f"{child_node} lower mesh has {num_v_lower} vertices while upper has {num_v_upper}. These should match."
                num_v = num_v_lower
                random_index = np.random.choice(num_v, min(num_v, 20), replace=False)
                from_vertices = lower_canonical_points[random_index]
                to_vertices = upper_canonical_points[random_index]

                # Find joint axis and joint limit
                r = R.align_vectors(
                    to_vertices - np.mean(to_vertices, axis=0),
                    from_vertices - np.mean(from_vertices, axis=0),
                )[0]
                upper_limit = r.magnitude()
                assert upper_limit < np.deg2rad(
                    180
                ), "upper limit of revolute joint should be <180 degrees"
                joint_axis = r.as_rotvec() / r.magnitude()

                # Let X = from_vertices_mean, Y = to_vertices_mean, R is rotation, T is translation
                # R * (X - T) + T = Y
                # => (I - R) T = Y - R * X
                # Find the translation part of the joint origin
                r_mat = r.as_matrix()
                from_vertices_mean = from_vertices.mean(axis=0)
                to_vertices_mean = to_vertices.mean(axis=0)
                left_mat = np.eye(3) - r_mat
                arbitrary_point_on_joint_axis = np.linalg.lstsq(
                    left_mat, (to_vertices_mean - np.dot(r_mat, from_vertices_mean)), rcond=None
                )[0]

                # The joint origin has infinite number of solutions along the joint axis
                # Find the translation part of the joint origin that is closest to the CoM of the link
                # by projecting the CoM onto the joint axis
                arbitrary_point_to_center = child_center - arbitrary_point_on_joint_axis
                joint_origin = arbitrary_point_on_joint_axis  + (
                     np.dot(arbitrary_point_to_center, joint_axis) * joint_axis)

                # Assign visual and collision mesh origin so that the offset from the joint origin is removed.
                mesh_offset = child_center - joint_origin
                visual_origin_xml.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}

                for collision_origin_xml in collision_origin_xmls:
                    collision_origin_xml.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}

                meta_links = normalize_meta_links(meta_links, mesh_offset)
            elif joint_type == "P":
                # Prismatic joint
                diff = np.mean(upper_canonical_points, axis=0) - np.mean(lower_canonical_points, axis=0)

                # Find joint axis and joint limit
                if not np.allclose(diff, 0):
                    upper_limit = np.linalg.norm(diff)
                    joint_axis = diff / upper_limit
                else:
                    upper_limit = 0
                    joint_axis = np.array([0, 0, 1])

                # Assign the joint origin relative to the parent CoM
                joint_origin = child_center

            # Save these joints' data
            joint_origin_xml = ET.SubElement(joint_xml, "origin")
            joint_origin_xml.attrib = {"xyz": " ".join([str(item) for item in joint_origin])}
            joint_axis_xml = ET.SubElement(joint_xml, "axis")
            joint_axis_xml.attrib = {"xyz": " ".join([str(item) for item in joint_axis])}
            joint_limit_xml = ET.SubElement(joint_xml, "limit")
            joint_limit_xml.attrib = {"lower": str(0.0), "upper": str(upper_limit)}
        else:
            # Fixed joints are quite simple.
            joint_origin = child_center

            if joint_type == "F":
                joint_origin_xml = ET.SubElement(joint_xml, "origin")
                joint_origin_xml.attrib = {"xyz": " ".join([str(item) for item in joint_origin])}
            else:
                raise ValueError("Unexpected joint type: " + str(joint_type))
        
        G.nodes[link_node]["link_frame_in_base"] = parent_frame + joint_origin
        G.nodes[link_node]["mesh_in_link_frame"] = mesh_offset

    out_metadata["meta_links"][link_name] = meta_links
    out_metadata["link_tags"][link_name] = G.nodes[link_node]["tags"]


def process_object(root_node, target, mesh_list, relevant_nodes, output_dir):
    try:
        obj_cat, obj_model, obj_inst_id, _ = root_node

        G = mesh_tree.build_mesh_tree(mesh_list, b1k_pipeline.utils.PipelineFS().target_output(target), filter_nodes=relevant_nodes)

        with OSFS(output_dir) as output_fs:
            obj_name = "-".join([obj_cat, obj_model])

            # Prepare the URDF tree
            tree_root = ET.Element("robot")
            tree_root.attrib = {"name": obj_model}

            # Extract base link orientation and position
            canonical_orientation = np.array(G.nodes[root_node]["canonical_orientation"])
            base_link_mesh = G.nodes[root_node]["lower_mesh"]
            base_link_center = get_mesh_center(base_link_mesh)

            out_metadata = {
                "meta_links": {},
                "link_tags": {},
                "object_parts": [],
            }

            # Iterate over each link.
            for link_node in nx.dfs_preorder_nodes(G, root_node):
                process_link(G, link_node, base_link_center, canonical_orientation, obj_name, output_fs, tree_root, out_metadata)

            # Save the URDF file.
            xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
            xmlio = io.StringIO(xmlstr)
            tree = ET.parse(xmlio)
            
            with output_fs.open(f"{obj_model}.urdf", "wb") as f:
                tree.write(f, xml_declaration=True)

            bbox_size, base_link_offset, _, _ = compute_object_bounding_box(G.nodes[root_node])

            # Compute part information
            for part_node_key in get_part_nodes(G, root_node):
                # Get the part node bounding box
                part_bb_size, _, part_bb_in_world_pos, part_bb_in_world_rot = compute_object_bounding_box(G.nodes[part_node_key])

                # Convert into our base link frame
                our_transform = np.eye(4)
                our_transform[:3, 3] = base_link_center
                our_transform[:3, :3] = R.from_quat(canonical_orientation).as_matrix()
                bb_transform = np.eye(4)
                bb_transform[:3, 3] = part_bb_in_world_pos
                bb_transform[:3, :3] = part_bb_in_world_rot.as_matrix()
                bb_transform_in_our = np.linalg.inv(our_transform) @ bb_transform
                bb_pos_in_our = bb_transform_in_our[:3, 3]
                bb_quat_in_our = R.from_matrix(bb_transform_in_our[:3, :3]).as_quat()

                # Get the part type
                part_tags = set(G.nodes[part_node_key]["tags"]) & ALLOWED_PART_TAGS
                assert(len(part_tags) == 1), f"Part node {part_node_key} has multiple part tags: {part_tags}"
                part_type, = part_tags

                # Add the metadata
                out_metadata["object_parts"].append({
                    "category": part_node_key[0],
                    "model": part_node_key[1],
                    "type": part_type,
                    "bb_pos": bb_pos_in_our,
                    "bb_orn": bb_quat_in_our,
                    "bb_size": part_bb_size,
                })

                # If it's a connectedpart, we also need to generate the corresponding female attachment point.
                if part_type == "connectedpart":
                    base_link_meta_links = out_metadata["meta_links"]["base_link"]
                    if "attachment" not in base_link_meta_links:
                        base_link_meta_links["attachment"] = {}
                    attachment_type = f"{part_node_key[1]}parent".lower() + "F"
                    if attachment_type not in base_link_meta_links["attachment"]:
                        base_link_meta_links["attachment"][attachment_type] = {}
                    next_id = len(base_link_meta_links["attachment"][attachment_type])

                    # pretend that the attachment point is at the center of the part with its transform
                    part_orn = np.array(G.nodes[part_node_key]["canonical_orientation"])
                    part_base_link_mesh = G.nodes[part_node_key]["lower_mesh"]
                    part_pos = get_mesh_center(part_base_link_mesh)
                    part_transform = np.eye(4)
                    part_transform[:3, 3] = part_pos
                    part_transform[:3, :3] = R.from_quat(part_orn).as_matrix()
                    part_transform_in_our = np.linalg.inv(our_transform) @ part_transform
                    part_pos_in_our = part_transform_in_our[:3, 3]
                    part_quat_in_our = R.from_matrix(part_transform_in_our[:3, :3]).as_quat()

                    # actually add the point
                    base_link_meta_links["attachment"][attachment_type][str(next_id)] = {
                        "position": part_pos_in_our,
                        "orientation": part_quat_in_our,                       
                    }

            # Similarly, if we are a connectedpart, we need to generate the corresponding male attachment point.
            if "connectedpart" in G.nodes[root_node]["tags"]:
                base_link_meta_links = out_metadata["meta_links"]["base_link"]
                if "attachment" not in base_link_meta_links:
                    base_link_meta_links["attachment"] = {}
                attachment_type = f"{obj_model}parent".lower() + "M"
                if attachment_type not in base_link_meta_links["attachment"]:
                    base_link_meta_links["attachment"][attachment_type] = {}
                next_id = len(base_link_meta_links["attachment"][attachment_type])
                
                # Pretend that the attachment point is at the center of the part with its transform
                next_id = len(base_link_meta_links["attachment"][attachment_type])
                base_link_meta_links["attachment"][attachment_type][str(next_id)] = {
                    "position": [0., 0., 0.],
                    "orientation": [0., 0., 0., 1.],                       
                }

            openable_joint_ids = [
                (i, joint.attrib["name"])
                for i, joint in enumerate(tree.findall("joint"))
                if "openable" in out_metadata["link_tags"][joint.find("child").attrib["link"]]
            ]

            # Save metadata json
            out_metadata.update({
                "base_link_offset": base_link_offset.tolist(),
                "bbox_size": bbox_size.tolist(),
                "orientations": [],
                "link_bounding_boxes": compute_link_aligned_bounding_boxes(G, root_node),
            })
            if openable_joint_ids:
                out_metadata["openable_joint_ids"] = openable_joint_ids
            with output_fs.makedir("misc").open("metadata.json", "w") as f:
                json.dump(out_metadata, f, cls=NumpyEncoder)
    except Exception as exc:
        return exc

def process_target(target, objects_path, executor):
    with b1k_pipeline.utils.PipelineFS() as pipeline_fs, OSFS(objects_path) as objects_fs:
        with pipeline_fs.target_output(target).open("object_list.json", "r") as f:
            mesh_list = json.load(f)["meshes"]

        # Build the mesh tree using our mesh tree library. The scene code also uses this system.
        G = mesh_tree.build_mesh_tree(mesh_list, pipeline_fs.target_output(target), load_meshes=False)

        # Go through each object.
        roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

        # Only save the 0th instance.
        saveable_roots = [root_node for root_node in roots if int(root_node[2]) == 0 and not G.nodes[root_node]["is_broken"]]
        object_futures = {}
        for root_node in saveable_roots:
            if root_node[0] not in ("car", "webcam"):
                continue

            # Start processing the object. We start by creating an object-specific
            # copy of the mesh tree (also including info about any parts)
            relevant_nodes = set(nx.dfs_tree(G, root_node).nodes())
            relevant_nodes |= {
                node
                for part_root_node in get_part_nodes(G, root_node)  # Get every part root node
                for node in nx.dfs_tree(G, part_root_node).nodes()}  # Get the subtree of each part

            obj_cat, obj_model, obj_inst_id, _ = root_node
            output_dirname = f"{obj_cat}/{obj_model}"
            object_futures[executor.submit(process_object, root_node, target, mesh_list, relevant_nodes, objects_fs.makedirs(output_dirname).getsyspath("/"))] = str(root_node)

        # Wait for all the futures - this acts as some kind of rate limiting on more futures being queued by blocking this thread
        # futures.wait(object_futures.keys())

        # Accumulate the errors
        error_msg = ""
        for future in futures.as_completed(object_futures.keys()):
            root_node = object_futures[future]
            exc = future.result()
            if exc:
                error_msg += f"{root_node}: {exc}\n\n"
        if error_msg:
            raise ValueError(error_msg)

def main():
    with b1k_pipeline.utils.ParallelZipFS("objects.zip", write=True) as archive_fs:
        objects_dir = archive_fs.makedir("objects").getsyspath("/")
        # Load the mesh list from the object list json.
        errors = {}
        target_futures = {}
     
        with futures.ThreadPoolExecutor(max_workers=50) as target_executor, futures.ProcessPoolExecutor(max_workers=16) as obj_executor:
            targets = ["scenes/Wainscott_0_garden", "objects/batch-02"]
            for target in tqdm.tqdm(targets):
                target_futures[target_executor.submit(process_target, target, objects_dir, obj_executor)] = target
            
            with tqdm.tqdm(total=len(target_futures)) as object_pbar:
                for future in futures.as_completed(target_futures.keys()):
                    try:
                        result = future.result()
                    except:
                        name = target_futures[future]
                        errors[name] = traceback.format_exc()

                    object_pbar.update(1)

                    remaining_targets = [v for k, v in target_futures.items() if not k.done()]
                    if len(remaining_targets) < 10:
                        print("Remaining:", remaining_targets)

            print("Time for executor shutdown")

        print("Finished processing")

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    with pipeline_fs.pipeline_output().open("export_objs.json", "w") as f:
        json.dump({"success": not errors, "errors": errors}, f)

if __name__ == "__main__":
    main()
