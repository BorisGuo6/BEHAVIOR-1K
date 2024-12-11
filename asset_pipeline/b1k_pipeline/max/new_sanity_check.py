# Install bddl by doing
# pip install git+https://github.com/StanfordVL/bddl.git@develop

import csv
import functools
import sys
import time

sys.path.append(r"D:\ig_pipeline")

import collections
import json
import pathlib
import traceback

import numpy as np
import pandas as pd
import pymxs
import trimesh
import trimesh.transformations
from scipy.spatial.transform import Rotation

import b1k_pipeline.utils
from b1k_pipeline.max.prebake_textures import (
    get_recorded_uv_unwrapping_hash,
    hash_object,
)

from bddl.object_taxonomy import ObjectTaxonomy

OBJECT_TAXONOMY = ObjectTaxonomy()

rt = pymxs.runtime

MAX_VERTICES = 100000
WARN_VERTICES = 20000

OUTPUT_FILENAME = "sanitycheck.json"


ALLOWED_TAGS = {
    "soft",
    "glass",
    "openable",
    "openablebothsides",
}

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}


ALLOWED_META_TYPES = {
    "fluidsource": "dimensionless",
    "togglebutton": "primitive",
    "attachment": "dimensionless",
    "heatsource": "dimensionless",
    "com": "dimensionless",
    "particleapplier": "primitive",
    "particleremover": "primitive",
    "fluidsink": "primitive",
    "slicer": "primitive",
    "fillable": "convexmesh",
    "openfillable": "convexmesh",
    "collision": "convexmesh",
}
assert not (
    set(ALLOWED_META_TYPES.values()) - {"dimensionless", "primitive", "convexmesh"}
), "Found invalid meta type mapping"


# Objects that don't require these meta types should not have them
REQUIRED_ONLY_META_TYPES = {
    "fluidsource",
    "togglebutton",
    "heatsource",
    "particleapplier",
    "particleremover",
    "fluidsink",
    "slicer",
    "fillable",
}


RENAMES = {}
with open(b1k_pipeline.utils.PIPELINE_ROOT / "metadata/object_renames.csv") as f:
    for row in csv.DictReader(f):
        key = (row["Original category (auto)"], row["ID (auto)"])
        RENAMES[key] = row["New Category"]


DELETION_QUEUE = set()
with open(b1k_pipeline.utils.PIPELINE_ROOT / "metadata/deletion_queue.csv", "r") as f:
    for row in csv.DictReader(f):
        DELETION_QUEUE.add(row["Object"].strip().split("-")[1])


def get_required_meta_links(category):
    synset = OBJECT_TAXONOMY.get_synset_from_category(category)
    if synset is None:
        raise ValueError(f"Category {category} not found in taxonomy.")
    return OBJECT_TAXONOMY.get_required_meta_links_for_synset(synset)


def is_light_required(category):
    # TODO: Make this more robust.
    return "light" in category or "lamp" in category


def compute_shear(obj):
    # Check that object satisfies no-shear rule.
    object_transform = np.hstack(
        [b1k_pipeline.utils.mat2arr(obj.objecttransform), [[0], [0], [0], [1]]]
    ).T
    transform = np.hstack(
        [b1k_pipeline.utils.mat2arr(obj.transform), [[0], [0], [0], [1]]]
    ).T
    obj_to_pivot = np.linalg.inv(transform) @ object_transform
    shear = trimesh.transformations.decompose_matrix(np.linalg.inv(obj_to_pivot))[1]
    return shear


def get_approved_categories():
    # gc = gspread.service_account(filename=KEY_FILE)
    # sh = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)

    exists = []
    approved = []

    return exists, approved


def get_providers():
    inventory_path = (
        b1k_pipeline.utils.PIPELINE_ROOT
        / "artifacts"
        / "pipeline"
        / "object_inventory.json"
    )
    with open(inventory_path, "r") as f:
        return {k.split("-")[-1]: v for k, v in json.load(f)["providers"].items()}


def quat2arr(q):
    return np.array([q.x, q.y, q.z, q.w])


class ObjectWrapper(object):
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        if attr in ["__array__", "__array_struct__", "__array_interface__", "_typ"]:
            raise AttributeError()
        return getattr(self._obj, attr)


def classOf(obj_wrapper):
    return rt.classOf(obj_wrapper._obj)


class SanityCheck:
    def __init__(self):
        self.reset()
        self._existing_categories, self._approved_categories = get_approved_categories()
        self._providers = get_providers()

    def expect(self, condition, message, level="ERROR"):
        if not condition:
            self.errors[level].append(message)

    def reset(self):
        self.errors = collections.defaultdict(list)

    @functools.lru_cache(maxsize=None)
    def get_verts_for_obj(self, obj):
        return np.array(
            rt.polyop.getVerts(
                obj.baseObject, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj))
            )
        )

    @functools.lru_cache(maxsize=None)
    def get_faces_for_obj(self, obj):
        return (
            np.array(
                rt.polyop.getFacesVerts(
                    obj.baseObject, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
                )
            )
            - 1
        )

    def maybe_rename_category(self, cat, model):
        if (cat, model) in RENAMES:
            return RENAMES[(cat, model)]
        elif any(m == model and RENAMES[(c, m)] != cat for c, m in RENAMES):
            self.expect(
                False,
                f"Model {model} has category {cat} that is neither the from or to element in the rename file.",
            )
        return cat

    def is_valid_object_type(self, row):
        is_valid_type = row.type in [
            rt.Editable_Poly,
            rt.VRayLight,
            rt.VRayPhysicalCamera,
            rt.Point,
            rt.Box,
            rt.Cylinder,
            rt.Sphere,
            rt.Cone,
            rt.Physical_Camera,
            rt.FreeCamera,
            rt.Plane,
        ]

        self.expect(is_valid_type, f"{row.object_name} has disallowed type {row.type}.")
        if row.object.parent is None:
            is_valid_root = row.type in [
                rt.Editable_Poly,
                rt.VRayLight,
                rt.VRayPhysicalCamera,
                rt.Physical_Camera,
                rt.FreeCamera,
                rt.Plane,
            ]
            self.expect(
                is_valid_root,
                f"Root-level object {row.object_name} has disallowed type {row.type}.",
            )
        return is_valid_type

    def is_valid_name(self, row):
        is_valid_name = (
            not pd.isna(row.object_name)
            and b1k_pipeline.utils.parse_name(row.object_name) is not None
        )
        self.expect(is_valid_name, f"{row.object_name} has bad name.")
        return is_valid_name

    def get_recorded_vertex_and_face_count(self, model_id):
        # Look the provider up from the inventory file
        provider = self._providers.get(model_id, None)
        self.expect(
            provider is not None,
            f"{model_id} has no provider in the inventory file.",
        )

        if provider is None:
            return

        # Get the vertex and edge count from the provider's object list file
        object_list = (
            b1k_pipeline.utils.PIPELINE_ROOT
            / "cad"
            / provider
            / "artifacts"
            / "object_list.json"
        )
        with open(object_list, "r") as f:
            object_list_data = json.load(f)
        vertex_and_face_counts = object_list_data["mesh_fingerprints"]
        self.expect(
            model_id in vertex_and_face_counts,
            f"{model_id} has no vertex/face count in the object list file for provider {provider}.",
        )

        if model_id not in vertex_and_face_counts:
            return

        recorded_vertex_count, recorded_face_count = vertex_and_face_counts[model_id][
            :2
        ]
        return recorded_vertex_count, recorded_face_count

    def validate_object(self, row):
        # Check that the category exists on the spreadsheet
        # self.expect(
        #   row.name_category in self._existing_categories,
        #   f"Category {row.name_category} for object {row.object_name} does not exist on spreadsheet.",
        #   level="WARNING")

        # Check that the category is approved on the spreadsheet
        # self.expect(
        #   row.name_category in self._existing_categories,
        #   f"Category {row.name_category} for object {row.object_name} is not approved on spreadsheet.",
        #   level="WARNING")

        # Check that it does not have too many vertices.
        # self.expect(
        #   row.vertex_count <= MAX_VERTICES,
        #   f"{row.object_name} has too many vertices: {row.vertex_count} > {MAX_VERTICES}")

        # Unwrap OBJ for below cases.
        obj = row.object._obj

        # Warn if it has too many but acceptable vertices.
        if not row["name_bad"]:
            self.expect(
                row.vertex_count <= WARN_VERTICES,
                f"{row.object_name} has too many vertices: {row.vertex_count} > {WARN_VERTICES}",
                level="WARNING",
            )

            self.expect(
                rt.classOf(obj.material) == rt.Shell_Material,
                f"{row.object_name} has non-shell material. Run texture baking.",
            )

            self.expect(
                rt.classOf(obj.material) == rt.Shell_Material
                and obj.material.renderMtlIndex == 1,
                f"{row.object_name} is not rendering the baked material. Select the baked material for rendering or rebake.",
            )

            self.expect(
                rt.classOf(obj.material) == rt.Shell_Material
                and obj.material.viewportMtlIndex == 1,
                f"{row.object_name} is not rendering the baked material in the viewport. Select the baked material for viewport.",
            )

            current_hash = hash_object(
                obj,
                verts=self.get_verts_for_obj(obj),
                faces=self.get_faces_for_obj(obj),
            )
            recorded_hash = get_recorded_uv_unwrapping_hash(obj)
            self.expect(
                recorded_hash == current_hash,
                f"{row.object_name} has different UV unwrapping than recorded. Reunwrap the object.",
            )

            # Run cloth object checks
            synset = OBJECT_TAXONOMY.get_synset_from_category(row.name_category)
            if synset is not None:
                obj_is_cloth = "cloth" in OBJECT_TAXONOMY.get_abilities(synset)
                if obj_is_cloth:
                    self.validate_cloth(row)
            else:
                self.expect(
                    False,
                    f"Cannot validate clothness: category {row.name_category} not found in taxonomy.",
                )

            # Check that each object zeroth instance object actually has a collision mesh
            if int(row.name_instance_id) == 0 and row.name_joint_side != "upper":
                for child in obj.children:
                    if "Mcollision" in child.name:
                        break
                else:
                    self.expect(
                        False,
                        f"{row.object_name} has no collision mesh. Create a collision mesh.",
                    )

        else:
            # Bad object tasks
            # Validate bad objects to make sure that they do NOT have upper meshes or meta links
            # or lights.
            self.expect(
                row.name_joint_side != "upper",
                f"Bad object {row.object_name} should not have upper side.",
            )

            self.expect(
                len(row.object.children) == 0,
                f"Bad object {row.object_name} should not have children.",
            )

        # Bad - nonbad common tasks.

        # Check that the object is renamed properly
        category = row.name_category
        model_id = row.name_model_id
        renamed_category = self.maybe_rename_category(category, model_id)
        self.expect(
            category == renamed_category,
            f"{row.object_name} has unapplied rename {category}.",
            level="WARNING",
        )
        if model_id in DELETION_QUEUE:
            self.expect(
                False,
                f"{row.object_name} is in the deletion queue. Delete the object.",
            )

        # Check that the object does not have any modifiers
        self.expect(
            len(row.object.modifiers) == 0, f"{row.object_name} has modifiers attached."
        )

        # Check that there are no dead elements
        self.expect(
            rt.polyop.GetHasDeadStructs(obj) == 0,
            f"{row.object_name} has dead structs. Apply the Triangulate script.",
        )

        self.expect(
            all(
                rt.polyop.getFaceDeg(obj, i + 1) == 3
                for i in range(rt.polyop.GetNumFaces(obj))
            ),
            f"{row.object_name} has non-triangular faces. Apply the Triangulate script.",
        )

        # Check that the object does not have self-intersecting faces
        faces = self.get_faces_for_obj(obj)
        # A face is self-intersecting if a vertex shows up more than once in the face.
        self_intersecting = np.any(np.unique(faces, return_counts=True, axis=-1)[1] > 1)
        self.expect(
            not self_intersecting,
            f"{row.object_name} has self-intersecting faces. Apply the Triangulate script.",
        )

        # Check that object satisfies the scale condition.
        scale = np.array(row.object.scale)
        self.expect(
            np.allclose(scale, 1),
            f"{row.object_name} has scale that is not 1. Reset scale.",
        )

        # Check that object does not have negative object offset scale.
        object_offset_scale = np.array(row.object.objectoffsetscale)
        self.expect(
            np.all(object_offset_scale > 0),
            f"{row.object_name} has negative object offset scale. Try mirroring + resetting scale.",
        )

        # Check that object does not have a light ID.
        self.expect(
            pd.isna(row.name_light_id),
            f"Non-light object {row.object_name} should not have light ID.",
        )

        # Check that object does not have any meta type.
        self.expect(
            pd.isna(row.name_meta_info),
            f"Non-meta object {row.object_name} should not have meta info.",
        )

        # Check object name format.
        self.expect(
            len(row.name_model_id) == 6,
            f"{row.object_name} does not have 6-digit model ID.",
        )

        # Check the tags
        tags = set()
        tags_str = row.name_tag
        if tags_str:
            tags = set([x[1:] for x in tags_str.split("-") if x])
        invalid_tags = tags - (ALLOWED_TAGS | ALLOWED_PART_TAGS)
        self.expect(
            not invalid_tags, f"{row.object_name} has invalid tags {invalid_tags}."
        )

        part_tags = tags & ALLOWED_PART_TAGS
        if row.object.parent is None:
            self.expect(
                not part_tags,
                f"{row.object_name} is not under another object but contains part tags {part_tags}.",
            )
        else:
            self.expect(
                part_tags,
                f"Part object {row.object_name} should be marked with a part tag, one of: {ALLOWED_PART_TAGS}.",
            )

        # Validate meta stuff
        if int(row.name_instance_id) == 0:
            self.validate_meta_links(row)

    def validate_cloth(self, row):
        # A cloth object should consist of a single connected component
        obj = row.object._obj
        verts = self.get_verts_for_obj(obj)
        faces = self.get_faces_for_obj(obj)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Split the faces into elements
        self.expect(
            tm.body_count == 1,
            f"Cloth object {obj.name} should consist of exactly 1 element. Currently it has {tm.body_count} elements.",
        )

    def validate_light(self, row):
        # Validate that the object has a light ID
        self.expect(
            not pd.isna(row.name_light_id),
            f"Light object {row.object_name} should have light ID.",
        )

        # Check that the type is one of plane, sphere, disk
        self.expect(
            row.object.type in (0, 2, 4),
            f"Light object {row.object_name} type should be one of SPHERE, DISC, PLANE",
        )

        # Light should not be scaled or offset
        self.expect(
            np.allclose(row.object.scale, 1),
            f"Light object {row.object_name} should not have scale.",
        )
        self.expect(
            np.allclose(row.object.objectoffsetscale, 1),
            f"Light object {row.object_name} should not have scale.",
        )
        self.expect(
            np.allclose(row.object.objectoffsetpos, 0),
            f"Light object {row.object_name} should not have object offset position. Reset pivot.",
        )
        self.expect(
            np.allclose(quat2arr(row.object.objectoffsetrot), [0, 0, 0, 1]),
            f"Light object {row.object_name} should not have object offset rotation. Reset pivot.",
        )

    def validate_group_of_max_instances(self, rows):
        # Pick an object as the base instance
        rows_with_id_zero = rows[rows["name_instance_id"] == "0"]
        obj_name = rows["object_name"].iloc[0]
        self.expect(
            len(rows_with_id_zero.index) > 0,
            f"No instance ID 0 instance of {obj_name}.",
        )
        base = (
            rows_with_id_zero.iloc[0]
            if len(rows_with_id_zero.index) > 0
            else rows.iloc[0]
        )

        # Check that they have the same model ID and same category
        unique_model_ids = rows.groupby(
            ["name_category", "name_model_id"], sort=False, dropna=False
        ).ngroups
        self.expect(
            unique_model_ids == 1,
            f"All instances of {base.object_name} do not share category/model ID.",
        )

        # Check that the keys start from 0 and are contiguous.
        unique_instance_ids = rows.groupby(
            ["name_category", "name_model_id", "name_instance_id"],
            sort=False,
            dropna=False,
        ).ngroups
        existing_instance_ids = set(rows["name_instance_id"])
        expected_instance_ids = set(str(x) for x in range(unique_instance_ids))
        self.expect(
            existing_instance_ids == expected_instance_ids,
            f"All instances of {base.object_name} do not have contiguous instance IDs. Missing: {sorted(expected_instance_ids - existing_instance_ids)}",
        )

        # Check that they all have the same object offset rotation and pos/scale and shear.
        desired_offset_pos = np.array(base.object.objectOffsetPos) / np.array(
            base.object.objectOffsetScale
        )
        desired_offset_rot_inv = Rotation.from_quat(
            quat2arr(base.object.objectOffsetRot)
        ).inv()
        desired_shear = compute_shear(base.object)
        for _, row in rows.iterrows():
            this_offset_pos = np.array(row.object.objectOffsetPos) / np.array(
                row.object.objectOffsetScale
            )
            pos_diff = this_offset_pos - desired_offset_pos
            self.expect(
                np.allclose(pos_diff, 0, atol=1e-1),
                f"{row.object_name} has different pivot offset position (by {pos_diff}). Match pivots on each instance.",
            )

            this_offset_rot = Rotation.from_quat(quat2arr(row.object.objectOffsetRot))
            rot_diff = (this_offset_rot * desired_offset_rot_inv).magnitude()
            self.expect(
                np.allclose(rot_diff, 0, atol=1e-3),
                f"{row.object_name} has different pivot offset rotation (by {rot_diff}). Match pivots on each instance.",
            )

            this_shear = compute_shear(row.object)
            self.expect(
                np.allclose(this_shear, desired_shear, atol=1e-3),
                f"{row.object_name} has different shear. Match scaling axes on each instance.",
            )

            self.expect(
                row.object.material == base.object.material,
                f"{row.object_name} has different material. Match materials on each instance.",
            )

    def validate_collision(self, obj):
        try:
            # For this case, unwrap the object
            obj = obj._obj

            # Expect that collision meshes do not share instances in the scene
            self.expect(
                not [
                    x for x in rt.objects if x.baseObject == obj.baseObject and x != obj
                ],
                f"{obj.name} should not have instances.",
            )

            # Check that there are no dead elements
            self.expect(
                rt.polyop.GetHasDeadStructs(obj) == 0,
                f"{obj.name} has dead structs. Apply the Triangulate script.",
            )

            # Get vertices and faces into numpy arrays for conversion
            verts = self.get_verts_for_obj(obj)
            faces = self.get_faces_for_obj(obj)
            self.expect(len(faces) > 0, f"{obj.name} has no faces.")
            self.expect(
                all(len(f) == 3 for f in faces),
                f"{obj.name} has non-triangular faces. Apply the Triangulate script.",
            )

            all_cmeshes = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

            # Split the faces into elements
            elems = all_cmeshes.split(only_watertight=False, repair=False)
            self.expect(
                len(elems) <= 32,
                f"{obj.name} should not have more than 32 elements. Has {len(elems)} elements.",
            )

            # Iterate through the elements
            for i, m in enumerate(elems):
                self.expect(
                    len(m.vertices) <= 60,
                    f"{obj.name} element {i} has too many vertices ({len(m.vertices)} > 60)",
                )
                self.expect(m.is_volume, f"{obj.name} element {i} is not a volume")
                # self.expect(m.is_convex, f"{obj.name} element {i} is not convex")
        except Exception as e:
            self.expect(False, str(e))

    def validate_meta_links(self, row):
        # Don't worry about bad objects or upper links.
        if row.name_joint_side == "upper":
            return

        # Get the children using the parent rows
        children = [ObjectWrapper(x) for x in row.object.children]

        # Expect that all of the children are valid meta-links / lights
        found_ml_types = set()
        found_ml_subids_for_id = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        found_convex_mesh_metas = collections.defaultdict(list)
        for child in children:
            # Otherwise, we can validate the individual meta link
            match = b1k_pipeline.utils.parse_name(child.name)
            if match is None:
                self.expect(
                    False,
                    f"{child.name} is an invalid meta link under {row.object_name}.",
                )
                continue

            # If it's an EditablePoly with no meta tag, then it will be processed as a root object too. Safe to skip.
            if classOf(child) == rt.Editable_Poly and not match.group("meta_type"):
                continue

            self.expect(
                match.group("mesh_basename") == row.name_mesh_basename,
                f"Meta link {child.name} base name doesnt match parent {row.object_name}",
            )

            # Keep track of the meta links we have
            self.expect(
                match.group("meta_type"),
                f"Meta object {child.name} should have a meta link type in its name.",
            )
            meta_link_type = match.group("meta_type")
            if meta_link_type not in ALLOWED_META_TYPES:
                self.expect(
                    False,
                    f"Meta link type {meta_link_type} not in list of allowed meta link types: {ALLOWED_META_TYPES.keys()}",
                )
                continue

            found_ml_types.add(meta_link_type)

            # Keep track of subids for this meta ID
            meta_id = "0"
            if match.group("meta_id"):
                meta_id = match.group("meta_id")
            meta_subid = "0"
            if match.group("meta_subid"):
                meta_subid = match.group("meta_subid")
            found_ml_subids_for_id[meta_link_type][meta_id].append(meta_subid)

            if ALLOWED_META_TYPES[meta_link_type] == "dimensionless":
                self.expect(
                    classOf(child) == rt.Point,
                    f"Dimensionless {meta_link_type} meta link {child.name} should be of Point instead of {classOf(child)}",
                )

                # Check that the position and orientation are essentially identical between the transform and the objecttransform
                transform_pos = child.transform.position
                transform_rot = Rotation.from_quat(quat2arr(child.transform.rotation))
                object_transform_pos = child.objecttransform.position
                object_transform_rot = Rotation.from_quat(
                    quat2arr(child.objecttransform.rotation)
                )
                self.expect(
                    np.allclose(transform_pos, object_transform_pos, atol=1e-3),
                    f"Dimensionless {meta_link_type} meta link {child.name} has nonzero object offset position.",
                )
                delta_rot = (transform_rot * object_transform_rot.inv()).magnitude()
                self.expect(
                    np.isclose(delta_rot, 0, atol=1e-3),
                    f"Dimensionless {meta_link_type} meta link {child.name} has nonzero object offset rotation.",
                )
            elif ALLOWED_META_TYPES[meta_link_type] == "primitive":
                volumetric_allowed_types = {
                    rt.Box,
                    rt.Cylinder,
                    rt.Sphere,
                    rt.Cone,
                }
                self.expect(
                    classOf(child) in volumetric_allowed_types,
                    f"Volumetric {meta_link_type} meta link {child.name} should be one of {volumetric_allowed_types} instead of {classOf(child)}",
                )

                scale = np.array(list(child.objecttransform.scale))
                if classOf(child) == rt.Sphere:
                    size = np.array([child.radius, child.radius, child.radius]) * scale
                elif classOf(child) == rt.Box:
                    size = np.array([child.width, child.length, child.height]) * scale
                elif classOf(child) == rt.Cylinder:
                    size = np.array([child.radius, child.radius, child.height]) * scale
                elif classOf(child) == rt.Cone:
                    # Cones should have radius1 as 0 and radius2 nonzero
                    self.expect(
                        np.isclose(child.radius1, 0),
                        f"Cone {child.name} radius1 should be zero.",
                    )
                    self.expect(
                        not np.isclose(child.radius2, 0) and child.radius2 > 0,
                        f"Cone {child.name} radius2 should be nonzero.",
                    )
                    size = (
                        np.array([child.radius2, child.radius2, child.height]) * scale
                    )
                # self.expect(np.all(size > 0), f"Volumetric {meta_link_type} meta link {child.name} should have positive size/scale combo.")

            elif ALLOWED_META_TYPES[meta_link_type] == "convexmesh":
                # TODO: Expect that each element is a convex mesh
                self.expect(
                    classOf(child) == rt.Editable_Poly,
                    f"Convex mesh {meta_link_type} meta link {child.name} should be of Editable Poly instead of {classOf(child)}",
                )
            else:
                raise ValueError(
                    "Don't know how to process meta type "
                    + ALLOWED_META_TYPES[meta_link_type]
                )

            if ALLOWED_META_TYPES.get(meta_link_type, None) == "convexmesh":
                found_convex_mesh_metas[meta_link_type].append(child)

            if meta_link_type == "collision":
                self.validate_collision(child)
            elif meta_link_type == "attachment":
                attachment_type = match.group("meta_id")
                self.expect(
                    len(attachment_type) > 0,
                    f"Missing attachment type on object {row.object_name}",
                )
                self.expect(
                    attachment_type[-1] in "MF",
                    f"Invalid attachment gender {attachment_type} on object {row.object_name}",
                )
                attachment_type = attachment_type[:-1]
                self.expect(
                    len(attachment_type) > 0,
                    f"Missing attachment type on object {row.object_name}",
                )

        # Validate that each object has no more than one of each kind of convexmesh meta.
        for mesh_type, items in found_convex_mesh_metas.items():
            self.expect(
                len(items) <= 1,
                f"Object {row.object_name} has {len(items)} {mesh_type} meshes. Should have no more than one.",
            )

        # Check that meta subids are correct:
        for meta_type, meta_ids_to_subids in found_ml_subids_for_id.items():
            for meta_id, meta_subids in meta_ids_to_subids.items():
                expected_subids = {str(x) for x in range(len(meta_subids))}
                self.expect(
                    set(meta_subids) == expected_subids,
                    f"{row.name_mesh_basename} meta type {meta_type} ID {meta_id} has non-continuous subids {sorted(meta_subids)}",
                )

    def validate_model_instance_link_group(self, group):
        # Check that each link that has is not the base link has a parent and a nonempty joint type
        model_id = group["name_model_id"].iloc[0]
        instance_id = group["name_instance_id"].iloc[0]
        link_name = group["name_link_name"].iloc[0]
        descriptor = f"{model_id}-{instance_id}-{link_name}"
        if link_name != "base_link":
            unique_values_for_joint_type = group["name_joint_type"].unique()
            self.expect(
                len(unique_values_for_joint_type) == 1
                and pd.notnull(unique_values_for_joint_type[0]),
                f"{descriptor} should have exactly one nonempty joint type: {unique_values_for_joint_type}.",
            )
            self.expect(
                group["name_parent_link_name"].nunique() == 1
                and pd.notnull(group["name_parent_link_name"].iloc[0]),
                f"{descriptor} should have exactly one nonempty parent link name: {group['name_parent_link_name'].unique()}.",
            )

        should_have_upper = (
            link_name != "base_link"
            and instance_id == "0"
            and not group["name_bad"].iloc[0]
            and group["name_joint_type"].iloc[0] != "F"
        )
        if should_have_upper:
            self.expect(
                "upper" in group["name_joint_side"].unique(),
                f"{descriptor} should have an upper side.",
            )
        else:
            self.expect(
                "upper" not in group["name_joint_side"].unique(),
                f"{descriptor} should not have an upper side.",
            )

    def validate_model_instance_group(self, group):
        # Check that the model instance group has a base link.
        self.expect(
            "base_link" in group["name_link_name"].unique(),
            f"Model ID {group['name_model_id'].iloc[0]} is missing 'base_link'.",
        )

        # Get the model and instance ID
        self.expect(
            group["name_category"].nunique() == 1,
            f"Model ID {group['name_model_id'].iloc[0]} has inconsistent categories.",
        )
        category = group["name_category"].iloc[0]
        assert group["name_model_id"].nunique() == 1
        model_id = group["name_model_id"].iloc[0]
        assert group["name_instance_id"].nunique() == 1
        instance_id = group["name_instance_id"].iloc[0]

        # First assert that the group has exactly one value for name_bad
        self.expect(
            group["name_bad"].nunique() <= 1,
            f"Model ID {group['name_model_id'].iloc[0]} has inconsistent bad values: {group['name_bad'].unique()}.",
        )

        # Validate the link groups
        group.groupby("name_link_name").apply(self.validate_model_instance_link_group)

        # If the instance group belongs to a bad object, validate the bad object matches the
        # total vertex and face count from the other file.
        if group["name_bad"].iloc[0]:
            # Compute the total vertex and face count for the model instance group
            real_vertex_count = group["vertex_count"].sum()
            real_face_count = group["face_count"].sum()
            # Get the recorded vertex and face count for the model ID
            recorded_vertex_count, recorded_face_count = (
                self.get_recorded_vertex_and_face_count(model_id)
            )
            # Check that the total vertex and face count matches the recorded vertex and face count
            self.expect(
                real_vertex_count == recorded_vertex_count,
                f"{model_id}-{instance_id} has different vertex count than recorded in provider: {real_vertex_count} != {recorded_vertex_count}.",
            )
            self.expect(
                real_face_count == recorded_face_count,
                f"{model_id}-{instance_id} has different face count than recorded in provider: {real_face_count} != {recorded_face_count}.",
            )

        # If this is the zeroth instance, check the object's meta links set
        if instance_id == "0" and not group["name_bad"].iloc[0]:
            try:
                # First validate that if the object requires joints, it has them
                renamed_category = self.maybe_rename_category(category, model_id)
                required_meta_types = get_required_meta_links(renamed_category)

                requires_joints = "joint" in required_meta_types
                if requires_joints:
                    # it should contain at least one revolute or prismatic joint
                    # TODO: Perhaps also assert the presence of an openable tag later.
                    self.expect(
                        set(group["name_joint_type"].unique()) & {"R", "P"},
                        f"Model ID {model_id} requires joints but has no joints.",
                    )

                # Then look at meta link types
                def get_meta_type(o):
                    pn = b1k_pipeline.utils.parse_name(o.name)
                    return pn.group("meta_type") if pn else None

                found_ml_types = {
                    get_meta_type(child)
                    for group_obj in group["object"]
                    for child in group_obj.children
                }
                found_ml_types -= {None}

                required_meta_types -= {
                    "subpart",
                    "joint",
                }
                existing_meta_types = set(found_ml_types)
                if "openfillable" in existing_meta_types:
                    existing_meta_types.add("fillable")
                missing_meta_types = required_meta_types - existing_meta_types
                if missing_meta_types:
                    for missing_meta_type in missing_meta_types:
                        self.expect(
                            False,
                            f"{model_id} is missing meta link: {missing_meta_type}.",
                        )

                # Also make sure that things that don't require a meta link don't have one,
                # for certain specific cases.
                found_nonoptional_meta_types = (
                    existing_meta_types & REQUIRED_ONLY_META_TYPES
                )
                found_extra_nonoptional_meta_types = (
                    found_nonoptional_meta_types - required_meta_types
                )
                if found_extra_nonoptional_meta_types:
                    for extra_meta_type in found_extra_nonoptional_meta_types:
                        self.expect(
                            False,
                            f"{model_id} has meta link not required by its synset: {extra_meta_type}.",
                        )
            except ValueError as e:
                self.expect(
                    False,
                    f"Cannot validate meta links and joints for model ID {model_id}: {e}",
                )

    def validate_model_group(self, group):
        category = group["name_category"].iloc[0]
        model_id = group["name_model_id"].iloc[0]

        # Check that the group's model ID does not contain the phrase "todo"
        self.expect(
            "todo" not in model_id,
            f"Model ID {model_id} contains 'todo'.",
        )

        # Check that every instance of this model group has the same set of links.
        self.expect(
            group.groupby("name_instance_id")["name_link_name"]
            .apply(frozenset)
            .nunique()
            == 1,
            f"Inconsistent link sets within model ID {model_id}.",
        )

        # Then individually validate each of the model instances
        group.groupby("name_instance_id").apply(self.validate_model_instance_group)

        # For each instance, record the scale of the base link, and the positional offset of the
        # child links in the base link's frame. Assert that after scaling these numbers are close.
        links_relative_transforms = {}

        def record_links(group):
            instance_id = group["name_instance_id"].iloc[0]
            base_link_row = group[group["name_link_name"] == "base_link"].iloc[0]
            base_link_transform = base_link_row.object.objecttransform
            inverse_base_link_transform = rt.inverse(base_link_transform)

            links_relative_transforms[instance_id] = {}
            for _, row in group.iterrows():
                link_name = row["name_link_name"]

                # Skip the base link
                if link_name == "base_link":
                    continue

                # Skip upper sides
                if row["name_joint_side"] == "upper":
                    continue

                # Compute the relative transform
                child_link_transform = row.object.objecttransform
                relative_transform = child_link_transform * inverse_base_link_transform
                links_relative_transforms[instance_id][link_name] = relative_transform

        group.groupby("name_instance_id").apply(record_links)

        # Go through all the available links
        for link_name in group["name_link_name"].unique():
            # Skip base link
            if link_name == "base_link":
                continue

            # Get the relative transform of the first object
            if link_name not in links_relative_transforms["0"]:
                self.expect(
                    False,
                    f"{model_id} link {link_name} is missing in instance 0, so relative transform check cannot be completed. Found zeroth instance links: {links_relative_transforms['0']}.",
                )
                continue
            instance_zero_relative_transform = links_relative_transforms["0"].get(
                link_name
            )
            inverse_instance_zero_relative_transform = rt.inverse(
                instance_zero_relative_transform
            )

            # For each instance, check that the relative transform is the same
            for instance_id, link_transforms in links_relative_transforms.items():
                if instance_id == "0":
                    continue

                if link_name not in link_transforms:
                    self.expect(
                        False,
                        f"{model_id} link {link_name} is missing in instance {instance_id}, so relative transform check cannot be completed.",
                    )
                    continue

                # Check that the relative transform is the same
                relative_transform = link_transforms[link_name]
                transform_difference = (
                    relative_transform * inverse_instance_zero_relative_transform
                )

                # Decompose the transform difference into position, rotation and scale
                scale_difference = np.array(transform_difference.scale)
                position_difference = np.array(transform_difference.position)
                rotation_difference = Rotation.from_quat(
                    quat2arr(transform_difference.rotation)
                )

                self.expect(
                    np.allclose(scale_difference, 1, atol=0.02),
                    f"{model_id} link {link_name} has different scale in instance {instance_id} compared to instance 0. Scale difference: {scale_difference}.",
                )
                self.expect(
                    np.allclose(position_difference, 0, atol=1),  # Up to 1mm is fine
                    f"{model_id} link {link_name} has different position in instance {instance_id} compared to instance 0. Position difference: {position_difference}.",
                )
                self.expect(
                    np.isclose(rotation_difference.magnitude(), 0, atol=np.deg2rad(1)),
                    f"{model_id} link {link_name} has different rotation in instance {instance_id} compared to instance 0. Rotation difference: {rotation_difference.magnitude()}.",
                )

        # Additionally, if the model group has more than one link, check warn if the base links of different instances
        # have different scale, because this might have broken things during runs of the match links script.
        if group["name_link_name"].nunique() > 1:
            one_base_link_scale = np.array(
                group[group["name_link_name"] == "base_link"]
                .iloc[0]
                .object.objectoffsetscale
            )
            base_link_rows = group[group["name_link_name"] == "base_link"]

            self.expect(
                all(
                    np.allclose(
                        one_base_link_scale, np.array(x.object.objectoffsetscale)
                    )
                    for _, x in base_link_rows.iterrows()
                ),
                f"Articulated object {group['name_model_id'].iloc[0]} instances have different scales for base links. This may have broken things during the match links script.",
                level="WARNING",
            )

    def run(self):
        self.reset()

        self.expect(rt.units.systemScale == 1, "System scale not set to 1mm.")

        self.expect(
            rt.units.systemType == rt.Name("millimeters"),
            "System scale not set to 1mm.",
        )

        # First get all of the objects
        df = pd.DataFrame(data={"object": [ObjectWrapper(x) for x in rt.objects]})

        # Add some helpful data
        df["object_name"] = df["object"].map(lambda x: x.name)
        df["base_object"] = df["object"].map(lambda x: x.baseObject)
        df["type"] = df["object"].map(lambda x: classOf(x))

        # Complain about and remove objects that are the wrong type.
        good_type = df.apply(self.is_valid_object_type, axis="columns")
        df = df[good_type]

        # Additionally filter out the cameras
        df = df[df["type"].isin([rt.Editable_Poly, rt.VRayLight])]

        # Complain about and remove objects that have unmatching name.
        good_name = df.apply(self.is_valid_name, axis="columns")
        df = df[good_name]

        if df.size == 0:
            return self.errors

        # Check that the object names are unique.
        duplicate_named_objs = df[df.duplicated(subset=["object_name"], keep="first")]
        duplicate_named_objs.apply(
            lambda row: self.expect(False, f"{row.object_name} is not unique."), axis=1
        )

        # Unwrap the name into its columns
        col_to_name = {
            v - 1: "name_" + k
            for k, v in b1k_pipeline.utils.NAME_PATTERN.groupindex.items()
        }
        applied_df = df.apply(
            lambda row: list(b1k_pipeline.utils.parse_name(row.object_name).groups()),
            axis="columns",
            result_type="expand",
        )
        applied_df.rename(columns=col_to_name, inplace=True)
        df = pd.concat([df, applied_df], axis="columns")
        # no-link-name objects are matched with the base link.
        df["name_link_name"] = df["name_link_name"].fillna("base_link")

        non_meta_polies = df[
            (df["type"] == rt.Editable_Poly) & df["name_meta_type"].isnull()
        ]
        non_meta_polies["vertex_count"] = non_meta_polies["object"].map(
            lambda x: rt.polyop.getNumVerts(x._obj)
        )
        non_meta_polies["face_count"] = non_meta_polies["object"].map(
            lambda x: rt.polyop.getNumFaces(x._obj)
        )

        # Check that when grouped by model ID, the instances all have the same set
        # of links, and that "base_link"  is included.
        grouped_by_model = non_meta_polies.groupby("name_model_id")
        grouped_by_model.apply(self.validate_model_group)

        # Run the single-object validation checks.
        non_meta_polies.apply(self.validate_object, axis="columns")

        # Check that instance name-based grouping is equal to instance-based grouping.
        groups_by_base_object = non_meta_polies.groupby(
            ["base_object"], sort=False, dropna=False
        )
        groups_by_base_object.apply(
            lambda group: self.expect(
                group.groupby(
                    ["name_category", "name_model_id", "name_link_name"],
                    sort=False,
                    dropna=False,
                ).ngroups
                == 1,
                f"{group.iloc[0].object_name} has instances that are named differently. If this is a link, note that different links should not be instances of each other.",
            )
        )

        # Here make groups that we EXPECT to all be instances of the same object.
        groups_by_max_instance_expectation = non_meta_polies.groupby(
            ["name_category", "name_model_id", "name_link_name"],
            sort=False,
            dropna=False,
        )
        groups_by_max_instance_expectation.apply(
            lambda group: self.expect(
                group.groupby(["base_object"], sort=False, dropna=False).ngroups == 1,
                f"{group.iloc[0].object_name} has objects that are expected to be instances upon looking at their name (e.g. same model ID and link name) but are not actually instances in 3ds Max.",
            )
        )

        # Let's use the base-object grouping for the instance group validation:
        groups_by_base_object.apply(self.validate_group_of_max_instances)

        lights = df[df["type"] == rt.VRayLight]
        lights.apply(self.validate_light, axis="columns")

        # TODO: Check that the lights are all non-bad
        # TODO: Check that each light object has a valid base.
        # TODO: Check that each light object is numbered consecutively.
        # TODO: Check that light-named objects have lights on them.

        return self.errors


def sanity_check_safe(batch=False):
    success = False
    errors = []
    warnings = []

    start_time = time.time()
    try:
        results = SanityCheck().run()
        errors = results["ERROR"]
        warnings = results["WARNING"]
        success = len(errors) == 0
    except Exception as e:
        t = traceback.format_exc()
        errors.append("Exception occurred:" + t)

    end_time = time.time()
    total_time = end_time - start_time

    # Print results in interactive mode.
    if not batch:
        print(f"Sanity check completed in {total_time:.2f} seconds.")
        if success:
            print("No errors found!")
        else:
            print("Errors found:")
            print("\n".join(errors))

        if warnings:
            print("Warnings:")
            print("\n".join(warnings))

    if batch:
        output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / OUTPUT_FILENAME, "w") as f:
            json.dump(
                {
                    "success": success,
                    "errors": errors,
                    "warnings": warnings,
                    "total_time": total_time,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    opts = rt.maxops.mxsCmdLineArgs

    batch = opts[rt.name("batch")] == "true"
    sanity_check_safe(batch=batch)
