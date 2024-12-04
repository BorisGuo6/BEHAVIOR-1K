import sys
import traceback
sys.path.append(r"D:\ig_pipeline")

import os
import tempfile
import traceback
import cv2
import numpy as np
from pymxs import runtime as rt
import json
import re
from collections import defaultdict
import b1k_pipeline.utils

import trimesh
from fs.zipfs import ZipFS
from fs.osfs import OSFS
from fs.tempfs import TempFS
import fs.copy

USE_NATIVE_EXPORT = False

allow_list = [
]
black_list = [
]

def save_collision_mesh(obj, output_fs):
    # Assert that collision meshes do not share instances in the scene
    assert not [x for x in rt.objects if x.baseObject == obj.baseObject and x != obj], f"{obj.name} should not have instances."

    # Get vertices and faces into numpy arrays for conversion
    verts = np.array([rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))])
    faces = np.array(rt.polyop.getFacesVerts(obj, range(1, rt.polyop.GetNumFaces(obj) + 1))) - 1
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    # Split the faces into elements
    elems = {tuple(rt.polyop.GetElementsUsingFace(obj, i + 1)) for i in range(rt.polyop.GetNumFaces(obj))}
    assert len(elems) <= 32, f"{obj.name} should not have more than 32 elements."
    elems = np.array(list(elems))
    assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"
    
    # Iterate through the elements
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        assert m.is_volume, f"{obj.name} element {i} is not a volume"
        # assert m.is_convex, f"{obj.name} element {i} is not convex"
        assert len(m.split()) == 1, f"{obj.name} element {i} has elements trimesh still finds splittable"
        
        # Save the mesh into an obj file
        assert b1k_pipeline.utils.save_mesh(m, output_fs, obj.name + f"-{i}.obj"), f"{obj.name} element {i} could not be saved"

def save_mesh(obj, output_path):  
    # Get vertices and faces into numpy arrays for conversion
    verts = np.array([rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))])
    faces = np.array(rt.polyop.getFacesVerts(obj, range(1, rt.polyop.GetNumFaces(obj) + 1))) - 1
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"
    
    # Convert to Trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.remove_unreferenced_vertices()
        
    # Save the mesh into an obj file
    dirname = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    assert b1k_pipeline.utils.save_mesh(m, OSFS(dirname), basename), f"{obj.name} could not be saved"

class ObjectExporter:
    def __init__(self, obj_out_dir):
        self.obj_out_dir = obj_out_dir
        assert os.path.exists(obj_out_dir)

    def get_process_objs(self):
        objs = []
        wrong_objs = []
        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue
            if allow_list and all(re.fullmatch(p, obj.name) is None for p in allow_list):
                continue
            if black_list and any(re.fullmatch(p, obj.name) is not None for p in black_list):
                continue
            parsed_name = b1k_pipeline.utils.parse_name(obj.name)
            if parsed_name is None:
                wrong_objs.append((obj.name, rt.ClassOf(obj)))
                continue
            if parsed_name.group("meta_type"):
                continue

            obj_dir = os.path.join(self.obj_out_dir, obj.name)
            obj_file = os.path.join(obj_dir, obj.name + ".obj")
            json_file = os.path.join(obj_dir, obj.name + ".json")
            if os.path.exists(obj_file) and os.path.exists(json_file):
               continue
            objs.append(obj)
        
        for light in rt.lights:
            if b1k_pipeline.utils.parse_name(light.name) is None:
                wrong_objs.append((light.name, rt.ClassOf(light)))
        
        if len(wrong_objs) != 0:
            for obj_name, obj_type in wrong_objs:
                print(obj_name, obj_type)
            assert False, wrong_objs

        objs.sort(key=lambda obj: int(b1k_pipeline.utils.parse_name(obj.name).group("instance_id")))

        return objs

    def export_obj(self, obj):
        print("export_meshes", obj.name)

        obj_dir = os.path.join(self.obj_out_dir, obj.name)
        os.makedirs(obj_dir, exist_ok=True)

        # WARNING: we don't know how to set fine-grained setting of OBJ export. It always inherits the setting of the last export. 
        obj_path = os.path.join(obj_dir, obj.name + ".obj")

        if USE_NATIVE_EXPORT:
            rt.select(obj)

            # rt.ObjExp.setIniName(os.path.join(os.path.parent(__file__), "gw_objexp.ini"))
            assert rt.getIniSetting(rt.ObjExp.getIniName(), "Material", "UseMapPath") == "1", "Map path not used."
            assert rt.getIniSetting(rt.ObjExp.getIniName(), "Material", "MapPath") == "./material/", "Wrong material path."
            assert rt.getIniSetting(rt.ObjExp.getIniName(), "Geometry", "FlipZyAxis") == "0", "Should not flip axes when exporting."
            assert rt.units.systemScale == 1, "System scale not set to 1mm."
            assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

            rt.exportFile(obj_path, rt.Name("noPrompt"), selectedOnly=True, using=rt.ObjExp)
        else:
            save_mesh(obj, obj_path)

        assert os.path.exists(obj_path), f"Could not export object {obj.name}"
        assert os.path.exists(os.path.join(obj_dir, "material")), f"Could not export materials for object {obj.name}"
        self.export_obj_metadata(obj, obj_dir)

    def export_obj_metadata(self, obj, obj_dir):
        print("export_meshes_metadata", obj.name)

        metadata = {}
        # Export the canonical orientation
        metadata["orientation"] = [obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w]
        metadata["meta_links"] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        metadata["parts"] = []
        metadata["layer_name"] = obj.layer.name
        obj_name_result = b1k_pipeline.utils.parse_name(obj.name)
        assert obj_name_result, f"Unparseable object name {obj.name}"
        for light in rt.lights:
            light_name_result = b1k_pipeline.utils.parse_name(light.name)
            assert light_name_result, f"Unparseable light name {light.name}"
            if obj_name_result.group("category") == light_name_result.group("category") and obj_name_result.group("model_id") == light_name_result.group("model_id") and obj_name_result.group("instance_id") == light_name_result.group("instance_id"):
                # assert light.normalizeColor == 1, "The light's unit is NOT lm."
                assert light_name_result.group("light_id"), "The light does not have an ID."
                light_id = str(int(light_name_result.group("light_id")))
                metadata["meta_links"]["lights"][light_id]["0"] = {
                    "type": light.type,
                    "length": light.sizeLength,
                    "width": light.sizeWidth,
                    "color": [light.color.r, light.color.g, light.color.b],
                    "intensity": light.multiplier,
                    "position": [light.objecttransform.position.x, light.objecttransform.position.y, light.objecttransform.position.z],
                    "orientation": [light.objecttransform.rotation.x, light.objecttransform.rotation.y, light.objecttransform.rotation.z, light.objecttransform.rotation.w],
                }

        for child in obj.children:
            child_name_result = b1k_pipeline.utils.parse_name(child.name)

            # Take care of exporting object parts.
            if rt.classOf(child) in (rt.Editable_Poly, rt.PolyMeshObject):
                if child_name_result.group("meta_type"):
                    # Save collision mesh.
                    assert child_name_result.group("meta_type") == "collision", f"Only Mcollision can be a mesh."
                    save_collision_mesh(child, OSFS(obj_dir))
                else:
                    # Save part metadata.
                    metadata["parts"].append(child.name)
                continue

            is_valid_meta = rt.classOf(child) in {rt.Point, rt.Box, rt.Cylinder, rt.Sphere, rt.Cone}
            assert is_valid_meta, f"Meta link {child.name} has unexpected type {rt.classOf(child)}"

            if not child_name_result.group("meta_type"):
                continue

            meta_info = child_name_result.group("meta_info")
            meta_type = child_name_result.group("meta_type")
            meta_id_str = child_name_result.group("meta_id")
            meta_id = "0" if meta_id_str is None else meta_id_str
            meta_subid_str = child_name_result.group("meta_subid")
            meta_subid = "0" if meta_subid_str is None else meta_subid_str
            assert meta_subid not in metadata["meta_links"][meta_type][meta_id], f"Meta subID {meta_info} is repeated in object {obj.name}"
            object_transform = child.objecttransform  # This is a 4x3 array
            position = object_transform.position
            rotation = object_transform.rotation
            scale = np.array(list(object_transform.scale))

            metadata["meta_links"][meta_type][meta_id][meta_subid] = {
                "position": list(position),
                "orientation": [rotation.x, rotation.y, rotation.z, rotation.w],
            }

            if rt.classOf(child) == rt.Sphere:
                size = np.array([child.radius, child.radius, child.radius]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "sphere"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)
            elif rt.classOf(child) == rt.Box:
                size = np.array([child.width, child.length, child.height]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "box"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)
            elif rt.classOf(child) == rt.Cylinder:
                size = np.array([child.radius, child.radius, child.height]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "cylinder"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)
            elif rt.classOf(child) == rt.Cone:
                assert np.isclose(child.radius1, 0), f"Cone radius1 should be 0 for {child.name}"
                size = np.array([child.radius2, child.radius2, child.height]) * scale
                metadata["meta_links"][meta_type][meta_id][meta_subid]["type"] = "cone"
                metadata["meta_links"][meta_type][meta_id][meta_subid]["size"] = list(size)

        # Convert the subID to a list
        for keyed_by_id in metadata["meta_links"].values():
            for id, keyed_by_subid in keyed_by_id.items():
                found_subids = set(keyed_by_subid.keys())
                expected_subids = {str(x) for x in range(len(found_subids))}
                assert found_subids == expected_subids, f"{obj.name} has non-continuous subids {sorted(found_subids)}"
                int_keyed = {int(subid): meshes for subid, meshes in keyed_by_subid.items()}
                keyed_by_id[id] = [meshes for _, meshes in sorted(int_keyed.items())]

        json_file = os.path.join(obj_dir, obj.name + ".json")
        with open(json_file, "w") as f:
            json.dump(metadata, f)

    def fix_mtl_files(self, obj):
        obj_dir = os.path.join(self.obj_out_dir, obj.name)
        for file in os.listdir(obj_dir):
            if file.endswith(".mtl"):
                mtl_file = os.path.join(obj_dir, file)
                new_lines = []
                with open(mtl_file, "r") as f:
                    for line in f.readlines():
                        if "map" in line:
                            line = line.replace("material\\", "material/")
                            map_path = os.path.join(obj_dir, line.split(" ")[1].strip())
                            # For some reason, pybullet won't load the texture files unless we save it again with OpenCV
                            img = cv2.imread(map_path)
                            # These two maps need to be flipped
                            # glossiness -> roughness
                            # translucency -> opacity
                            if "VRayMtlReflectGlossinessBake" in map_path or "VRayRawRefractionFilterMap" in map_path:
                                print(f"Flipping {os.path.basename(map_path)}")
                                img = 1 - img.astype(float)/255

                                # Apply a sqrt here to glossiness to make it harder for things to be very shiny.
                                if "VRayMtlReflectGlossinessBake" in map_path: 
                                    img = np.sqrt(img)
                                    
                                img = (img * 255).astype(np.uint8)

                            cv2.imwrite(map_path, img)

                        new_lines.append(line)

                with open(mtl_file, "w") as f:
                    for line in new_lines:
                        f.write(line)

    def run(self):
        # assert rt.classOf(rt.renderers.current) == rt.V_Ray_5__update_2_3, f"Renderer should be set to V-Ray 5.2.3 CPU instead of {rt.classOf(rt.renderers.current)}"
        assert rt.execute('max modify mode')

        objs = self.get_process_objs()
        should_bake_current = 0
        failures = {}
        for i, obj in enumerate(objs):
            try:
                print(f"{i+1} / {len(objs)} total")

                obj.isHidden = False
                for child in obj.children:
                    child.isHidden = False

                self.export_obj(obj)   
                self.fix_mtl_files(obj)
            except Exception as e:
                failures[obj.name] = traceback.format_exc()

        failure_msg = "\n".join(f"{obj}: {err}" for obj, err in failures.items())
        assert len(failures) == 0, f"Some objects could not be exported:\n{failure_msg}"

def main():
    out_dir = os.path.join(rt.maxFilePath, "artifacts")

    success = True
    error_msg = ""
    try:
        with TempFS(temp_dir=r"D:\tmp") as obj_out_fs, \
             ZipFS(os.path.join(out_dir, "meshes.zip"), write=True, temp_fs=obj_out_fs) as zip_fs:
            exp = ObjectExporter(obj_out_fs.getsyspath("/"))
            exp.run()

            print("Move files to archive.")

        print("Finished copying.")
    except:
        success = False
        error_msg = traceback.format_exc()

    json_file = os.path.join(out_dir, "export_meshes.json")
    with open(json_file, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f)

    if success:
        print("Export successful!")
    else:
        print("Export failed.")
        print(error_msg)


if __name__ == "__main__":
    main()