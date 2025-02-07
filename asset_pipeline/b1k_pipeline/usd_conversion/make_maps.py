import csv
import os
import pathlib

import numpy as np
from PIL import Image

PIPELINE_ROOT = pathlib.Path(__file__).parents[2]

RESOLUTION = 1.0
Z_START = 2.  # Just above the typical robot height
Z_END = -0.1  # Just below the floor
HALF_Z = (Z_START + Z_END) / 2.
HALF_HEIGHT = (Z_START - Z_END) / 2.

WALL_CATEGORIES = ["walls", "fence"]
FLOOR_CATEGORIES = ["floors", "driveway", "lawn"]
DOOR_CATEGORIES = ["door", "garage_door", "gate"]
IGNORE_CATEGORIES = ["carpet"]
NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES + WALL_CATEGORIES

LOAD_NOT_LOAD_MAPPING = {
    # key: (load, not_load)
    "floor_trav_no_obj_0.png": (NEEDED_STRUCTURE_CATEGORIES, None),
    "floor_trav_0.png": (None, IGNORE_CATEGORIES),
    "floor_trav_no_door_0.png": (None, DOOR_CATEGORIES + IGNORE_CATEGORIES),
    "floor_trav_open_door_0.png": (None, IGNORE_CATEGORIES),
}


def world_to_map(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in world (simulator) reference frame into map reference frame

    :param xy: 2D location in world reference frame (metric)
    :return: 2D location in map reference frame (image)
    """
    return np.flip((np.array(xy) / trav_map_resolution + trav_map_size / 2.0)).round().astype(int)


def map_to_world(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in map reference frame into world (simulator) reference frame

    Args:
        xy (2-array or (N, 2)-array): 2D location(s) in map reference frame (in image pixel space)

    Returns:
        2-array or (N, 2)-array: 2D location(s) in world reference frame (in metric space)
    """
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((xy - trav_map_size / 2.0) * trav_map_resolution, axis=axis)


def generate_maps_for_current_scene(scene_id):
    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.object_states as object_states

    # Create the output directory
    save_path = os.path.join(gm.DATASET_PATH, "scenes", scene_id, "layout")
    os.makedirs(save_path, exist_ok=True)

    # Get the room type to room id mapping
    with open(PIPELINE_ROOT / "metadata/allowed_room_types.csv") as f:
        sem_to_id = {row["Room Name"].strip(): i + 1 for i, row in enumerate(csv.DictReader(f))}

    for fname, (load_categories, not_load_categories) in LOAD_NOT_LOAD_MAPPING.items():
        # Move the doors to the open position if necessary
        if fname == "floor_trav_open_door_0.png":
            for door_cat in DOOR_CATEGORIES:
                for door in og.sim.scenes[0].object_registry("category", door_cat, []):
                    if object_states.Open not in door.states:
                        continue
                    door.states[object_states.Open].set_value(True, fully=True)

        # Compute the map dimensions by finding the AABB of all objects and calculating max distance from origin.
        floor_objs = {
            floor
            for floor_cat in FLOOR_CATEGORIES
            for floor in og.sim.scenes[0].object_registry("category", floor_cat, [])
        }
        roomless_floor_objs = [(floor, len(floor.in_rooms)) for floor in floor_objs if len(floor.in_rooms) != 1]
        assert not roomless_floor_objs, f"Found {len(roomless_floor_objs)} floor objects without exactly one room: {roomless_floor_objs}"
        aabb_corners = np.concatenate([floor.aabb for floor in floor_objs], axis=0)
        combined_low = np.min(list(aabb_corners), axis=0)
        combined_high = np.max(list(aabb_corners), axis=0)
        combined_aabb = np.array([combined_low, combined_high])
        aabb_dist_from_zero = np.abs(combined_aabb)
        dist_from_center = np.max(aabb_dist_from_zero)
        map_size_in_meters = dist_from_center * 2
        map_size_in_pixels = map_size_in_meters / RESOLUTION
        map_size_in_pixels = int(np.ceil(map_size_in_pixels / 2) * 2)  # Round to nearest multiple of 2

        # Initialize the map array
        new_trav_map = np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8)
        assert new_trav_map.shape[0] == new_trav_map.shape[1]

        # Get a view to the part of the map that we will actually cast rays for (e.g. the occupied section)
        x_min, y_min = world_to_map(combined_aabb[0][:2], RESOLUTION, map_size_in_pixels)
        x_max, y_max = world_to_map(combined_aabb[1][:2], RESOLUTION, map_size_in_pixels)
        scannable_map = new_trav_map[x_min:x_max+1, y_min:y_max+1]

        # Get the points to cast rays from
        pixel_indices = np.array(list(np.ndindex(scannable_map.shape)), dtype=int)
        corresponding_world_centers = map_to_world(pixel_indices + np.array([[x_min, y_min]]), RESOLUTION, map_size_in_pixels)

        # Using the load/not load params, build the set of allowed hits
        allowed_hit_paths = {
            link.prim_path: obj
            for obj in og.sim.scenes[0].objects
            for link in obj.links.values()
            if not load_categories or obj.category in load_categories
        }
        if not_load_categories:
            for obj in og.sim.scenes[0].objects:
                for link in obj.links.values():
                    if obj.category in not_load_categories:
                        allowed_hit_paths.pop(link.prim_path, None)

        # Get the ray cast results (in batches so that pybullet does not complain)
        hit_object_sets = [set() for _ in corresponding_world_centers]
        for i, cwc in enumerate(corresponding_world_centers):
            def _check_hit(hit):
                if hit.rigid_body in allowed_hit_paths:
                    hit_object_sets[i].add(allowed_hit_paths[hit.rigid_body])
                
                return True
                
            og.sim.psqi.overlap_box(
                halfExtent=np.array([RESOLUTION / 2, RESOLUTION / 2, HALF_HEIGHT]),
                pos=np.array([cwc[0], cwc[1], HALF_Z]),
                rot=np.array([0, 0, 0, 1.0]),
                reportFn=_check_hit,
            )

        # Check which rays hit *only* floors
        hit_floor = np.array([hit_objects.issubset(floor_objs) for hit_objects in hit_object_sets]).astype(np.uint8)
        scannable_map[:, :] = np.reshape(hit_floor * 255, scannable_map.shape)
        Image.fromarray(new_trav_map).save(os.path.join(save_path, fname))

        # At the same time as the no-obj trav map, we generate the segmentation maps.
        if fname == "floor_trav_no_obj_0.png":
            # Get a list of all of the room instances in the scene
            all_insts = {
                room
                for floor in og.sim.scenes[0].objects
                for room in (floor.in_rooms if floor.in_rooms else [])
            }
            sorted_all_insts = sorted(all_insts)

            # Map those rooms into a contiguous range of integers starting from 1
            inst_to_id = {inst: i + 1 for i, inst in enumerate(sorted_all_insts)}

            # Color the instance segmentation map using the hit objects' 
            insseg_map_fname = "floor_insseg_0.png"
            insseg_map = np.zeros_like(new_trav_map, dtype=np.uint8)
            scannable_insseg_map = insseg_map[x_min:x_max+1, y_min:y_max+1]
            first_hit_floors = [next(iter(sorted(hit_objects & floor_objs, key=lambda x: x.name)), None) if hit_objects else None for hit_objects in hit_object_sets]
            hit_room_inst_name = [
                hit_obj.in_rooms[0] if hit_obj and hit_obj.in_rooms else None
                for hit_obj in first_hit_floors
            ]
            insseg_val = np.array([inst_to_id[inst] if inst else 0 for inst in hit_room_inst_name], dtype=np.uint8)
            scannable_insseg_map[:, :] = np.reshape(insseg_val, scannable_insseg_map.shape)
            Image.fromarray(insseg_map).save(os.path.join(save_path, insseg_map_fname))

            # Now the same for the semseg map
            semseg_map_fname = "floor_semseg_0.png"
            semseg_map = np.zeros_like(new_trav_map, dtype=np.uint8)
            scannable_semseg_map = semseg_map[x_min:x_max+1, y_min:y_max+1]
            hit_room_type = [x.rsplit("_", 1)[0] if x else None for x in hit_room_inst_name]
            semseg_val = np.array([sem_to_id[rm_type] if rm_type else 0 for rm_type in hit_room_type], dtype=np.uint8)
            scannable_semseg_map[:, :] = np.reshape(semseg_val, scannable_semseg_map.shape)
            Image.fromarray(semseg_map).save(os.path.join(save_path, semseg_map_fname))
