import argparse
from collections import Counter, defaultdict
import glob
import json
import os

import yaml
import b1k_pipeline.utils

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}


def main(use_future=False):
    needed = set()
    providers = defaultdict(list)
    meta_links = defaultdict(set)
    needed_by = defaultdict(list)
    seen_as = defaultdict(set)
    skipped_files = []

    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        # Get the providers.
        with pipeline_fs.open("params.yaml", "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            targets = (
                params["combined_unfiltered"] if use_future else params["combined"]
            )

        # Merge the object lists.
        for target in targets:
            with pipeline_fs.target_output(target) as target_output_fs:
                with target_output_fs.open("object_list.json", "r") as f:
                    object_list = json.load(f)

                if not object_list["success"]:
                    skipped_files.append(target)
                    continue

                needed |= set(object_list["needed_objects"])
                for obj in object_list["needed_objects"]:
                    needed_by[obj].append(target)
                for provided in object_list["provided_objects"]:
                    providers[provided].append(target)
                for obj, links in object_list["meta_links"].items():
                    meta_links[obj].update(links)

                # Manually generate pseudo-metalinks for parts
                for name, _, parent in object_list["max_tree"]:
                    if not parent:
                        continue

                    # Check parseable name
                    parsed_name = b1k_pipeline.utils.parse_name(name)
                    if parsed_name is None:
                        continue

                    # Record the seen-as categories
                    seen_as[parsed_name.group("model_id")].add(
                        parsed_name.group("category")
                    )

                    # Get the tags that are on the parent
                    tags_str = parsed_name.group("tag")
                    tags = set()
                    if tags_str:
                        tags = {x[1:] for x in tags_str.split("-") if x}

                    part_tags = tags & ALLOWED_PART_TAGS
                    if not part_tags:
                        continue

                    # If we have part tags, we need to add them to the parent's meta links.
                    # For that, get the parent's name.
                    parsed_parent_name = b1k_pipeline.utils.parse_name(parent)
                    assert (
                        parsed_parent_name
                    ), f"Parent {parent} of {name} is not parseable"
                    parent_category = parsed_parent_name.group("category")
                    parent_model = parsed_parent_name.group("model_id")
                    parent_name = f"{parent_category}-{parent_model}"

                    # Check that the parent shows up in the inventory
                    assert (
                        parent_name in needed_by
                    ), f"Parent {parent_name} of {name} is not in the inventory"

                    # Add the part tags as meta links to the parent
                    meta_links[parent_name].update(part_tags)

        # Check the multiple-provided copies.
        multiple_provided = {k: v for k, v in providers.items() if len(v) > 1}
        single_provider = {k: v[0] for k, v in providers.items()}

        provided_objects = set(single_provider.keys())
        missing_objects = {x.split("-")[1] for x in needed} - {
            x.split("-")[1] for x in provided_objects
        }
        missing_objects &= {
            obj.split("-")[1]
            for obj, needers in needed_by.items()
            if any(t in params["final_scenes"] for t in needers)
        }  # Limit this to stuff that shows up in final scenes

        seen_as_multiple_categories = {
            obj_id: sorted(categories)
            for obj_id, categories in seen_as.items()
            if len(categories) > 1
        }

        id_occurrences = defaultdict(list)
        for obj_name, provider in single_provider.items():
            id_occurrences[obj_name.split("-")[1]].append(provider)
        id_collisions = {
            obj_id: obj_names
            for obj_id, obj_names in id_occurrences.items()
            if len(obj_names) > 1
        }

        success = (
            len(skipped_files) == 0
            and len(multiple_provided) == 0
            and len(missing_objects) == 0
            and len(id_collisions) == 0
            and len(seen_as_multiple_categories) == 0
        )
        results = {
            "success": success,
            "providers": single_provider,
            "needed_by": needed_by,
            "meta_links": {x: sorted(y) for x, y in meta_links.items()},
            "error_skipped_files": sorted(skipped_files),
            "error_multiple_provided": multiple_provided,
            "error_missing_objects": sorted(missing_objects),
            "error_id_collisions": id_collisions,
            "error_seen_as_multiple_categories": seen_as_multiple_categories,
        }
        with pipeline_fs.pipeline_output() as pipeline_output_fs:
            json_path = (
                "object_inventory_future.json"
                if use_future
                else "object_inventory.json"
            )
            with pipeline_output_fs.open(json_path, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce a list of objects included in the pipeline."
    )
    parser.add_argument("--future", action="store_true")
    args = parser.parse_args()

    main(args.future)
