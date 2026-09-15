#!/usr/bin/env python3
"""Add render-only visuals copied from URDF collision geometry."""

import argparse
import copy
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument(
        "--replace-existing-visuals",
        action="store_true",
        help=(
            "Remove mesh visuals before adding collision primitives. Isaac Gym's "
            "offscreen URDF importer may render only the first visual per link."
        ),
    )
    args = parser.parse_args()

    tree = ET.parse(args.source)
    root = tree.getroot()
    for link in root.findall("link"):
        if args.replace_existing_visuals:
            for visual in list(link.findall("visual")):
                link.remove(visual)
        for index, collision in enumerate(list(link.findall("collision"))):
            visual = copy.deepcopy(collision)
            visual.tag = "visual"
            visual.set("name", f"collision_visual_{index}")
            material = ET.SubElement(visual, "material", {"name": "collision_gray"})
            ET.SubElement(material, "color", {"rgba": "0.35 0.38 0.42 1"})
            link.append(visual)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
