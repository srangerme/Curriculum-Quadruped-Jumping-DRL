#!/usr/bin/env python3
"""Build linearly time-warped action replays from an evaluator trajectory."""

import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_json")
    parser.add_argument("output_dir")
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--amplitude", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.trajectory_json, encoding="utf-8") as source_file:
        trajectory = json.load(source_file)
    source_actions = [record["actions_policy"] for record in trajectory["records"]]
    if not source_actions:
        raise ValueError("trajectory contains no action records")
    os.makedirs(args.output_dir, exist_ok=True)

    for scale in (float(value) for value in args.scales.split(",")):
        if scale <= 0.0:
            raise ValueError("time scales must be positive")
        output_length = max(1, round((len(source_actions) - 1) * scale) + 1)
        output_path = os.path.join(
            args.output_dir,
            "actions_time_{:.3f}_amp_{:.3f}.csv".format(scale, args.amplitude),
        )
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            for output_index in range(output_length):
                source_position = min(output_index / scale, len(source_actions) - 1)
                lower_index = int(source_position)
                upper_index = min(lower_index + 1, len(source_actions) - 1)
                blend = source_position - lower_index
                action = [
                    args.amplitude
                    * ((1.0 - blend) * lower + blend * upper)
                    for lower, upper in zip(
                        source_actions[lower_index], source_actions[upper_index]
                    )
                ]
                writer.writerow(action)
        print(output_path)


if __name__ == "__main__":
    main()
