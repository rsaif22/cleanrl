import subprocess
import os
from pathlib import Path
import argparse
import json
import re


def read_experiment_config(config_path="experiment_config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def read_and_update_exp_name(method_name: str, entity_file: str = "wandb_entity.txt") -> str:
    records = {}
    if os.path.exists(entity_file):
        with open(entity_file, "r") as f:
            for line in f:
                match = re.match(r"(\w+):\s*(\d+)", line.strip())
                if match:
                    records[match.group(1)] = int(match.group(2))
    new_count = records.get(method_name, 0) + 1
    records[method_name] = new_count
    with open(entity_file, "w") as f:
        for k, v in sorted(records.items()):
            f.write(f"{k}: {v}\n")
    return f"{method_name}{new_count}"


def run_experiment(script_path, seed, exp_name, output_dir, track=False, capture_video=False, extra_args=""):
    script_path = Path(script_path).resolve()
    run_name = f"{exp_name}_seed{seed}"
    log_path = Path(output_dir) / f"{run_name}.log"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", str(script_path),
        "--seed", str(seed),
        "--exp_name", exp_name,
    ]

    if track:
        cmd.extend([
            "--track",
            "--wandb_project_name", exp_name  # project name same as exp_name (e.g., SAC4)
        ])

    if capture_video:
        cmd.append("--capture_video")

    if extra_args:
        cmd.extend(extra_args.split())

    print(f"Running: {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", type=str, required=True,
                        help="Short method name (e.g. SAC, PPO)")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of seeds to run")
    parser.add_argument("--track", action="store_true",
                        help="Enable W&B tracking")
    parser.add_argument("--capture_video", action="store_true",
                        help="Capture env videos")
    parser.add_argument("--extra_args", type=str, default="",
                        help="Extra args to pass to script")
    parser.add_argument("--entity_file", type=str, default="wandb_entity.txt",
                        help="File tracking experiment counts per method")
    parser.add_argument("--config_file", type=str, default="experiment_config.json",
                        help="Config JSON with script/output_dir for methods")

    args = parser.parse_args()
    config = read_experiment_config(args.config_file)

    if args.method_name not in config:
        raise ValueError(f"Method {args.method_name} not found in config file.")

    script_path = config[args.method_name]["script"]
    output_dir = config[args.method_name]["output_dir"]
    exp_name = read_and_update_exp_name(args.method_name, args.entity_file)

    print(f"Auto-generated experiment/project name: {exp_name}")

    for seed in range(args.num_seeds):
        run_experiment(
            script_path=script_path,
            seed=seed,
            exp_name=exp_name,
            output_dir=output_dir,
            track=args.track,
            capture_video=args.capture_video,
            extra_args=args.extra_args,
        )


if __name__ == "__main__":
    main()
