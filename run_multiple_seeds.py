import subprocess
import os
from pathlib import Path
import argparse
import json
import re


def read_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def read_and_update_exp_name(method_name: str, entity_file: str = "wandb_entity.txt") -> int:
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
    return new_count


def run_experiment(script_path, seed, exp_name, output_dir, gym_env_id, track=False, capture_video=False, extra_args=""):
    script_path = Path(script_path).resolve()
    run_name = f"{exp_name}_seed{seed}"
    log_path = Path(output_dir) / f"{run_name}.log"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", str(script_path),
        "--seed", str(seed),
        "--exp_name", exp_name,
        "--env_id", gym_env_id
    ]

    if track:
        cmd.extend([
            "--track",
            "--wandb_project_name", exp_name
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
                        help="Short name of the method (e.g. SAC)")
    parser.add_argument("--env_name", type=str, required=True,
                        help="Short name of the environment (e.g. Hopper)")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of seeds to run")
    parser.add_argument("--track", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--capture_video", action="store_true",
                        help="Enable video capture")
    parser.add_argument("--extra_args", type=str, default="",
                        help="Extra args to pass to script")
    parser.add_argument("--entity_file", type=str, default="wandb_entity.txt",
                        help="File that tracks experiment numbers")
    parser.add_argument("--config_file", type=str, default="experiment_config.json",
                        help="Mapping of method -> script/output_dir")
    parser.add_argument("--env_map_file", type=str, default="env_map.json",
                        help="Mapping of env_name -> gym env ID")

    args = parser.parse_args()

    # Load configs
    method_config = read_json(args.config_file)
    env_map = read_json(args.env_map_file)

    if args.method_name not in method_config:
        raise ValueError(f"Method '{args.method_name}' not found in {args.config_file}")
    if args.env_name not in env_map:
        raise ValueError(f"Env '{args.env_name}' not found in {args.env_map_file}")

    script_path = method_config[args.method_name]["script"]
    output_dir = method_config[args.method_name]["output_dir"]
    gym_env_id = env_map[args.env_name]

    # Create exp_name like SAC5_Hopper
    exp_num = read_and_update_exp_name(args.method_name, args.entity_file)
    exp_name = f"{args.method_name}{exp_num}_{args.env_name}"

    print(f"Launching experiment: {exp_name} → {gym_env_id}")

    for seed in range(args.num_seeds):
        run_experiment(
            script_path=script_path,
            seed=seed,
            exp_name=exp_name,
            output_dir=output_dir,
            gym_env_id=gym_env_id,
            track=args.track,
            capture_video=args.capture_video,
            extra_args=args.extra_args,
        )


if __name__ == "__main__":
    main()
