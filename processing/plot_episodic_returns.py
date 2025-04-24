import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_algo_paths(env):
    return {
        "SAC": f"data/sac_{env}_aligned.csv",
        # "SAC-POV": f"data/sac_pov_{env}_aligned.csv",
        "SACD": f"data/sacd_{env}_aligned.csv",
        "SACD5": f"data/sacd5_{env}_aligned.csv",
        "SAC-3Q": f"data/sac3q_{env}_aligned.csv",
        "SAC-5Q": f"data/sac5q_{env}_aligned.csv",
        "SAC-10Q": f"data/sac10q_{env}_aligned.csv",
        "PPO": f"data/ppo_{env}_aligned.csv",
        "TD3": f"data/td3_{env}_aligned.csv",
    }


def plot_multiple_episodic_returns(selected_algos, algo_paths, output_image=None, smoothing=1):
    plt.figure(figsize=(10, 6))
    for label in selected_algos:
        if label not in algo_paths:
            print(f"Warning: Algorithm '{label}' not recognized. Skipping.")
            continue
        csv_path = algo_paths[label]
        df = pd.read_csv(csv_path)
        steps = df["global_step"].values
        means = df["episodic_return_mean"].values
        stds = df["episodic_return_std"].values

        # Optional smoothing
        window_size = smoothing
        means = np.convolve(means, np.ones(window_size)/window_size, mode='same')

        plt.plot(steps, means, label=label)
        plt.fill_between(
            steps,
            means - stds,
            means + stds,
            alpha=0.3
        )
    plt.xlabel("Global Step")
    plt.ylabel("Episodic Return")
    plt.title("Episodic Return vs Global Step")
    plt.legend()
    plt.grid(True)
    if output_image:
        if not output_image.startswith("figures/"):
            output_image = "figures/" + output_image
        plt.savefig(output_image)
        print(f"Plot saved to {output_image}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, default="hopper",
    )
    parser.add_argument(
        "--smoothing", type=int, default=1,
        help="Window size for moving average smoothing")
    
    parser.add_argument(
        "--algos", nargs="+", default=["SAC", "SAC-3Q", "SAC-5Q"],
        help="List of algorithm labels to include in the plot (e.g., SAC SAC-3Q SAC-5Q)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the plot as an image file (optional)"
    )
    parser.add_argument(
        "--all_algos", action="store_true",
        help="Plot all algorithms available for the environment"
    )
    args = parser.parse_args()
    algo_paths = get_algo_paths(args.env_name)
    if args.all_algos:
        args.algos = list(algo_paths.keys())
    plot_multiple_episodic_returns(args.algos, algo_paths, args.output, args.smoothing)
