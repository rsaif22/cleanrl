
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

# HalfCheetah
sac_file = "sac_cheetah_aligned.csv"
# sacd_file = "sacd_cheetah_aligned.csv"
# sac3q_file = "sac3q_cheetah_aligned.csv"
# sac5q_file = "sac5q_cheetah_aligned.csv"
sac_pov_file = "sac_pov_cheetah_aligned.csv"
sacfs_pov_file = "sacfs_pov_cheetah_aligned.csv"

# Hopper2D
# sac_file = "sac_hopper_aligned.csv"
# sacd_file = "sacd_hopper_aligned.csv"
# sac3q_file = "sac3q_hopper_aligned.csv"
# sac5q_file = "sac5q_hopper_aligned.csv"
# sac_pov_file = "sac_pov_hopper_aligned.csv"
# sacfs_pov_file = "sacfs_pov_hopper_aligned.csv"

algos = {
    "SAC": sac_file,
    "SAC (POMDP)": sac_pov_file,
    "SAC-FS (POMDP)": sacfs_pov_file,
    # "SAC-D": sacd_file,
    # "SAC-3Q": sac3q_file,
    # "SAC-5Q": sac5q_file
}

def plot_multiple_episodic_returns(algos, output_image=None):
    plt.figure(figsize=(10, 6))
    for label, csv_path in algos.items():
        df = pd.read_csv(csv_path)
        steps = df["global_step"].values
        means = df["episodic_return_mean"].values
        stds = df["episodic_return_std"].values

        # Moving average for smoothing
        window_size = 50
        means = np.convolve(means, np.ones(window_size)/window_size, mode='same')
        # steps = steps[window_size-1:]  # Adjust steps to match the smoothed means
        # steps = steps[:len(means)]  # Adjust steps to match the smoothed means
        # stds = np.convolve(stds, np.ones(window_size)/window_size, mode='valid')

        plt.plot(steps, means, label=label)
        plt.fill_between(
            steps,
            means - stds,
            means + stds,
            alpha=0.3
            # label=f"{label} ±1 Std Dev"
        )
    plt.xlabel("Global Step")
    plt.ylabel("Episodic Return")
    plt.title("Episodic Return vs Global Step")
    plt.legend()
    plt.grid(True)
    if output_image:
        plt.savefig(output_image)
        print(f"Plot saved to {output_image}")
    else:
        plt.show()

def plot_episodic_returns(csv_path, output_image=None):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    steps = df["global_step"].values
    means = df["episodic_return_mean"].values
    stds = df["episodic_return_std"].values

    # Moving average for smoothing
    window_size = 10
    # means = np.convolve(means, np.ones(window_size)/window_size, mode='same')
    # plt.plot(df["global_step"], df["episodic_return_mean"], label="Mean Episodic Return")
    plt.plot(steps, means, label="Mean Episodic Return", color='blue')


    # plt.fill_between(
    #     df["global_step"],
    #     df["episodic_return_mean"] - df["episodic_return_std"],
    #     df["episodic_return_mean"] + df["episodic_return_std"],
    #     alpha=0.3,
    #     label="±1 Std Dev"
    # )
    plt.fill_between(
        steps,
        means - stds,
        means + stds,
        alpha=0.3,
        label="±1 Std Dev",
        color='blue'
    )
    plt.xlabel("Global Step")
    plt.ylabel("Episodic Return")
    plt.title("Episodic Return vs Global Step")
    plt.legend()
    plt.grid(True)

    if output_image:
        plt.savefig(output_image)
        print(f"Plot saved to {output_image}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("csv_path", type=str, help="Path to processed CSV file")
    # parser.add_argument("--output", type=str, help="Path to save the plot (optional)")
    # args = parser.parse_args()
    # plot_episodic_returns(args.csv_path, args.output)
    plot_multiple_episodic_returns(algos)