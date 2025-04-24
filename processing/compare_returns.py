import pandas as pd
import argparse

def get_algo_paths(env):
    return {
        "SAC": f"data/sac_{env}_aligned.csv",
        "SAC-POV": f"data/sac_pov_{env}_aligned.csv",
        "SAC-D": f"data/sacd_{env}_aligned.csv",
        "SAC-D5": f"data/sacd5_{env}_aligned.csv",
        "SAC-3Q": f"data/sac3q_{env}_aligned.csv",
        "SAC-5Q": f"data/sac5q_{env}_aligned.csv",
        "SAC-10Q": f"data/sac10q_{env}_aligned.csv",
        "PPO": f"data/ppo_{env}_aligned.csv",
        "TD3": f"data/td3_{env}_aligned.csv",
    }

# def get_returns(algo_paths):
#     returns = {}
#     for label, path in algo_paths.items():
#         try:
#             df = pd.read_csv(path)
#             curr_return = {
#                 "mean": df["episodic_return_mean"].values[-1],
#                 "std": df["episodic_return_std"].values[-1]
#             }
#             returns[label] = curr_return
#         except FileNotFoundError:
#             print(f"Warning: File not found for {label} at {path}")
#     return returns

def get_returns(algo_paths, tail=100):
    returns = {}
    for label, path in algo_paths.items():
        try:
            df = pd.read_csv(path)

            # Only consider the last 100 steps
            df_last = df.tail(tail)

            curr_return = {
                "mean": df_last["episodic_return_mean"].mean(),
                "std": df_last["episodic_return_mean"].std()  # std over the means
            }
            returns[label] = curr_return
        except FileNotFoundError:
            print(f"Warning: File not found for {label} at {path}")
    return returns

def summarize_final_rewards(envs, tail):
    for env in envs:
        print(f"\nEnvironment: {env}")
        algo_paths = get_algo_paths(env)
        returns = get_returns(algo_paths, tail)
        if not returns:
            print("No data available.")
            continue

        print(f"{'Algorithm':<10} | {'Mean':>10} | {'Std':>10}")
        print("-" * 35)
        for algo, result in returns.items():
            print(f"{algo:<10} | {result['mean']:10.2f} | {result['std']:10.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs", nargs="+", required=True,
        help="List of environment names (e.g., cheetah hopper)"
    )
    parser.add_argument(
        "--tail", type=int, default=100,
        help="Number of last steps to consider for mean and std calculation"
    )
    args = parser.parse_args()

    summarize_final_rewards(args.envs, args.tail)
