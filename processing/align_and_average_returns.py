
import pandas as pd
import numpy as np
import argparse

def process_returns(csv_path, output_path):
    df = pd.read_csv(csv_path)
    global_step_col = df.columns[0]
    episodic_return_cols = [col for col in df.columns if "charts/episodic_return" in col and "__MIN" not in col and "__MAX" not in col]

    run_dfs = []
    for col in episodic_return_cols:
        run_df = df[[global_step_col, col]].dropna()
        run_df.columns = ["global_step", "episodic_return"]
        run_df = run_df.astype({"global_step": int})
        run_dfs.append(run_df)

    all_steps = sorted(set().union(*[set(run_df["global_step"]) for run_df in run_dfs]))
    all_steps = np.array(all_steps)

    aligned_returns = []
    for run_df in run_dfs:
        interpolated = run_df.set_index("global_step").reindex(all_steps).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        aligned_returns.append(interpolated["episodic_return"].values)

    returns_matrix = np.stack(aligned_returns, axis=0)
    mean_returns = np.mean(returns_matrix, axis=0)
    std_returns = np.std(returns_matrix, axis=0)

    result_df = pd.DataFrame({
        "global_step": all_steps,
        "episodic_return_mean": mean_returns,
        "episodic_return_std": std_returns
    })
    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to input CSV file without .csv extension")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.csv_path + "_aligned.csv"
    else:
        if not args.output.endswith(".csv"):
            args.output += ".csv"
    if not args.csv_path.endswith(".csv"):
        args.csv_path += ".csv"
    process_returns(args.csv_path, args.output)