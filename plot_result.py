import json
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
import argparse

def parse_group_name(group_name):
    # Example: numba_better_unrolled-reduction-sum-dims-[3, 3, 3]-npts-1000
    match = re.match(r"(.+)-reduction-(\w+)-dims-\[(.*?)\]-npts-(\d+)", group_name)
    if not match:
        return None
    func_name, reduction, dims_str, npts = match.groups()
    dims = tuple(map(int, dims_str.split(",")))
    return func_name, reduction, dims, int(npts)

def load_results_from_folder(folder_path):
    results = defaultdict(lambda: defaultdict(list))  # results[(reduction, dims)][func_name] = [(num_points, mean, stddev)]
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder '{folder_path}' does not exist or is not a directory.")

    files = list(folder.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in '{folder_path}'.")

    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {file.name}: JSON load error: {e}")
            continue

        for b in data.get("benchmarks", []):
            group = b.get("group", "")
            parsed = parse_group_name(group)
            if parsed is None:
                continue
            func_name, reduction, dims, num_points = parsed
            stats = b.get("stats", {})
            mean = stats.get("mean")
            stddev = stats.get("stddev")
            if mean is None or stddev is None:
                continue
            results[(reduction, dims)][func_name].append((num_points, mean, stddev))

    return results

def plot_results(results, output_dir="plots"):
    Path(output_dir).mkdir(exist_ok=True)

    for (reduction, dims), func_results in results.items():
        plt.figure()

        for func_name, entries in func_results.items():
            entries.sort()
            x = [e[0] for e in entries]
            y = [e[1] for e in entries]
            yerr = [e[2] for e in entries]

            plt.errorbar(x, y, yerr=yerr, fmt="o-", label=func_name, capsize=3, alpha=0.7)

        plt.title(f"Reduction: {reduction}, Dims: {dims}")
        plt.xlabel("Number of Points")
        plt.ylabel("Mean Time (s)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = f"{reduction}_{'_'.join(map(str, dims))}.png"
        plt.savefig(
            Path(output_dir) / filename,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing benchmark JSON files")
    parser.add_argument("--output", default="plots", help="Output folder for plots")
    parser.add_argument(
        "--use_paper_style",
        action="store_true",
        help="Use the paper style (default: False)",
    )
    args = parser.parse_args()

    if args.use_paper_style:
        plt.style.use("./paper_2.mplstyle")

    try:
        results = load_results_from_folder(args.folder)
        if not results:
            print("No valid benchmark entries found.")
        else:
            plot_results(results, output_dir=args.output)
            print(f"Saved plots to '{args.output}'")
    except Exception as e:
        print(f"Error: {e}")
