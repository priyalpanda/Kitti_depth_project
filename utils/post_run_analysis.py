import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def generate_evaluation_plots():
    # Find the 'Data' directory located one folder up from this script
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "Data"
    
    # Locate all JSON files matching the pattern
    json_pattern = str(data_dir / "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in target folder: {data_dir.resolve()}")
        return

    # Containers for data extraction
    algorithms = []
    rmse_vals = []
    fill_vals = []
    bad3m_vals = []
    depth_times = []

    for file_path in sorted(json_files):
        path_obj = Path(file_path)
        filename = path_obj.name
        
        # Strip the suffix to extract clean algorithm name
        suffix = "-evaluation-report.json"
        if filename.endswith(suffix):
            algo_name = filename[:-len(suffix)]
        else:
            algo_name = path_obj.stem  # Fallback if pattern changes
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract required values safely using .get()
            agg = data.get("aggregate", {})
            timings = data.get("timings", {})
            
            rmse = agg.get("rmse")
            fill = agg.get("fill_rate")
            bad3m = agg.get("bad_3m")
            depth_t = timings.get("depth")
            
            # Only append if all critical fields were parsed cleanly
            if None not in (rmse, fill, bad3m, depth_t):
                algorithms.append(algo_name)
                rmse_vals.append(rmse)
                fill_vals.append(fill)
                bad3m_vals.append(bad3m)
                depth_times.append(depth_t)
                
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Skipping malformed or unreadable file {filename}: {e}")

    if not algorithms:
        print("No valid evaluation statistics were successfully extracted.")
        return

    # Create one overall figure containing 4 clean subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stereo Matching Algorithm Benchmark Comparison", fontsize=12, fontweight='bold')
    
    # Flatten axes array for straightforward 1D indexing
    axes = axes.ravel()
    
    # 1. RMSE Subplot (Lower is Better)
    axes[1].bar(algorithms, rmse_vals, color='royalblue', edgecolor='black', width=0.5)
    axes[1].set_title("Root Mean Squared Error - RMSE (Lower is Better)", fontsize=12, pad=10)
    axes[1].set_ylabel("Error (meters)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Fill Rate Subplot (Higher is Better)
    axes[0].bar(algorithms, fill_vals, color='emerald' if 'emerald' in plt.colormaps else 'mediumseagreen', edgecolor='black', width=0.5)
    axes[0].set_title("Fill Rate (Higher is Better)", fontsize=12, pad=10)
    axes[0].set_ylabel("Fill Rate [0 - 1]")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Bad 3m Outliers Subplot (Lower is Better)
    axes[2].bar(algorithms, bad3m_vals, color='crimson', edgecolor='black', width=0.5)
    axes[2].set_title("Outlier Ratio (Lower is Better)", fontsize=12, pad=10)
    axes[2].set_ylabel("Outlier (more than 3m) Ratio [0 - 1]")
    axes[2].set_ylim(0, max(bad3m_vals) * 1.2 if bad3m_vals else 1.0)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Depth Runtime Subplot (Lower is Better)
    axes[3].bar(algorithms, depth_times, color='darkorange', edgecolor='black', width=0.5)
    axes[3].set_title("Execution Time (Lower is Better)", fontsize=12, pad=10)
    axes[3].set_ylabel("Time (seconds)")
    axes[3].grid(axis='y', linestyle='--', alpha=0.7)

    # Automatically rotate labels if there are many algorithms to prevent overlaps
    for ax in axes:
        ax.set_xticklabels(algorithms, rotation=0, ha='center')

    # Adjust layout dynamically to ensure text fields do not cut off
    plt.tight_layout()
    
    # Save image to active workspace and display
    output_img = data_dir / "stereo_benchmark_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated comparison figure at: {output_img.resolve()}")
    plt.show()

if __name__ == "__main__":
    generate_evaluation_plots()
