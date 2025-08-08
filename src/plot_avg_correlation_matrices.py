import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as joinpath
from scipy.stats import kurtosis, skew
import json
import argparse
import matplotlib.ticker as mticker
from scipy.ndimage import zoom


def downsample_matrix(matrix, max_size=500):
    """
    Downsample a correlation matrix to a maximum size while preserving structure.

    Args:
        matrix (np.ndarray): Input correlation matrix
        max_size (int): Maximum dimension for the downsampled matrix

    Returns:
        np.ndarray: Downsampled matrix
    """
    if matrix.shape[0] <= max_size and matrix.shape[1] <= max_size:
        return matrix

    # Calculate zoom factor
    zoom_factor = min(max_size / matrix.shape[0], max_size / matrix.shape[1])

    # Downsample using scipy.ndimage.zoom with cubic interpolation
    downsampled = zoom(matrix, zoom_factor, order=3, mode="nearest")

    return downsampled


def plot_correlation_matrices(
    corr_results_dir, save_dir, num_last_layers_to_plot, max_matrix_size=500
):
    """
    Plots correlation matrices and their distributions, saving results into specified directories.

    Args:
        corr_results_dir (str): Path to the input directory containing NPY correlation matrices.
        save_dir (str): Path to the directory where plots and statistics will be saved.
        num_last_layers_to_plot (int): The number of last layers to plot.
        max_matrix_size (int): Maximum size for matrix visualization (default: 500)
    """

    os.makedirs(save_dir, exist_ok=True)

    files_list = os.listdir(corr_results_dir)
    layer_names_raw = [
        f.split("_")[0] + "_" + f.split("_")[1]
        for f in files_list
        if f.endswith(".npy")
    ]
    layer_names_unique_sorted = sorted(
        list(dict.fromkeys(layer_names_raw)),
        key=lambda x: int(x.split("_")[1]),
    )

    layers_to_plot = layer_names_unique_sorted[-num_last_layers_to_plot:]

    if not layers_to_plot:
        print(
            "No layers to plot based on the specified number of last layers."
        )
        return

    # --- Global Font Size Increase ---
    plt.rcParams.update({"font.size": 16})
    sns.set_context("talk")

    for i, layer_name in enumerate(layers_to_plot):
        layer_save_dir = joinpath(save_dir, layer_name)
        os.makedirs(layer_save_dir, exist_ok=True)

        correct_corr_path = joinpath(
            corr_results_dir, f"{layer_name}_correct_avg_correlation.npy"
        )
        incorrect_corr_path = joinpath(
            corr_results_dir, f"{layer_name}_incorrect_avg_correlation.npy"
        )

        if not os.path.exists(correct_corr_path) or not os.path.exists(
            incorrect_corr_path
        ):
            print(f"Skipping {layer_name}: Correlation files not found.")
            continue

        correct_corr = np.load(correct_corr_path)
        incorrect_corr = np.load(incorrect_corr_path)

        # Store original matrices for statistical analysis
        correct_flat = correct_corr.flatten()
        incorrect_flat = incorrect_corr.flatten()

        # Downsample matrices for visualization only
        print(f"Original matrix size: {correct_corr.shape}")
        correct_corr_vis = downsample_matrix(correct_corr, max_matrix_size)
        incorrect_corr_vis = downsample_matrix(incorrect_corr, max_matrix_size)
        print(f"Visualization matrix size: {correct_corr_vis.shape}")

        # --- Statistical Analysis (using original full-resolution data) ---
        percentile_threshold = 99
        correct_99th_percentile = np.percentile(
            correct_flat, percentile_threshold
        )
        incorrect_99th_percentile = np.percentile(
            incorrect_flat, percentile_threshold
        )

        fixed_high_threshold = 0.9
        num_correct_gt_fixed = np.sum(correct_flat > fixed_high_threshold)
        num_incorrect_gt_fixed = np.sum(incorrect_flat > fixed_high_threshold)

        kurt_correct = kurtosis(correct_flat)
        kurt_incorrect = kurtosis(incorrect_flat)

        skew_correct = skew(correct_flat)
        skew_incorrect = skew(incorrect_flat)

        # Store statistics
        stats_dict = {
            "original_matrix_shape": correct_corr.shape,
            "visualization_matrix_shape": correct_corr_vis.shape,
            "correct": {
                "kurtosis": kurt_correct,
                "skewness": skew_correct,
                f"percentile_{percentile_threshold}": correct_99th_percentile,
                f"count_gt_{fixed_high_threshold}": int(num_correct_gt_fixed),
            },
            "incorrect": {
                "kurtosis": kurt_incorrect,
                "skewness": skew_incorrect,
                f"percentile_{percentile_threshold}": incorrect_99th_percentile,
                f"count_gt_{fixed_high_threshold}": int(
                    num_incorrect_gt_fixed
                ),
            },
        }

        # Save statistics to JSON
        stats_filepath = joinpath(
            layer_save_dir, f"{layer_name}_distribution_statistics.json"
        )
        with open(stats_filepath, "w") as f:
            json.dump(stats_dict, f, indent=4)
        print(f"  Statistics saved to {stats_filepath}")

        # --- Plotting Correlation Matrices (using downsampled data) ---
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        cbar_ax = fig.add_subplot(gs[0, 2])

        # Plot using downsampled matrices
        sns.heatmap(
            correct_corr_vis,
            cmap="viridis",
            cbar=False,
            vmin=-1,
            vmax=1,
            ax=ax1,
            rasterized=True,  # Rasterize to reduce file size
        )
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_ylabel("")
        ax1.set_xlabel("")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis="both", which="both", length=0)

        heatmap_plot = sns.heatmap(
            incorrect_corr_vis,
            cmap="viridis",
            cbar=True,
            vmin=-1,
            vmax=1,
            ax=ax2,
            cbar_ax=cbar_ax,
            cbar_kws={"label": "Correlation Coefficient"},
            rasterized=True,  # Rasterize to reduce file size
        )
        cbar = heatmap_plot.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Correlation Coefficient", fontsize=20)

        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis="both", which="both", length=0)

        # Save with optimized settings
        corr_matrices_plot_path = joinpath(
            layer_save_dir, f"{layer_name}_correlation_matrices.pdf"
        )
        plt.tight_layout()
        plt.savefig(
            corr_matrices_plot_path,
            dpi=400,  # Reduced from 600 to 300 (still high quality)
            bbox_inches="tight",
        )
        plt.close(fig)
        print(
            f"  Correlation matrices plot saved to {corr_matrices_plot_path}"
        )

        # --- Plotting Distributions (using original full data) ---
        fontsize_annotations = 16
        plt.figure(figsize=(18, 8))

        # Distribution plots remain the same since they're not the bottleneck
        ax_dist1 = plt.subplot(1, 2, 1)
        sns.histplot(
            correct_flat, kde=True, color="skyblue", bins=50, ax=ax_dist1
        )
        ax_dist1.set_xlabel("Correlation Value", fontsize=18)
        ax_dist1.set_ylabel("Density", fontsize=18)
        ax_dist1.tick_params(axis="x", labelsize=16)
        ax_dist1.tick_params(axis="y", labelsize=16)
        ax_dist1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        y_coord = 0.7
        ax_dist1.text(
            y_coord,
            0.95,
            f"Kurtosis: {kurt_correct:.2f}",
            transform=ax_dist1.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )
        ax_dist1.text(
            y_coord,
            0.90,
            f"Skewness: {skew_correct:.2f}",
            transform=ax_dist1.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )
        ax_dist1.text(
            y_coord,
            0.85,
            f"99th Pctile: {correct_99th_percentile:.2f}",
            transform=ax_dist1.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )
        ax_dist1.text(
            y_coord,
            0.80,
            f"Values > {fixed_high_threshold}: {num_correct_gt_fixed}",
            transform=ax_dist1.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )

        ax_dist2 = plt.subplot(1, 2, 2)
        sns.histplot(
            incorrect_flat, kde=True, color="salmon", bins=50, ax=ax_dist2
        )
        ax_dist2.set_xlabel("Correlation Value", fontsize=18)
        ax_dist2.set_ylabel("")
        ax_dist2.tick_params(axis="x", labelsize=16)
        ax_dist2.tick_params(axis="y", labelsize=16)
        ax_dist2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        ax_dist2.text(
            y_coord,
            0.95,
            f"Kurtosis: {kurt_incorrect:.2f}",
            transform=ax_dist2.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )
        ax_dist2.text(
            y_coord,
            0.90,
            f"Skewness: {skew_incorrect:.2f}",
            transform=ax_dist2.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )
        ax_dist2.text(
            y_coord,
            0.85,
            f"99th Pctile: {incorrect_99th_percentile:.2f}",
            transform=ax_dist2.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )
        ax_dist2.text(
            y_coord,
            0.80,
            f"Values > {fixed_high_threshold}: {num_incorrect_gt_fixed}",
            transform=ax_dist2.transAxes,
            fontsize=fontsize_annotations,
            verticalalignment="top",
        )

        distribution_plot_path = joinpath(
            layer_save_dir, f"{layer_name}_correlation_distributions.pdf"
        )
        plt.tight_layout()
        plt.savefig(
            distribution_plot_path,
            dpi=400,  # Reduced from 400 to 300
            bbox_inches="tight",
        )
        plt.close()
        print(f"  Distribution plots saved to {distribution_plot_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot correlation matrices and their distributions with statistics."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing NPY correlation matrices.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to the directory where plots and statistics will be saved.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="The number of last layers to plot.",
    )
    parser.add_argument(
        "--max-matrix-size",
        type=int,
        default=500,
        help="Maximum size for matrix visualization (default: 500)",
    )

    args = parser.parse_args()

    plot_correlation_matrices(
        args.input_dir, args.save_dir, args.num_layers, args.max_matrix_size
    )
