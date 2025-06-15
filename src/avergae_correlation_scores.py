import os
import numpy as np
from os.path import join as joinpath
import argparse
from tqdm import tqdm

FIRST_DIM_TARGET_SIZE = 10


def compute_correlation_from_activity_matrix(activity_matrix):
    if activity_matrix.shape[0] != FIRST_DIM_TARGET_SIZE:
        activity_matrix = activity_matrix[0]

    activity_matrix = activity_matrix.reshape(activity_matrix.shape[0], -1)

    corr_matrix = np.corrcoef(activity_matrix.T)
    corr_matrix[np.isnan(corr_matrix)] = 0
    return corr_matrix


def compute_average_layer_correlation_scores(
    root_data_path: str, save_dir: str = None
):
    layer_correlations_correct = {}
    layer_correlations_incorrect = {}

    layer_correct_sample_counts = {}
    layer_incorrect_sample_counts = {}

    total_samples_processed = 0

    for label_dir in os.listdir(root_data_path):
        # label level
        print(label_dir)
        label_dir_path = joinpath(root_data_path, label_dir)
        if not os.path.isdir(label_dir_path):
            continue

        if label_dir == "label_3":
            # skip label 3, as full noise samples result in predicting label 3
            continue

        samples_in_current_label_dir = 0
        progbar = tqdm(
            os.listdir(label_dir_path),
            desc=f"Processing label {label_dir}",
            total=len(os.listdir(label_dir_path)),
            unit="samples",
        )
        for sample_dir in progbar:
            if samples_in_current_label_dir >= 2:
                break
            sample_dir_path = joinpath(label_dir_path, sample_dir)
            if not os.path.isdir(sample_dir_path):
                continue

            samples_in_current_label_dir += 1
            total_samples_processed += 1

            original_activities_dir = joinpath(
                sample_dir_path, "correct_original"
            )

            incorrect_noise_activities_dirs = [
                joinpath(sample_dir_path, f)
                for f in os.listdir(sample_dir_path)
                if "incorrect" in f and "noise" in f
            ]

            if not incorrect_noise_activities_dirs:
                continue
            incorrect_noise_activities_dir = incorrect_noise_activities_dirs[0]

            correct_layer_activity_files = [
                joinpath(original_activities_dir, f)
                for f in os.listdir(original_activities_dir)
                if "spike" in f
            ]
            for layer_activity_file in correct_layer_activity_files:
                parts = os.path.basename(layer_activity_file).split("_")
                if len(parts) >= 2:
                    layer_num_str = parts[-1].split(".")[0]
                    layer_name_prefix = parts[-2]
                    layer_name = f"{layer_name_prefix}_{layer_num_str}"
                else:
                    continue

                corr_matrix = compute_correlation_from_activity_matrix(
                    np.load(layer_activity_file)
                )
                if layer_name not in layer_correlations_correct:
                    layer_correlations_correct[layer_name] = np.zeros_like(
                        corr_matrix
                    )
                    layer_correct_sample_counts[layer_name] = 0
                layer_correlations_correct[layer_name] += corr_matrix
                layer_correct_sample_counts[layer_name] += 1

            incorrect_layer_activity_files = [
                joinpath(incorrect_noise_activities_dir, f)
                for f in os.listdir(incorrect_noise_activities_dir)
                if "spike" in f
            ]
            for layer_activity_file in incorrect_layer_activity_files:
                parts = os.path.basename(layer_activity_file).split("_")
                if len(parts) >= 2:
                    layer_num_str = parts[-1].split(".")[0]
                    layer_name_prefix = parts[-2]
                    layer_name = f"{layer_name_prefix}_{layer_num_str}"
                else:
                    continue

                corr_matrix = compute_correlation_from_activity_matrix(
                    np.load(layer_activity_file)
                )
                if layer_name not in layer_correlations_incorrect:
                    layer_correlations_incorrect[layer_name] = np.zeros_like(
                        corr_matrix
                    )
                    layer_incorrect_sample_counts[layer_name] = 0
                layer_correlations_incorrect[layer_name] += corr_matrix
                layer_incorrect_sample_counts[layer_name] += 1

    for layer_name in layer_correlations_correct:
        if layer_correct_sample_counts[layer_name] > 0:
            layer_correlations_correct[
                layer_name
            ] /= layer_correct_sample_counts[layer_name]

    for layer_name in layer_correlations_incorrect:
        if layer_incorrect_sample_counts[layer_name] > 0:
            layer_correlations_incorrect[
                layer_name
            ] /= layer_incorrect_sample_counts[layer_name]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for layer_name, corr_matrix in layer_correlations_correct.items():
            np.save(
                joinpath(
                    save_dir, f"{layer_name}_correct_avg_correlation.npy"
                ),
                corr_matrix,
            )
        for layer_name, corr_matrix in layer_correlations_incorrect.items():
            np.save(
                joinpath(
                    save_dir, f"{layer_name}_incorrect_avg_correlation.npy"
                ),
                corr_matrix,
            )

    return layer_correlations_correct, layer_correlations_incorrect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average layer correlation scores."
    )
    parser.add_argument(
        "--root-data-path",
        type=str,
        required=True,
        help="Root directory containing the data.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the averaged correlation matrices. If not specified, results are not saved.",
    )

    args = parser.parse_args()

    avg_correct, avg_incorrect = compute_average_layer_correlation_scores(
        args.root_data_path, args.save_dir
    )

    print("\nComputation complete.")
    if args.save_dir:
        print(f"Average correlation matrices saved to: {args.save_dir}")
    else:
        print(
            "Average correlation matrices were not saved as --save-dir was not provided."
        )
