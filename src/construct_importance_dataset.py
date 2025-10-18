import argparse
import glob
import os
from sympy import arg
from tqdm import tqdm
import cupy as cp
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from copy import deepcopy

from argparse import ArgumentParser

from typing import List, Any


FIXED_HIGH_THRESHOLD_4TH_PERCENTILE = 0.75
FIXED_HIGH_THRESHOLD_LAST_DECILE = 0.9
FIXED_LOW_THRESHOLD = -0.9
FIXED_LOW_THRESHOLD_4TH_PERCENTILE = -0.75
MEAN_MEDIAN_SCALING_FACTOR = 1e3  # make median and mean more readable


class ScriptArgs(argparse.Namespace):
    input_dir: str
    save_dir: str


class MatrixStats:
    """
    A data class to hold various statistical measurements of a matrix.

    Casts specific fields (mean, median, kurt, skewness, p99_value) to float,
    and count fields (all others) to int in the constructor.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the stats fields and performs mandatory type casting.
        """
        self.mean: float = float(kwargs.get("mean", 0.0))
        self.std_dev: float = float(kwargs.get("std_dev", 0.0))
        self.median: float = float(kwargs.get("median", 0.0))
        self.kurt: float = float(kwargs.get("kurt", 0.0))
        self.skewness: float = float(kwargs.get("skewness", 0.0))
        self.p99_value: float = float(kwargs.get("p99_value", 0.0))

        self.vals_over_high_positive_threshold: int = int(
            kwargs.get("vals_over_high_positive_threshold", 0)
        )
        self.vals_over_high_positive_threshold_4th_percentile: int = int(
            kwargs.get("vals_over_high_positive_threshold_4th_percentile", 0)
        )
        self.vals_below_low_negative_threshold: int = int(
            kwargs.get("vals_below_low_negative_threshold", 0)
        )
        self.vals_below_low_negative_threshold_4th_percentile: int = int(
            kwargs.get("vals_below_low_negative_threshold_4th_percentile", 0)
        )
        self.values_above_p99: int = int(kwargs.get("values_above_p99", 0))

    def __repr__(self) -> str:
        """Provides a clean string representation for printing."""
        return (
            f"MatrixStats(\n"
            f"  Mean: {self.mean:.4f},\n"
            f"  Std Dev: {self.std_dev:.4f},\n"
            f"  Median: {self.median:.4f},\n"
            f"  Kurtosis: {self.kurt:.4f},\n"
            f"  Skewness: {self.skewness:.4f},\n"
            f"  P99 Value: {self.p99_value:.4f},\n"
            f"  --- Counts ---\n"
            f"  Over High Threshold (Decile): {self.vals_over_high_positive_threshold},\n"
            f"  Over High Threshold (4th Pct): {self.vals_over_high_positive_threshold_4th_percentile},\n"
            f"  Below Low Threshold: {self.vals_below_low_negative_threshold},\n"
            f"  Below Low Threshold (4th Pct): {self.vals_below_low_negative_threshold_4th_percentile},\n"
            f"  Values Above P99: {self.values_above_p99}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Converts the stats to a dictionary."""
        return {
            "mean": self.mean,
            "std_dev": self.std_dev,
            "median": self.median,
            "kurt": self.kurt,
            "skewness": self.skewness,
            "p99_value": self.p99_value,
            "vals_over_high_positive_threshold": self.vals_over_high_positive_threshold,
            "vals_over_high_positive_threshold_4th_percentile": self.vals_over_high_positive_threshold_4th_percentile,
            "vals_below_low_negative_threshold": self.vals_below_low_negative_threshold,
            "vals_below_low_negative_threshold_4th_percentile": self.vals_below_low_negative_threshold_4th_percentile,
            "values_above_p99": self.values_above_p99,
        }

    def get_keys(self) -> List[str]:
        """Returns the list of statistic keys."""
        return list(self.to_dict().keys())


def prepare_paths_list_npy_files(
    root_dir: str, search_pattern: str = "*spike_layer_2.npy"
):
    """
    Given root dir, recurse into it and find paths to all files matching the pattern
    """
    search_path_pattern = os.path.join(root_dir, "**", search_pattern)
    found_npy_files_paths: List[str] = glob.glob(
        search_path_pattern, recursive=True
    )
    assert found_npy_files_paths, "No results found for given pattern!"

    return found_npy_files_paths


def compute_correlation_from_activity_matrix(activity_matrix: np.ndarray):

    activity_matrix = activity_matrix.reshape(activity_matrix.shape[0], -1)
    corr_matrix: cp.ndarray = cp.corrcoef(activity_matrix.T)
    corr_matrix[cp.isnan(corr_matrix)] = 0
    return cp.asnumpy(corr_matrix)


def compute_matrix_statistics(input_matrix: np.ndarray):
    """
    Calculate stats from input matrix.
    """
    flattened_matrix = input_matrix.flatten()
    mean = np.mean(flattened_matrix) * MEAN_MEDIAN_SCALING_FACTOR
    median = np.median(flattened_matrix) * MEAN_MEDIAN_SCALING_FACTOR
    std_dev = np.std(flattened_matrix) * MEAN_MEDIAN_SCALING_FACTOR
    kurt = kurtosis(flattened_matrix)
    skewness = skew(flattened_matrix)
    p99_value = np.percentile(flattened_matrix, 99)
    vals_over_high_positive_threshold = np.sum(
        flattened_matrix > FIXED_HIGH_THRESHOLD_LAST_DECILE
    )
    vals_over_high_positive_threshold_4th_percentile = np.sum(
        flattened_matrix > FIXED_HIGH_THRESHOLD_4TH_PERCENTILE
    )
    vals_below_low_negative_threshold = np.sum(
        flattened_matrix < FIXED_LOW_THRESHOLD
    )
    vals_below_low_negative_threshold_4th_percentile = np.sum(
        flattened_matrix < FIXED_LOW_THRESHOLD_4TH_PERCENTILE
    )
    values_above_p99 = np.sum(
        flattened_matrix > np.percentile(flattened_matrix, 99)
    )
    return MatrixStats(
        mean=mean,
        std_dev=std_dev,
        median=median,
        kurt=kurt,
        skewness=skewness,
        p99_value=p99_value,
        vals_over_high_positive_threshold=vals_over_high_positive_threshold,
        vals_over_high_positive_threshold_4th_percentile=vals_over_high_positive_threshold_4th_percentile,
        vals_below_low_negative_threshold=vals_below_low_negative_threshold,
        vals_below_low_negative_threshold_4th_percentile=vals_below_low_negative_threshold_4th_percentile,
        values_above_p99=values_above_p99,
    )


def create_features_for_activity_matrix(activty_matrix: np.ndarray):
    """Calculate correlation matrix of neuronal activity and its statistics."""
    correlation_matrix = compute_correlation_from_activity_matrix(
        activty_matrix
    )
    correlation_stats = compute_matrix_statistics(correlation_matrix)
    return correlation_stats


def initialize_dataframe(data: MatrixStats) -> pd.DataFrame:

    return pd.DataFrame(data.to_dict(), index=[0])


def extract_exp_name(file_path: str):
    path_elements = file_path.split("/")
    for elem in path_elements:
        if "results_" in elem:
            return elem
    assert False, f"Could not find experiment name in path {file_path}"


def main(args: ScriptArgs):

    npy_files_to_analyze = prepare_paths_list_npy_files(args.input_dir)
    correct_paths = list(
        filter(lambda p: "_original" in p, npy_files_to_analyze)
    )

    incorrect_paths = list(
        filter(lambda p: "incorrect_" in p, npy_files_to_analyze)
    )

    row_dfs_correct = []
    row_dfs_incorrect = []

    for correct_path in tqdm(
        correct_paths,
        total=len(correct_paths),
        desc="Processing correct files",
    ):
        activity_matrix = np.load(correct_path)
        stats = create_features_for_activity_matrix(activity_matrix)
        row_dfs_correct.append(initialize_dataframe(stats))

    for incorrect_path in tqdm(
        incorrect_paths,
        total=len(incorrect_paths),
        desc="Processing incorrect files",
    ):
        activity_matrix = np.load(incorrect_path)
        stats = create_features_for_activity_matrix(activity_matrix)
        row_dfs_incorrect.append(initialize_dataframe(stats))

    df_correct = pd.concat(row_dfs_correct, ignore_index=True)
    df_incorrect = pd.concat(row_dfs_incorrect, ignore_index=True)

    correct_path = os.path.join(
        args.save_dir,
        f"{extract_exp_name(args.input_dir)}/correct_stats.csv",
    )
    incorrect_path = os.path.join(
        args.save_dir,
        f"{extract_exp_name(args.input_dir)}/incorrect_stats.csv",
    )
    os.makedirs(os.path.dirname(correct_path), exist_ok=True)
    os.makedirs(os.path.dirname(incorrect_path), exist_ok=True)

    df_correct.to_csv(
        correct_path,
        index=False,
    )
    df_incorrect.to_csv(
        incorrect_path,
        index=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Root results directory"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save results"
    )
    args_namespace = ScriptArgs()
    args = parser.parse_args(namespace=args_namespace)
    main(args)
