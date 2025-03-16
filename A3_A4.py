import numpy as np
import pandas as pd


def bin_numeric_data(column, num_bins=4, binning_method="equal_width"):
    if binning_method == "equal_width":
        bin_edges = np.linspace(column.min(), column.max(), num_bins + 1)
    elif binning_method == "equal_frequency":
        bin_edges = np.percentile(column, np.linspace(0, 100, num_bins + 1))
    else:
        raise ValueError("Invalid binning method. Choose 'equal_width' or 'equal_frequency'.")

    return np.digitize(column, bin_edges, right=True)


def compute_entropy(column):
    probabilities = column.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


def calculate_information_gain(dataset, feature_column, target_column):
    total_entropy = compute_entropy(dataset[target_column])
    unique_values, value_counts = np.unique(dataset[feature_column], return_counts=True)

    weighted_entropy = sum(
        (value_counts[i] / np.sum(value_counts)) * compute_entropy(dataset[dataset[feature_column] == unique_values[i]][target_column])
        for i in range(len(unique_values))
    )

    return total_entropy - weighted_entropy


def determine_root_feature(dataset, target_column):
    information_gains = {}

    for column in dataset.columns:
        if column != target_column:
            if dataset[column].dtype in ["float64", "int64"]:
                dataset[column] = bin_numeric_data(dataset[column])
            information_gains[column] = calculate_information_gain(dataset, column, target_column)

    return max(information_gains, key=information_gains.get)
