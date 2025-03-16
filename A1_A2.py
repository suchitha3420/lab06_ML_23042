import numpy as np
import pandas as pd

def bin_numeric_data(data_column, bins_count=4):
    bin_edges = np.linspace(data_column.min(), data_column.max(), bins_count + 1)
    return np.digitize(data_column, bin_edges, right=True)

def calculate_entropy(data_column):
    probabilities = data_column.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))

def compute_gini_index(data_column):
    probabilities = data_column.value_counts(normalize=True)
    return 1 - np.sum(probabilities ** 2)