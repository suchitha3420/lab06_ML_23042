import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def bin_data(series, num_bins=4, binning_type="equal_width"):
    if binning_type == "equal_width":
        bins = np.linspace(series.min(), series.max(), num_bins + 1)
    elif binning_type == "equal_frequency":
        bins = np.percentile(series, np.linspace(0, 100, num_bins + 1))
    else:
        raise ValueError("Invalid binning type. Choose 'equal_width' or 'equal_frequency'.")
    return np.digitize(series, bins, right=True)

class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
        self.feature_names = []
        self.target_name = None

    def preprocess_data(self, df, target):
        self.feature_names = [col for col in df.columns if col != target]
        self.target_name = target

        X = df[self.feature_names].copy()
        y = df[target].copy()

        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        else:
            le_target = None

        return X, y, label_encoders, le_target

    def train(self, df, target):
        X, y, _, _ = self.preprocess_data(df, target)
        self.model.fit(X, y)

    def visualize_tree(self, output_file="decision_tree.png"):
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=self.feature_names, filled=True, rounded=True, class_names=True)
        plt.savefig(output_file)
        plt.show()
