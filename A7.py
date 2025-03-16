import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


def load_dataset(file_path, feature_x, feature_y, target_label):
    dataset = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    dataset[target_label] = label_encoder.fit_transform(dataset[target_label])
    feature_data = dataset[[feature_x, feature_y]].values
    target_data = dataset[target_label].values
    return feature_data, target_data


def split_dataset(feature_data, target_data, test_ratio=0.2, seed=42):
    return train_test_split(feature_data, target_data, test_size=test_ratio, random_state=seed)


def train_decision_tree_classifier(X_train, y_train, tree_depth=5):
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=tree_depth)
    decision_tree.fit(X_train, y_train)
    return decision_tree


def visualize_decision_boundary(X, y, classifier, feature_x, feature_y):
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X, y, clf=classifier, legend=2)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title("Decision Boundary of Decision Tree Classifier")
    plt.show()


