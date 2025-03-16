import pandas as pd
from A1_A2 import bin_numeric_data, calculate_entropy,compute_gini_index
from A3_A4 import determine_root_feature
from A5_A6 import DecisionTreeModel
from A7 import load_dataset,split_dataset,train_decision_tree_classifier,visualize_decision_boundary


def main():
    file_path = "DCT_withoutduplicate 6 1 1 1.csv"
    df = pd.read_csv(file_path)

    while True:
        print("\nMenu:")
        print("1. A1,A2")
        print("2. A3,A4")
        print("3. A4,A5")
        print("4. A7")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            if df["LABEL"].nunique() > 10:
                df["Binned_LABEL"] = bin_numeric_data(df["LABEL"])
            else:
                df["Binned_LABEL"] = df["LABEL"]

            dataset_entropy = calculate_entropy(df["Binned_LABEL"])
            dataset_gini = compute_gini_index(df["Binned_LABEL"])

            print("Entropy:", dataset_entropy)
            print("Gini Index:", dataset_gini)

        elif choice == "2":
            target_column = "LABEL"
            if "Binned_LABEL" in df.columns:
                df = df.drop(columns=["Binned_LABEL"])

            best_feature = determine_root_feature(df, target_column)
            print(f"Best Root Node Feature: {best_feature}")

        elif choice == "3":
            target_column = "LABEL"
            dt = DecisionTreeModel(max_depth=3)
            dt.train(df, target_column)

            print("Decision Tree Visualization:")
            dt.visualize_tree()

        elif choice == "4":
            feature1 = "7"
            feature2 = "10"
            target = "LABEL"

            X, y = load_dataset(file_path, feature1, feature2, target)
            X_train, X_test, y_train, y_test = split_dataset(X, y)

            dt_model = train_decision_tree_classifier(X_train, y_train)
            visualize_decision_boundary(X, y, dt_model, feature1, feature2)

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
