from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
import csv


def get_data_from_csv(filename="normalized_data.csv"):
    with open(filename, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter='\n')
        headers = next(reader)
        X = []
        Y = []
        for row in reader:
            row = row[0].split(',')
            X.append(row[:4] + row[5:])
            Y.append(1 if row[4] == 'yes' else 0)
        return np.array(X[:1000], dtype=float), np.array(Y[:1000], dtype=int)


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if (depth == self.max_depth or
                num_samples < self.min_samples_split or
                len(set(y)) == 1 or
                num_samples <= self.min_samples_leaf):
            return Counter(y).most_common(1)[0][0]

        best_split = self._find_best_split(X, y, num_features)
        if best_split is None:
            return Counter(y).most_common(1)[0][0]

        left_tree = self._build_tree(best_split["left_X"], best_split["left_y"], depth + 1)
        right_tree = self._build_tree(best_split["right_X"], best_split["right_y"], depth + 1)

        return {
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def _find_best_split(self, X, y, num_features):
        best_split = {"information_gain": -1}
        current_entropy = self._entropy(y)

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
                    continue

                left_y, right_y = y[left_indices], y[right_indices]
                entropy_left, entropy_right = self._entropy(left_y), self._entropy(right_y)

                weighted_entropy = (len(left_y) * entropy_left + len(right_y) * entropy_right) / len(y)
                information_gain = current_entropy - weighted_entropy

                if information_gain > best_split["information_gain"]:
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "information_gain": information_gain,
                        "left_X": X[left_indices],
                        "right_X": X[right_indices],
                        "left_y": left_y,
                        "right_y": right_y,
                    }

        return best_split if best_split["information_gain"] > 0 else None

    @staticmethod
    def _entropy(y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature_value = x[tree["feature_index"]]
        if feature_value <= tree["threshold"]:
            return self._traverse_tree(x, tree["left"])
        else:
            return self._traverse_tree(x, tree["right"])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, min_samples_leaf=5, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)


############# ГРАФИКИ #############

# 1


def calculate_bibl_tree_heights(param_name, values):
    tree_heights = []
    for value in values:
        params = {param_name: value}
        tree = DecisionTreeClassifier(random_state=42, **params)
        tree.fit(X_train, y_train)
        tree_heights.append(tree.get_depth())
    return tree_heights


def first_plot():
    param_values = {
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": [None, 5, 3, 1],
    }
    plt.figure(figsize=(10, 6))
    for param_name, values in param_values.items():
        heights = calculate_bibl_tree_heights(param_name, values)
        plt.plot(values, heights, marker="o", label=f"{param_name}")

    plt.xlabel("Значение гиперпараметра")
    plt.ylabel("Высота дерева")
    plt.title("Зависимость высоты дерева от гиперпараметров")
    plt.legend()
    plt.grid()
    plt.show()

# 2


def calculate_tree_height(tree):
    if not isinstance(tree, dict):
        return 0
    left_height = calculate_tree_height(tree["left"])
    right_height = calculate_tree_height(tree["right"])
    return 1 + max(left_height, right_height)


def second_plot():
    min_samples_split_values = [2, 5, 10, 20, 50]
    min_samples_leaf_values = [1, 2, 5, 10, 20]
    max_depth_values = [None, 5, 10, 15, 20]

    heights_min_samples_split = []
    heights_min_samples_leaf = []
    heights_max_depth = []

    for value in min_samples_split_values:
        tree = DecisionTree(max_depth=None, min_samples_split=value, min_samples_leaf=5)
        tree.fit(X_train, y_train)
        height = calculate_tree_height(tree.tree)
        heights_min_samples_split.append(height)

    for value in min_samples_leaf_values:
        tree = DecisionTree(max_depth=None, min_samples_split=2, min_samples_leaf=value)
        tree.fit(X_train, y_train)
        height = calculate_tree_height(tree.tree)
        heights_min_samples_leaf.append(height)

    for value in max_depth_values:
        tree = DecisionTree(max_depth=value, min_samples_split=2, min_samples_leaf=5)
        tree.fit(X_train, y_train)
        height = calculate_tree_height(tree.tree)
        heights_max_depth.append(height)

    plt.figure(figsize=(10, 6))
    plt.plot(min_samples_split_values, heights_min_samples_split, label='min_samples_split', marker='o')
    plt.plot(min_samples_leaf_values, heights_min_samples_leaf, label='min_samples_leaf', marker='o')
    plt.plot(max_depth_values[:-1], heights_max_depth[1:], label='max_depth', marker='o')

    plt.xlabel("Значение гиперпараметра")
    plt.ylabel("Высота дерева")
    plt.title("Зависимость высоты дерева от гиперпараметров")
    plt.legend()
    plt.grid()
    plt.show()


# 3


def third_plot():
    max_depth_values = range(2, 20, 2)

    custom_train_accuracies = []
    custom_test_accuracies = []
    custom_tree_heights = []

    sklearn_train_accuracies = []
    sklearn_test_accuracies = []
    sklearn_tree_heights = []

    for max_depth in max_depth_values:
        custom_tree = DecisionTree(max_depth=max_depth, min_samples_split=2, min_samples_leaf=1)
        custom_tree.fit(X_train, y_train)
        custom_tree_heights.append(calculate_tree_height(custom_tree.tree))

        y_train_pred_custom = custom_tree.predict(X_train)
        y_test_pred_custom = custom_tree.predict(X_test)
        custom_train_accuracies.append(accuracy_score(y_train, y_train_pred_custom))
        custom_test_accuracies.append(accuracy_score(y_test, y_test_pred_custom))

        sklearn_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=2, min_samples_leaf=1)
        sklearn_tree.fit(X_train, y_train)
        sklearn_tree_heights.append(sklearn_tree.get_depth())

        y_train_pred_sklearn = sklearn_tree.predict(X_train)
        y_test_pred_sklearn = sklearn_tree.predict(X_test)
        sklearn_train_accuracies.append(accuracy_score(y_train, y_train_pred_sklearn))
        sklearn_test_accuracies.append(accuracy_score(y_test, y_test_pred_sklearn))

    plt.figure(figsize=(12, 6))

    plt.plot(custom_tree_heights, custom_train_accuracies, label='Train (Custom Tree)', marker='o')
    plt.plot(custom_tree_heights, custom_test_accuracies, label='Test (Custom Tree)', marker='o')

    plt.plot(sklearn_tree_heights, sklearn_train_accuracies, label='Train (Sklearn Tree)', marker='s')
    plt.plot(sklearn_tree_heights, sklearn_test_accuracies, label='Test (Sklearn Tree)', marker='s')

    plt.xlabel('Высота дерева')
    plt.ylabel('Точность')
    plt.title('Зависимость точности от высоты дерева')
    plt.legend()
    plt.grid()
    plt.show()


# 4


def fourth_plot():
    n_trees_values = range(1, 51, 5)

    custom_train_accuracies = []
    custom_test_accuracies = []

    sklearn_train_accuracies = []
    sklearn_test_accuracies = []

    for n_trees in n_trees_values:
        custom_forest = RandomForest(n_trees=n_trees, max_depth=10, min_samples_split=2, max_features=None)
        custom_forest.fit(X_train, y_train)

        y_train_pred_custom = custom_forest.predict(X_train)
        y_test_pred_custom = custom_forest.predict(X_test)
        custom_train_accuracies.append(accuracy_score(y_train, y_train_pred_custom))
        custom_test_accuracies.append(accuracy_score(y_test, y_test_pred_custom))

        sklearn_forest = RandomForestClassifier(n_estimators=n_trees, max_depth=10, min_samples_split=2,
                                                min_samples_leaf=1, random_state=42)
        sklearn_forest.fit(X_train, y_train)

        y_train_pred_sklearn = sklearn_forest.predict(X_train)
        y_test_pred_sklearn = sklearn_forest.predict(X_test)
        sklearn_train_accuracies.append(accuracy_score(y_train, y_train_pred_sklearn))
        sklearn_test_accuracies.append(accuracy_score(y_test, y_test_pred_sklearn))

    plt.figure(figsize=(12, 6))

    plt.plot(n_trees_values, custom_train_accuracies, label='Train (Custom Forest)', marker='o')
    plt.plot(n_trees_values, custom_test_accuracies, label='Test (Custom Forest)', marker='o')

    plt.plot(n_trees_values, sklearn_train_accuracies, label='Train (Sklearn Forest)', marker='s')
    plt.plot(n_trees_values, sklearn_test_accuracies, label='Test (Sklearn Forest)', marker='s')

    plt.xlabel('Число деревьев')
    plt.ylabel('Точность')
    plt.title('Зависимость точности от числа деревьев')
    plt.legend()
    plt.grid()
    plt.show()


# 5


def fifth_plot():
    n_trees_values = range(1, 51, 5)

    train_accuracies = []
    test_accuracies = []

    for n_trees in n_trees_values:
        boosting = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=0.1, max_depth=3, random_state=42)
        boosting.fit(X_train, y_train)

        y_train_pred = boosting.predict(X_train)
        y_test_pred = boosting.predict(X_test)

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees_values, train_accuracies, label='Train Accuracy (Boosting)', marker='o')
    plt.plot(n_trees_values, test_accuracies, label='Test Accuracy (Boosting)', marker='s')
    plt.xlabel('Число деревьев')
    plt.ylabel('Точность')
    plt.title('Зависимость точности от числа деревьев (для Boosting)')
    plt.legend()
    plt.grid()
    plt.show()


X, y = get_data_from_csv()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
fifth_plot()

# tree = DecisionTree(max_depth=8, min_samples_split=10, min_information_gain=0.001)
# tree.fit(X_train, y_train)
#
# y_pred = tree.predict(X_test)
# accuracy = np.mean(y_pred == y_test)
# print(accuracy)

# forest = RandomForest(n_trees=10, max_depth=5, min_samples_split=3)
# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# print("Точность:", accuracy_score(y_test, y_pred))
