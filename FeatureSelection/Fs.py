import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth) or len(y) == 0:
            return Counter(y).most_common(1)[0][0] if len(y) > 0 else None

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return Counter(y).most_common(1)[0][0]

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_indices], y[left_indices], depth + 1),
            'right': self._build_tree(X[right_indices], y[right_indices], depth + 1)
        }

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        return parent_entropy - child_entropy

    @staticmethod
    def _entropy(y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sampled_features = np.random.choice(n_features,
                                                self.max_features or int(np.sqrt(n_features)),
                                                replace=False)
            X_sample = X[indices][:, sampled_features]
            y_sample = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, sampled_features))

            for feature in sampled_features:
                self.feature_importances_[feature] += self._compute_feature_importance(tree.tree, X_sample, y_sample, feature)

        self.feature_importances_ /= np.sum(self.feature_importances_)

    def _compute_feature_importance(self, tree, X, y, feature):
        if not isinstance(tree, dict):
            return 0

        if tree['feature'] == feature:
            left_indices = X[:, feature] < tree['threshold']
            right_indices = ~left_indices
            parent_entropy = self._entropy(y)
            left_entropy = self._entropy(y[left_indices])
            right_entropy = self._entropy(y[right_indices])

            n = len(y)
            n_left = len(y[left_indices])
            n_right = len(y[right_indices])
            gain = parent_entropy - (n_left / n) * left_entropy - (n_right / n) * right_entropy
            return gain

        left_gain = self._compute_feature_importance(tree['left'], X, y, feature)
        right_gain = self._compute_feature_importance(tree['right'], X, y, feature)
        return left_gain + right_gain

    @staticmethod
    def _entropy(y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            predictions.append(np.array([self._predict_tree(tree.tree, x, features) for x in X]))
        return np.round(np.mean(predictions, axis=0))

    def _predict_tree(self, tree, x, features):
        while isinstance(tree, dict):
            feature = features[tree['feature']]
            if x[feature] < tree['threshold']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree


class RecursiveFeatureElimination:
    def __init__(self, model, step=1):
        self.model = model
        self.step = step
        self.ranking_ = None
        self.support_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.ranking_ = np.ones(n_features, dtype=int)

        current_features = np.arange(n_features)
        while len(current_features) > 1:
            self.model.fit(X[:, current_features], y)
            importances = self.model.feature_importances_
            indices_to_remove = np.argsort(importances)[:self.step]
            self.ranking_[current_features[indices_to_remove]] = n_features - len(current_features) + 1
            current_features = np.delete(current_features, indices_to_remove)

        self.support_ = self.ranking_ == 1

    def transform(self, X):
        return X[:, self.support_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class FilterFeatureSelection:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.support_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        correlations = []

        for i in range(n_features):
            correlation = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(correlation))

        correlations = np.array(correlations)
        self.support_ = correlations > self.threshold

    def transform(self, X):
        return X[:, self.support_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def embedded_feature_selection(X, y, n):
    rf = RandomForest(n_trees=30, max_depth=10)
    rf.fit(X, y)
    top_n_indices = np.argsort(rf.feature_importances_)[-n:][::-1]
    top_n_features = rf.feature_importances_[top_n_indices]
    return top_n_indices, top_n_features


def wrapper_feature_selection(X, y, n=5):
    rf = RandomForest(n_trees=30, max_depth=10)
    rfe = RecursiveFeatureElimination(rf, step=1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    top_n_indices = np.argsort(ranking)[:n]
    top_n_values = ranking[top_n_indices]
    return top_n_indices, top_n_values


def filter_feature_selection(X, y, threshold=0.2, n=5):
    filter_method = FilterFeatureSelection(threshold)
    filter_method.fit(X, y)
    ranking = np.argsort(filter_method.support_)[::-1]
    top_n_indices = ranking[:n]
    top_n_values = filter_method.support_[top_n_indices]
    return top_n_indices, top_n_values


def lib_embedded_feature_selection(X, y, threshold="median", n=5):
    model = RandomForestClassifier(n_estimators=15, random_state=42)
    model.fit(X, y)
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    importances = model.feature_importances_
    top_n_indices = np.argsort(importances)[-n:][::-1]
    top_n_values = importances[top_n_indices]
    return top_n_indices, top_n_values


def lib_wrapper_feature_selection(X, y, n=5):
    model = LogisticRegression(max_iter=1000, random_state=42)
    selector = RFE(model, n_features_to_select=n)
    selector.fit(X, y)
    ranking = selector.ranking_
    top_n_indices = np.argsort(ranking)[:n]
    top_n_values = ranking[top_n_indices]
    return top_n_indices, top_n_values


def lib_filter_feature_selection(X, y, n=5):
    selector = SelectKBest(score_func=chi2, k=n)
    selector.fit(X, y)
    scores = selector.scores_
    top_n_indices = np.argsort(scores)[-n:][::-1]
    top_n_values = scores[top_n_indices]
    return top_n_indices, top_n_values


# Задачи

def get_top_30_words():
    indices, values = lib_embedded_feature_selection(X, y, n=30)
    words = [word for word, idx in vectorizer.vocabulary_.items() if idx in indices]
    '''
    ['free', 'to', 'win', 'text', 'txt', 'now', '50', 'your', 'customer', 
    'prize', 'claim', 'call', 'mobile', 'or', 'latest', 'cash', '100', '150p', 
    '16', 'reply', 'won', 'www', 'uk', 'new', '18', 'stop', 'service', 'guaranteed', '5000', 'cs']
    '''

    indices, values = lib_filter_feature_selection(X, y, n=30)
    words = [word for word, idx in vectorizer.vocabulary_.items() if idx in indices]
    '''
    ['free', 'to', 'win', 'text', 'txt', 'now', '50', 'your', 'prize', 'claim',
     'call', 'mobile', 'or', 'co', 'cash', '150p', '16', 'reply', 'urgent', 'won', 
     'www', 'uk', 'nokia', '18', 'stop', 'service', 'guaranteed', '500', 'cs', 'tone']
    '''

    indices, values = filter_feature_selection(X, y, n=30)
    words = [word for word, idx in vectorizer.vocabulary_.items() if idx in indices]
    '''
    ['free', 'to', 'win', 'text', 'txt', '50', 'your', 'customer', 'claim', 'call', 
    'mobile', 'or', 'co', '100', '150p', '16', 'reply', 'urgent', 'won', 'our', 'www', 
    'nokia', 'awarded', '18', 'service', 'guaranteed', 'contact', '500', 'cs', 'tone']
    '''

    indices, values = wrapper_feature_selection(X, y, n=30)
    words = [word for word, idx in vectorizer.vocabulary_.items() if idx in indices]
    '''
    ['free', 'to', 'win', 'text', 'txt', '50', 'your', 'prize', 'claim', 'call', 
    'mobile', 'won', 'co', 'cash', '150p', '16', 'reply', 'urgent', 'or', 'our', 'www', 
    'service', 'awarded', '18', 'nokia', 'guaranteed', '5000', 'cs', 'contact', 'tone']
    '''


def evaluate_classifiers_with_feature_selection(X, y, n_features=30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifiers = {
        "Логистическая регрессия": LogisticRegression(max_iter=1000, random_state=42),
        "Случайный лес": RandomForestClassifier(n_estimators=30, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }

    feature_selectors = {
        "libEmbedded": lib_embedded_feature_selection,
        "Filter": filter_feature_selection,
        "LibFilter": lib_filter_feature_selection
    }

    results = pd.DataFrame(columns=["До"] + list(feature_selectors.keys()), index=classifiers.keys())

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results.loc[name, "До"] = accuracy_score(y_test, y_pred)

    for method_name, feature_selector in feature_selectors.items():
        top_indices, _ = feature_selector(X_train, y_train, n=n_features)
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]

        for name, clf in classifiers.items():
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            results.loc[name, method_name] = accuracy_score(y_test, y_pred)

    '''
                                До     libEmbedded  Filter   LibFilter
    Логистическая регрессия  0.985048    0.965311  0.958134  0.866029
    Случайный лес            0.976675    0.971292  0.966507  0.869325
    SVM                      0.988636      0.9689  0.963517  0.8729
    '''


def cluster_and_evaluate(X, y_true=None, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    silhouette = silhouette_score(X, cluster_labels)
    metrics = {"Silhouette Score": silhouette}

    if y_true is not None:
        rand_index = adjusted_rand_score(y_true, cluster_labels)
        metrics["Adjusted Rand Index"] = rand_index

    return metrics


def cluster():
    print("Кластеризация до выбора признаков:")
    metrics_before = cluster_and_evaluate(X, y_true=y, n_clusters=2)
    print(metrics_before)

    X_selected = lib_embedded_feature_selection(X, y, n=30)

    print("Кластеризация после выбора признаков:")
    metrics_after = cluster_and_evaluate(X_selected, y_true=y, n_clusters=2)
    print(metrics_after)

    '''
    Кластеризация до выбора признаков:
    {'Silhouette Score': 0.19201211142663818, 'Adjusted Rand Index': 0.10356131658799739}
    Кластеризация после выбора признаков:
    {'Silhouette Score': 0.31012311582911495, 'Adjusted Rand Index': 0.17379022517592118}
    '''


def reduce_and_visualize(X, y, clusters, method_name, reduction_method, title):
    X_reduced = reduction_method.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='bwr', s=20, alpha=0.8)
    plt.title(f"{title} - Реальные классы ({method_name})")
    plt.xlabel("Первая компонента")
    plt.ylabel("Вторая компонента")

    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='cool', s=20, alpha=0.8)
    plt.title(f"{title} - Кластеры ({method_name})")
    plt.xlabel("Первая компонента")
    plt.ylabel("Вторая компонента")

    plt.tight_layout()
    plt.show()


def reduce():
    kmeans_full = KMeans(n_clusters=2, random_state=42)
    clusters_full = kmeans_full.fit_predict(X)

    reduce_and_visualize(X, y, clusters_full, "PCA", PCA(n_components=2), "До выбора признаков")
    reduce_and_visualize(X, y, clusters_full, "t-SNE", TSNE(n_components=2, random_state=42), "До выбора признаков")

    X_selected = lib_embedded_feature_selection(X, y, n=30)

    kmeans_selected = KMeans(n_clusters=2, random_state=42)
    clusters_selected = kmeans_selected.fit_predict(X_selected)

    reduce_and_visualize(X_selected, y, clusters_selected, "PCA", PCA(n_components=2), "После выбора признаков")
    reduce_and_visualize(X_selected, y, clusters_selected, "t-SNE", TSNE(n_components=2, random_state=42), "После выбора признаков")


data = pd.read_csv("SMS.tsv", sep='\t', header=0, names=['class', 'text'])
X_text = data['text']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)
X = X.toarray()
y = data['class'].apply(lambda x: 1 if x == 'spam' else 0).values
