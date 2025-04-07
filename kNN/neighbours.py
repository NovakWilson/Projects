from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, minkowski
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import csv
import math


def get_data_from_csv(filename="normalized_data.csv"):
    with open(filename, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter='\n')
        headers = next(reader)
        X = []
        Y = []
        for row in reader:
            row = row[0].split(',')
            X.append(row[:4] + row[5:])
            Y.append(row[4])
        return np.array(X[:1500]), np.array(Y[:1500])


def calculate_distance(x1, x2, metric='cos', p=2):
    if metric == 'cos':
        return cosine(x1, x2)
    elif metric == 'minkow':
        return minkowski(x1, x2, p=p)
    elif metric == 'euclid':
        return np.linalg.norm(x1 - x2)


def kernel_function(u, kernel_type='uniform', a=1, b=1):
    if kernel_type == 'uniform':
        return 1 if abs(u) <= 1 else 0
    elif kernel_type == 'gauss':
        return math.exp(-0.5 * u ** 2) / math.sqrt(2 * math.pi)
    elif kernel_type == 'general':
        return (1 - abs(u) ** a) ** b if abs(u) <= 1 else 0
    elif kernel_type == 'triangular':
        return 1 - abs(u) if abs(u) <= 1 else 0


def knn_predict(X_train, y_train, X_test, weights, k=82, metric='cos', kernel='general', window_type='fixed',
                radius=None):
    predictions = []
    for x_test in X_test:
        distances = [(calculate_distance(x_test, x_train, metric=metric), y_train[i], weights[i])
                     for i, x_train in enumerate(X_train)]

        distances.sort(key=lambda x: x[0])
        neighbors = []

        if window_type == 'fixed':
            neighbors = distances[:k]
        elif window_type == 'variable' and radius is not None:
            neighbors = [d for d in distances if d[0] <= radius]

        weighted_classes = Counter()
        for dist, label, prior_weight in neighbors:
            kernel_weight = kernel_function(dist, kernel_type=kernel)
            total_weight = kernel_weight * prior_weight
            weighted_classes[label] += total_weight
        try:
            predicted_class = weighted_classes.most_common(1)[0][0]
            predictions.append(predicted_class)
        except:
            predictions.append("no")
    return predictions


def find_best_hyperparameters():
    k_values = range(50, 100)
    metrics = ['cos', 'minkow', 'euclid']
    kernels = ['uniform', 'gauss', 'general', 'triangular']
    window_types = ['fixed', 'variable']
    radius_values = [0.1 * i for i in range(1, 11)]
    best_params = None
    best_score = 0

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cnt = 1
    for k, metric, kernel, window_type in itertools.product(k_values, metrics, kernels, window_types):
        cnt += 1
        if window_type == 'variable':
            for radius in radius_values:
                scores = []
                for train_index, val_index in kf.split(X_train):
                    X_tr, X_val = X_train[train_index], X_train[val_index]
                    y_tr, y_val = y_train[train_index], y_train[val_index]

                    weights = [1.0] * len(X_tr)

                    y_pred = knn_predict(X_tr, y_tr, X_val, weights=weights, k=k, metric=metric, kernel=kernel, window_type=window_type,
                                         radius=radius)
                    scores.append(accuracy_score(y_val, y_pred))

                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'k': k, 'metric': metric, 'kernel': kernel, 'window_type': window_type,
                                   'radius': radius}
        else:
            scores = []
            for train_index, val_index in kf.split(X_train):
                X_tr, X_val = X_train[train_index], X_train[val_index]
                y_tr, y_val = y_train[train_index], y_train[val_index]

                weights = [1.0] * len(X_tr)

                y_pred = knn_predict(X_tr, y_tr, X_val, weights=weights, k=k, metric=metric, kernel=kernel, window_type=window_type)
                scores.append(accuracy_score(y_val, y_pred))

            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'k': k, 'metric': metric, 'kernel': kernel, 'window_type': window_type}

    print("Лучшие гиперпараметры:", best_params)
    print("Лучшая оценка точности:", best_score)


def get_plot():
    k_values = range(75, 100)

    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        print(k)
        y_train_pred = knn_predict(X_train, y_train, X_train, weights=[1.0] * len(X_train),
                                   k=k, metric='cos', kernel='general', window_type='fixed')
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        y_test_pred = knn_predict(X_train, y_train, X_test, weights=[1.0] * len(X_train),
                                  k=k, metric='cos', kernel='general', window_type='fixed')
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)
    plt.ylim(0.65, 0.85)
    plt.xlim(73, 102)
    plt.plot(k_values, train_accuracies, label='Точность на тренировочной выборке')
    plt.plot(k_values, test_accuracies, label='Точность на тестовой выборке')
    plt.xlabel('Число соседей (k)')
    plt.ylabel('Точность')
    plt.title('Зависимость точности от числа соседей')
    plt.legend()
    plt.show()


def lowess(X, y, kernel=None):
    if kernel is None:
        kernel = 'gauss'

    w = []
    n = len(X)
    for i in range(n):
        cur_x = np.delete(X, i, axis=0)
        cur_y = np.delete(y, i, axis=0)

        weights = np.ones(len(cur_x))

        y_pred = knn_predict(
            X_train=cur_x,
            y_train=cur_y,
            X_test=X[i:i+1],
            weights=weights,
        )[0]

        distance = 0 if y[i] == y_pred else 1

        w_i = kernel_function(distance, kernel_type=kernel)
        w.append(w_i)

    return w


X, y = get_data_from_csv()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)


# find_best_hyperparameters()
# {'k': 82, 'metric': 'cos', 'kernel': 'general', 'window_type': 'fixed'}
# Лучшая оценка точности:  0.77


# weights = [1.0] * len(X_train)
# y_pred = knn_predict(X_train, y_train, X_test, weights=weights, k=82, metric='cos', kernel='general', window_type='fixed')
# accuracy = accuracy_score(y_train, y_pred)
# print(f"Точность модели: {accuracy}")
# Точность модели: 0.726
