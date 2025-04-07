from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
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
            Y.append(1 if row[4] == 'yes' else -1)
        return np.array(X[:1500], dtype=float), np.array(Y[:1500], dtype=int)


def ridge_regression(X, y, lambd=10000):
    n, p = X.shape
    I = np.eye(p)
    beta = np.linalg.inv(X.T @ X + lambd * I) @ X.T @ y
    return beta


def gradient_descent(X, y, loss='hinge', alpha=0.5, lambda_=1.0, learn_rate=0.01, iters=1000):
    n, p = X.shape
    w = np.zeros(p)
    b = 0

    for _ in range(iters):
        margins = y * (np.dot(X, w) + b)

        if loss == 'hinge':
            grad_loss_w = -np.dot((margins < 1) * y, X) / n
            grad_loss_b = -(margins < 1).dot(y) / n
        elif loss == 'logistic':
            probabilities = 1 / (1 + np.exp(-margins))
            grad_loss_w = -np.dot((1 - probabilities) * y, X) / n
            grad_loss_b = -(1 - probabilities).dot(y) / n
        else:  # 'для exp'
            grad_loss_w = -np.dot(np.exp(-margins) * y, X) / n
            grad_loss_b = -np.exp(-margins).dot(y) / n

        grad_regularization = alpha * np.sign(w) + (1 - alpha) * 2 * w

        w -= learn_rate * (grad_loss_w + lambda_ * grad_regularization)
        b -= learn_rate * grad_loss_b

    return w, b


def kernel(x1, x2, kernel_type='linear', d=3, sigma=1.0, c=1):
    if kernel_type == 'linear':
        return np.dot(x1, x2)
    elif kernel_type == 'polynomial':
        return (np.dot(x1, x2) + c) ** d
    elif kernel_type == 'rbf':
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))


def kernel_matrix(X, kernel_type='linear', d=3, sigma=1.0, c=1):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], kernel_type, d, sigma, c)
    return K


def svm_smo(X, y, kernel_type='polynomial', C=1.0, tol=1e-3, max_iters=1000, d=3, sigma=1.0, c=1):
    n, p = X.shape
    alpha = np.zeros(n)
    b = 0
    K = kernel_matrix(X, kernel_type=kernel_type, d=d, sigma=sigma, c=c)

    for _ in range(max_iters):
        num_changed_alphas = 0
        for i in range(n):
            E_i = np.sum(alpha * y * K[:, i]) + b - y[i]

            if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0):
                j = np.random.choice([x for x in range(n) if x != i])
                E_j = np.sum(alpha * y * K[:, j]) + b - y[j]

                alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                if y[i] == y[j]:
                    L, H = max(0, alpha[j] + alpha[i] - C), min(C, alpha[j] + alpha[i])
                else:
                    L, H = max(0, alpha[j] - alpha[i]), min(C, C + alpha[j] - alpha[i])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alpha[j] -= y[j] * (E_i - E_j) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < tol:
                    continue

                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1

        if num_changed_alphas == 0:
            break

    return alpha, b


def predict(X_train, X_test, y_train, alpha, b, kernel_type='polynomial', d=3, sigma=1.0, c=1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    y_pred = np.zeros(n_test)

    for i in range(n_test):
        prediction = 0
        for j in range(n_train):
            if alpha[j] > 0:
                prediction += alpha[j] * y_train[j] * kernel(X_train[j], X_test[i], kernel_type, d, sigma, c)
        y_pred[i] = prediction + b

    return np.sign(y_pred)


def learning_curve(X_train, y_train, model_type='linear', kernel_type='linear', lambda_=0.01, lr=0.1, max_iters=1000, C=1.0):
    risks = []
    iterations = range(1, max_iters + 1, 100)

    for iter_count in iterations:
        if model_type == 'linear':
            w, b = gradient_descent(X_train, y_train, learn_rate=lr, lambda_=lambda_, iters=iter_count)
            risk = np.mean(np.maximum(0, 1 - y_train * (X_train @ w + b)))
        elif model_type == 'svm':
            alpha, b = svm_smo(X_train, y_train, kernel_type=kernel_type, C=C, max_iters=iter_count)
            risk = np.mean(np.maximum(0, 1 - y_train * (np.sum(alpha * y_train * kernel_matrix(X_train, kernel_type=kernel_type), axis=0) + b)))
        risks.append(risk)

    return iterations, risks


def get_plot():
    iterations_linear, risks_linear = learning_curve(X_train, y_train, model_type='linear', lambda_=0.01, lr=0.1, max_iters=2000)

    iterations_svm, risks_svm = learning_curve(X_train, y_train, model_type='svm', kernel_type='polynomial', C=1.0, max_iters=2000)

    plt.plot(iterations_linear, risks_linear, label='Линейная классификация (Hinge-loss)', marker='o')
    plt.plot(iterations_svm, risks_svm, label='SVM (Hinge-loss)', marker='s')
    plt.xlabel('Количество итераций')
    plt.ylabel('Эмпирический риск')
    plt.title('Кривая обучения')
    plt.legend()
    plt.grid()
    plt.show()


def learning_curve_with_confidence(X, y, model_type='linear', kernel_type='polynomial', lambda_=0.01, lr=0.1, max_iters=1000, C=1.0, n_splits=5):
    iterations = range(10, max_iters + 1, 10)
    mean_scores = []
    std_scores = []

    for iter_count in iterations:
        split_scores = []
        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
            if model_type == 'linear':
                w, b = gradient_descent(X_train, y_train, learn_rate=lr, lambda_=lambda_, iters=iter_count)
                y_pred = np.sign(X_test @ w + b)
            elif model_type == 'svm':
                alpha, b = svm_smo(X_train, y_train, kernel_type=kernel_type, C=C, max_iters=iter_count)
                y_pred = predict(X_train, X_test, y_train, alpha, b, kernel_type=kernel_type)
            score = accuracy_score(y_test, y_pred)
            split_scores.append(score)
        mean_scores.append(np.mean(split_scores))
        std_scores.append(np.std(split_scores))

    return iterations, mean_scores, std_scores


def plot_learning_curve_with_test_baseline(X, y, X_test, y_test, model_type='linear', kernel_type='polynomial', lambda_=0.01, lr=0.1, max_iters=2000, C=2.0, n_splits=5):
    iterations_linear, mean_scores_linear, std_scores_linear = learning_curve_with_confidence(
        X, y, model_type='linear', lambda_=lambda_, lr=lr, max_iters=max_iters)

    iterations_svm, mean_scores_svm, std_scores_svm = learning_curve_with_confidence(
        X, y, model_type='svm', kernel_type=kernel_type, C=C, max_iters=max_iters)

    if model_type == 'linear':
        w, b = gradient_descent(X, y, learn_rate=lr, lambda_=lambda_, iters=max_iters)
        y_test_pred = np.sign(X_test @ w + b)
        test_accuracy = accuracy_score(y_test, y_test_pred)
    elif model_type == 'svm':
        alpha, b = svm_smo(X, y, kernel_type=kernel_type, C=C, max_iters=max_iters)
        y_test_pred = predict(X, X_test, y, alpha, b, kernel_type=kernel_type)
        test_accuracy = accuracy_score(y_test, y_test_pred)

    plt.figure(figsize=(10, 6))

    plt.plot(iterations_linear, mean_scores_linear, label='Линейная классификация (точность)', marker='o')
    plt.fill_between(iterations_linear,
                     np.array(mean_scores_linear) - np.array(std_scores_linear),
                     np.array(mean_scores_linear) + np.array(std_scores_linear),
                     alpha=0.2)

    plt.plot(iterations_svm, mean_scores_svm, label='SVM (точность)', marker='s')
    plt.fill_between(iterations_svm,
                     np.array(mean_scores_svm) - np.array(std_scores_svm),
                     np.array(mean_scores_svm) + np.array(std_scores_svm),
                     alpha=0.2)

    plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Тестовая точность ({test_accuracy:.2f})')

    plt.xlabel('Число итераций')
    plt.ylabel('Точность')
    plt.title('Кривая обучения с доверительными интервалами и тестовой базовой линией')
    plt.legend()
    plt.grid()
    plt.show()


X, y = get_data_from_csv()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plot_learning_curve_with_test_baseline(X_train, y_train, X_test, y_test, model_type='linear', kernel_type='polynomial')
