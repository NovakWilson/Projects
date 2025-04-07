from sklearn.model_selection import train_test_split
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
        return np.array(X[:1500], dtype=float), np.array(Y[:1500], dtype=int)


def identity(x):
    return x


def identity_grad(x):
    return np.ones_like(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1.0 - np.tanh(x)**2


activations = {
    'identity': identity,
    'relu': relu,
    'tanh': tanh
}
activation_grads = {
    'identity': identity_grad,
    'relu': relu_grad,
    'tanh': tanh_grad
}


class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = np.ones((input_dim, output_dim))
        self.b = np.zeros(output_dim)

        self.activation_name = activation
        self.activation = activations[activation]
        self.activation_grad = activation_grads[activation]

        self.x = None
        self.z = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return self.activation(self.z)

    def backward(self, d_out):
        dZ = d_out * self.activation_grad(self.z)
        self.grad_W = self.x.T @ dZ
        self.grad_b = np.sum(dZ, axis=0)
        d_x = dZ @ self.W.T
        return d_x

    def update_params(self, dW, db):
        self.W += dW
        self.b += db


class RBFLayer:
    def __init__(self, input_dim, output_dim, sigma=1.0, activation='identity'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma

        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.b = np.zeros(output_dim)

        self.activation_name = activation
        self.activation = activations[activation]
        self.activation_grad = activation_grads[activation]

        self.x = None
        self.dist_sq = None
        self.rbf_values = None
        self.out_pre_activation = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        x_expanded = x[:, np.newaxis, :]
        W_expanded = self.W[np.newaxis, :, :]

        diff = x_expanded - W_expanded
        dist_sq = np.sum(diff ** 2, axis=2)
        self.dist_sq = dist_sq

        rbf_vals = np.exp(-dist_sq / (2 * (self.sigma ** 2)))
        self.rbf_values = rbf_vals
        out_pre_act = rbf_vals + self.b
        self.out_pre_activation = out_pre_act

        out = self.activation(out_pre_act)
        return out

    def backward(self, d_out):
        d_pre_act = d_out * self.activation_grad(self.out_pre_activation)
        self.grad_b = np.sum(d_pre_act, axis=0)
        d_rbf_vals = d_pre_act
        d_dist_sq = d_rbf_vals * (-1.0 / (2 * self.sigma * self.sigma)) * self.rbf_values

        x_expanded = self.x[:, np.newaxis, :]
        W_expanded = self.W[np.newaxis, :, :]
        diff = x_expanded - W_expanded

        d_dist_sq_3d = d_dist_sq[:, :, np.newaxis]

        d_x = d_dist_sq_3d * 2.0 * diff
        d_x = np.sum(d_x, axis=1)
        d_W = - d_dist_sq_3d * 2.0 * diff
        self.grad_W = np.sum(d_W, axis=0)
        return d_x

    def update_params(self, dW, db):
        self.W += dW
        self.b += db


def softmax_crossentropy(logits, targets):
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits_stable = logits - max_logits

    exp_scores = np.exp(logits_stable)
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sums

    log_probs = np.log(probs + 1e-12)
    loss_each = - np.sum(targets * log_probs, axis=1)
    loss = np.mean(loss_each)
    return loss


def softmax_crossentropy_grad(logits, targets):
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits_stable = logits - max_logits

    exp_scores = np.exp(logits_stable)
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sums

    batch_size = logits.shape[0]
    grad = (probs - targets) / batch_size
    return grad


class AdagradOptimizer:
    def __init__(self, layers, lr=0.01, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.eps = eps

        self.accum_W = []
        self.accum_b = []

        for layer in layers:
            self.accum_W.append(np.zeros_like(layer.W))
            self.accum_b.append(np.zeros_like(layer.b))

    def step(self):
        for i, layer in enumerate(self.layers):
            self.accum_W[i] += layer.grad_W ** 2
            self.accum_b[i] += layer.grad_b ** 2

            dW = - self.lr * layer.grad_W / (np.sqrt(self.accum_W[i]) + self.eps)
            db = - self.lr * layer.grad_b / (np.sqrt(self.accum_b[i]) + self.eps)

            layer.update_params(dW, db)


def forward_pass(layers, X):
    out = X
    for layer in layers:
        out = layer.forward(out)
    return out


def backward_pass(layers, d_out):
    for layer in reversed(layers):
        d_out = layer.backward(d_out)


def to_one_hot(y, num_classes=2):
    return np.eye(num_classes)[y]


def compute_accuracy(logits, y_true):
    y_pred = np.argmax(logits, axis=1)
    return np.mean(y_pred == y_true)


def train_model(layers, X_train, Y_train, X_test, Y_test,
                           epochs=30, lr=0.01):
    optimizer = AdagradOptimizer(layers, lr=lr)

    history = []
    for epoch in range(epochs):
        logits = forward_pass(layers, X_train)

        train_loss = softmax_crossentropy(logits, Y_train)

        d_logits = softmax_crossentropy_grad(logits, Y_train)

        backward_pass(layers, d_logits)

        optimizer.step()

        train_acc = compute_accuracy(logits, np.argmax(Y_train, axis=1))

        test_logits = forward_pass(layers, X_test)
        test_loss = softmax_crossentropy(test_logits, Y_test)
        test_acc = compute_accuracy(test_logits, np.argmax(Y_test, axis=1))

        history.append((train_loss, test_loss, train_acc, test_acc))

        # print(f"Epoch {epoch + 1}/{epochs}: "
        #       f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
        #       f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    return np.array(history)


def first_plot(all_histories):
    for i, hist in enumerate(all_histories, start=1):
        plt.plot(hist[:, 0], label=f'Model {i} Train Loss')
    plt.title("Train Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Test Loss
    for i, hist in enumerate(all_histories, start=1):
        plt.plot(hist[:, 1], label=f'Model {i} Test Loss')
    plt.title("Test Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Train Accuracy
    for i, hist in enumerate(all_histories, start=1):
        plt.plot(hist[:, 2], label=f'Model {i} Train Accuracy')
    plt.title("Train Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Test Accuracy
    for i, hist in enumerate(all_histories, start=1):
        plt.plot(hist[:, 3], label=f'Model {i} Test Accuracy')
    plt.title("Test Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def count_parameters(layers):
    total = 0
    for layer in layers:
        w_params = layer.W.size
        b_params = layer.b.size
        total += w_params + b_params
    return total


def test_accuracy_params(model_configs, all_histories):
    param_list = []
    transform_list = []
    test_acc_list = []

    for i, config in enumerate(model_configs, start=1):
        layers_for_count = []
        for layer_template in config:
            layer = DenseLayer(
                layer_template.input_dim,
                layer_template.output_dim,
                layer_template.activation_name
            )
            layers_for_count.append(layer)

        param_count = count_parameters(layers_for_count)
        transform_count = len(config)

        history = all_histories[i - 1]
        final_test_acc = history[-1, 3]

        param_list.append(param_count)
        transform_list.append(transform_count)
        test_acc_list.append(final_test_acc)

    # Число Параметров
    plt.scatter(param_list, test_acc_list)
    for i in range(len(param_list)):
        plt.annotate(f"M{i + 1}", (param_list[i], test_acc_list[i]))
    plt.title("Зависимость точности на тесте от числа параметров")
    plt.xlabel("Число параметров")
    plt.ylabel("Test Accuracy")
    plt.show()

    # Число слоев
    plt.scatter(transform_list, test_acc_list)
    for i in range(len(transform_list)):
        plt.annotate(f"M{i + 1}", (transform_list[i], test_acc_list[i]))
    plt.title("Зависимость точности на тесте от числа слоев")
    plt.xlabel("Число слоёв")
    plt.ylabel("Test Accuracy")
    plt.show()


X, y = get_data_from_csv()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Y_train = to_one_hot(y_train, num_classes=2)
Y_test = to_one_hot(y_test, num_classes=2)

# model_configs = [
#     [DenseLayer(X_train.shape[1], 5, activation='relu'),
#      DenseLayer(5, 2, activation='identity')],
#
#     [DenseLayer(X_train.shape[1], 10, activation='relu'),
#      DenseLayer(10, 5, activation='relu'),
#      DenseLayer(5, 2, activation='identity')]
# ]

# model_configs = [
#     [DenseLayer(X_train.shape[1], 5, 1, activation='relu'),
#      DenseLayer(5, 2, 1, activation='identity')],
#
#     [DenseLayer(X_train.shape[1], 10, 1, activation='relu'),
#      DenseLayer(10, 5, 1, activation='relu'),
#      DenseLayer(5, 2, 1, activation='identity')]
# ]

model_configs = [
    [
        DenseLayer(X_train.shape[1], 10, activation='relu'),
        RBFLayer(10, 5, sigma=1.0, activation='tanh'),
        DenseLayer(5, 2, activation='identity')
    ],
    [
        DenseLayer(X_train.shape[1], 10, activation='relu'),
        RBFLayer(10, 5, sigma=1.0, activation='relu'),
        DenseLayer(5, 5, activation='relu'),
        RBFLayer(5, 2, sigma=1.0, activation='identity')
    ]
]

all_histories = []
for i, config in enumerate(model_configs, start=1):
    layers = []
    for layer_template in config:
        if isinstance(layer_template, DenseLayer):
            layer = DenseLayer(
                layer_template.input_dim,
                layer_template.output_dim,
                layer_template.activation_name
            )
        else:
            layer = RBFLayer(
                layer_template.input_dim,
                layer_template.output_dim,
                layer_template.sigma,
                layer_template.activation_name
            )
        layers.append(layer)

    history = train_model(layers, X_train, Y_train, X_test, Y_test,
                          epochs=1000, lr=0.01)
    all_histories.append(history)

first_plot(all_histories)
#test_accuracy_params(model_configs, all_histories)
