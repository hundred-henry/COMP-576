import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from three_layer_neural_network import plot_decision_boundary, generate_data, NeuralNetwork
from sklearn.model_selection import train_test_split

def predict_other_dataset():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = DeepNeuralNetwork(layer_dims=[30, 50, ], actFun_type='sigmoid', epsilon=0.005, reg_lambda=0.005, seed=0)
    model.fit_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: %.4f" % np.mean(y_pred == y_test))


class Layer:
    def __init__(self, input_dim: int, output_dim: int, actFun_type: str = 'tanh',
                 reg_lambda: float = 0.005, seed: int = 0):
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

    def actFun(self, z: np.ndarray) -> np.ndarray:
        if self.actFun_type == 'tanh' or self.actFun_type == 'Tanh':
            return np.tanh(z)
        elif self.actFun_type == 'sigmoid' or self.actFun_type == 'Sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.actFun_type == 'ReLU' or self.actFun_type == 'relu':
            return np.maximum(0, z)

    def diff_actFun(self, z: np.ndarray) -> np.ndarray:
        if self.actFun_type == 'tanh' or self.actFun_type == 'Tanh':
            return 1 - np.power(np.tanh(z), 2)
        elif self.actFun_type == 'sigmoid' or self.actFun_type == 'Sigmoid':
            return np.exp(-z) / np.power(1 + np.exp(-z), 2)
        elif self.actFun_type == 'ReLU' or self.actFun_type == 'relu':
            return np.where(z > 0, 1, 0)

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a

    def backprop(self, delta: np.ndarray, input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta_actFun = delta * self.diff_actFun(self.z)
        dW = np.dot(input_data.T, delta_actFun)
        db = np.sum(delta_actFun, axis=0, keepdims=True)
        delta_next = np.dot(delta_actFun, self.W.T)
        return delta_next, dW, db


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_dims: list[int], actFun_type: str = 'tanh',
                 epsilon: float = 0.005, reg_lambda: float = 0.005, seed: int = 0):
        np.random.seed(seed)
        self.epsilon = epsilon
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(Layer(layer_dims[i], layer_dims[i + 1], actFun_type, reg_lambda, seed))

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.a = [X]
        for layer in self.layers:
            self.a.append(layer.feedforward(self.a[-1]))
        exp_scores = np.exp(self.a[-1])
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backprop(self, X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        num_examples = len(X)
        delta = self.probs.copy()
        delta[range(num_examples), y] -= 1

        grads = []
        for i in reversed(range(len(self.layers))):
            delta, dW, db = self.layers[i].backprop(delta, self.a[i])
            dW += self.layers[i].reg_lambda * self.layers[i].W  # L2 regularization term
            grads.append((dW, db))

        return list(reversed(grads))

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        num_examples = len(X)
        self.feedforward(X)
        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))


        # Add regularization terms (L2 regularization)
        reg_loss = 0.5 * sum(np.sum(np.square(layer.W)) for layer in self.layers)
        data_loss += self.layers[0].reg_lambda * reg_loss

        return (1. / num_examples) * data_loss

    def fit_model(self, X: np.ndarray, y: np.ndarray, num_passes: int = 20000, print_loss: bool = True):
        for i in range(num_passes):
            # Forward propagation
            self.feedforward(X)

            # Backpropagation
            grads = self.backprop(X, y)

            # Gradient descent parameter update
            for layer, (dW, db) in zip(self.layers, grads):
                layer.W += -self.epsilon * dW
                layer.b += -self.epsilon * db

            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y)}")

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    layer_dims = [2, 20, 20, 2]
    deep_nn = DeepNeuralNetwork(layer_dims, actFun_type='tanh'
                                , epsilon=0.001, reg_lambda=0.001, seed=0)
    deep_nn.fit_model(X, y)
    plot_decision_boundary(lambda x: deep_nn.feedforward(x).argmax(axis=1), X, y)

    predict_other_dataset()


if __name__ == "__main__":
    main()