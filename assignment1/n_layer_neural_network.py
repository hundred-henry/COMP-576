import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from three_layer_neural_network import plot_decision_boundary, generate_data, NeuralNetwork

class Layer:
    def __init__(self, input_dim: int, output_dim: int, actFun_type: str = 'tanh', reg_lambda: float = 0.01, seed: int = 0):
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

    def actFun(self, z: np.ndarray) -> np.ndarray:
        if self.actFun_type == 'tanh':
            return np.tanh(z)
        elif self.actFun_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.actFun_type == 'ReLU':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unsupported activation function: {self.actFun_type}")

    def diff_actFun(self, z: np.ndarray) -> np.ndarray:
        if self.actFun_type == 'tanh':
            return 1 - np.power(np.tanh(z), 2)
        elif self.actFun_type == 'sigmoid':
            return np.exp(-z) / np.power(1 + np.exp(-z), 2)
        elif self.actFun_type == 'ReLU':
            return np.where(z > 0, 1, 0)
        else:
            raise ValueError(f"Unsupported activation function: {self.actFun_type}")

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

class NeuralNetwork:
    def __init__(self, nn_input_dim: int, nn_hidden_dim: int, nn_output_dim: int, actFun_type: str = 'tanh', reg_lambda: float = 0.01, seed: int = 0):
        np.random.seed(seed)
        self.W1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
        self.b1 = np.zeros((1, nn_hidden_dim))
        self.W2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)
        self.b2 = np.zeros((1, nn_output_dim))
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

    # Activation and other methods similar to the original implementation...

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_dims: list[int], actFun_type: str = 'tanh', reg_lambda: float = 0.01, seed: int = 0):
        """
        Initialize a deep neural network with given layer dimensions
        :param layer_dims: List of integers representing the number of units in each layer
        :param actFun_type: Activation function type to use in all layers
        :param reg_lambda: L2 regularization lambda value
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(Layer(layer_dims[i], layer_dims[i + 1], actFun_type, reg_lambda, seed))

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        """
        Feedforward propagation through all layers.
        :param X: Input data
        :return: Output probabilities from the last layer
        """
        self.a = [X]
        for layer in self.layers:
            self.a.append(layer.feedforward(self.a[-1]))
        exp_scores = np.exp(self.a[-1])
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backprop(self, X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Backpropagation through all layers.
        :param X: Input data
        :param y: True labels
        :return: List of tuples (dW, db) for each layer
        """
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
        """
        Compute the loss with L2 regularization
        """
        num_examples = len(X)
        self.feedforward(X)
        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add regularization terms (L2 regularization)
        reg_loss = 0.5 * sum(np.sum(np.square(layer.W)) for layer in self.layers)
        data_loss += self.layers[0].reg_lambda * reg_loss

        return (1. / num_examples) * data_loss

    def fit_model(self, X: np.ndarray, y: np.ndarray, epsilon: float = 0.01, num_passes: int = 20000, print_loss: bool = True):
        """
        Train the model using backpropagation and gradient descent.
        """
        for i in range(num_passes):
            # Forward propagation
            self.feedforward(X)

            # Backpropagation
            grads = self.backprop(X, y)

            # Gradient descent parameter update
            for layer, (dW, db) in zip(self.layers, grads):
                layer.W += -epsilon * dW
                layer.b += -epsilon * db

            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y)}")


def main():
    X, y = datasets.make_moons(n_samples=200, noise=0.2)
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    layer_dims = [2, 5, 5, 2]  # 3-layer deep network: Input-2, Hidden1-5, Hidden2-5, Output-2
    deep_nn = DeepNeuralNetwork(layer_dims, actFun_type='sigmoid')
    deep_nn.fit_model(X, y)
    plot_decision_boundary(lambda x: deep_nn.feedforward(x).argmax(axis=1), X, y)


if __name__ == "__main__":
    main()