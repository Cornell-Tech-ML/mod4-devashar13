"""
Be sure you have minitorch installed in your Virtual Env.
>>> pip install -Ue .
"""

import random
import minitorch


class Network(minitorch.Module):
    """
    A neural network class with 3 linear layers.

    Args:
        hidden_layers (int): Number of hidden layers.

    Attributes:
        layer1 (Linear): First linear layer (input to hidden).
        layer2 (Linear): Second linear layer (hidden to hidden).
        layer3 (Linear): Third linear layer (hidden to output).
    """
    def __init__(self, hidden_layers):
        super().__init__()
        input_size = 2
        output_size = 1

        self.layer1 = Linear(input_size, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (tuple): Input data.

        Returns:
            minitorch.Scalar: Output after passing through all layers and applying sigmoid.
        """
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    """
    A linear layer that computes the weighted sum of inputs with biases.

    Args:
        in_size (int): Number of input features.
        out_size (int): Number of output features.

    Attributes:
        weights (list): List of weights for each connection.
        bias (list): List of biases for each output node.
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        """
        Forward pass through the linear layer.

        Args:
            inputs (tuple): Input data.

        Returns:
            list: List of outputs after applying the weighted sum and bias.
        """
        outputs = []
        for j in range(len(self.bias)):
            result = self.bias[j].value
            for i, input_value in enumerate(inputs):
                result += self.weights[i][j].value * input_value
            outputs.append(result)
        return outputs


def default_log_fn(epoch, total_loss, correct, losses):
    """
    Default logging function for training.

    Args:
        epoch (int): The current epoch number.
        total_loss (float): The total loss for the epoch.
        correct (int): The number of correct predictions.
        losses (list): List of losses.
    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    """
    Training class for the scalar neural network.

    Args:
        hidden_layers (int): Number of hidden layers.

    Attributes:
        hidden_layers (int): The number of hidden layers.
        model (Network): The neural network model.
    """
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        """
        Run a forward pass with a single input.

        Args:
            x (tuple): Input data.

        Returns:
            minitorch.Scalar: Model output.
        """
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=610, log_fn=default_log_fn):
        """
        Train the model using Stochastic Gradient Descent (SGD).

        Args:
            data (minitorch.Dataset): Training dataset.
            learning_rate (float): Learning rate for optimization.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 750.
            log_fn (function, optional): Logging function. Defaults to default_log_fn.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 3
    RATE = 0.5
    data = minitorch.datasets["Split"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)
