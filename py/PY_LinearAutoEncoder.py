import numpy as np


class LinearAutoEncoder:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.W_encoder = np.random.randn(input_size, hidden_size)
        self.b_encoder = np.zeros(hidden_size)
        self.W_decoder = np.random.randn(hidden_size, input_size)
        self.b_decoder = np.zeros(input_size)
        self.losses = []

    def forward(self, X):
        # Encoding
        encoded = np.dot(X, self.W_encoder) + self.b_encoder

        # Decoding
        decoded = np.dot(encoded, self.W_decoder) + self.b_decoder
        return encoded, decoded

    def compute_loss(self, X, decoded):
        # Mean Squared Error
        return np.mean((X - decoded) ** 2)

    def train(self, X, epochs, learning_rate):
        self.losses = []
        for epoch in range(epochs):
            # Forward pass, done once per epoch
            encoded, decoded = self.forward(X)

            # Compute loss and check for NaN
            loss = self.compute_loss(X, decoded)
            if np.isnan(loss):
                print(f'Epoch {epoch}, Loss: NaN (Stopping training)')
                break

            # Backward pass
            error = decoded - X
            dW_decoder = np.dot(encoded.T, error)
            db_decoder = np.sum(error, axis=0)
            dW_encoder = np.dot(X.T, error.dot(self.W_decoder.T))
            db_encoder = np.sum(error.dot(self.W_decoder.T), axis=0)

            # Gradient Clipping
            dW_decoder = np.clip(dW_decoder, -1, 1)
            db_decoder = np.clip(db_decoder, -1, 1)
            dW_encoder = np.clip(dW_encoder, -1, 1)
            db_encoder = np.clip(db_encoder, -1, 1)

            # Update weights and biases
            self.W_decoder -= learning_rate * dW_decoder
            self.b_decoder -= learning_rate * db_decoder
            self.W_encoder -= learning_rate * dW_encoder
            self.b_encoder -= learning_rate * db_encoder

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
                self.losses.append(loss)

    def getLoss(self):
        return self.losses[2:]

