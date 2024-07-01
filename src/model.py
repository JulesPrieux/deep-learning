import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SAE, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Create encoder layers
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.encoder.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        # Create decoder layers (reverse of encoder)
        for h_dim in reversed(hidden_dims[:-1]):
            self.decoder.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.decoder.append(nn.Linear(prev_dim, input_dim))

    def forward(self, x):
        # Encode
        for layer in self.encoder:
            x = torch.relu(layer(x))

        # Decode
        for layer in self.decoder:
            x = torch.relu(layer(x))

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim

        # Create hidden layers
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


def fit_sae_mlp(
    X,
    y,
    hidden_dims_sae,
    hidden_dims_mlp,
    output_dim,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
):
    """
    Fits a Stacked Autoencoder Multilayer Perceptron (SAE-MLP) to the given data.

    Parameters:
    X (np.ndarray): Input features.
    y (np.ndarray): Target values.
    hidden_dims_sae (list): List of hidden layer sizes for the SAE.
    hidden_dims_mlp (list): List of hidden layer sizes for the MLP.
    output_dim (int): Dimension of the output layer.
    num_epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    learning_rate (float): Learning rate for the optimizer.

    Returns:
    model (nn.Module): The trained MLP model.
    """
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create data loaders
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the SAE and train it
    sae = SAE(input_dim=X.shape[1], hidden_dims=hidden_dims_sae)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    sae.train()
    for epoch in range(num_epochs):
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            outputs = sae(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], SAE Loss: {loss.item():.4f}")

    # Use the trained SAE to encode the inputs
    sae.eval()
    with torch.no_grad():
        X_encoded = sae.encoder[0](X_tensor)
        for layer in sae.encoder[1:]:
            X_encoded = torch.relu(layer(X_encoded))

    # Define the MLP and train it
    mlp = MLP(
        input_dim=hidden_dims_sae[-1],
        hidden_dims=hidden_dims_mlp,
        output_dim=output_dim,
    )
    criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    mlp.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            # Encode the input
            with torch.no_grad():
                X_batch_encoded = sae.encoder[0](X_batch)
                for layer in sae.encoder[1:]:
                    X_batch_encoded = torch.relu(layer(X_batch_encoded))

            optimizer.zero_grad()
            outputs = mlp(X_batch_encoded)
            loss = criterion(outputs, y_batch.long())
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], MLP Loss: {loss.item():.4f}")

    return sae, mlp


def predict_sae_mlp_classification(sae, mlp, X_new):
    """
    Predicts the class labels for new input data using the trained SAE-MLP model.

    Parameters:
    sae (nn.Module): The trained SAE model.
    mlp (nn.Module): The trained MLP model.
    X_new (np.ndarray): New input features.

    Returns:
    np.ndarray: The predicted class labels.
    """
    # Convert new input data to PyTorch tensor
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

    # Encode the input using the trained SAE encoder
    sae.eval()
    with torch.no_grad():
        X_encoded = sae.encoder[0](X_new_tensor)
        for layer in sae.encoder[1:]:
            X_encoded = torch.relu(layer(X_encoded))

    # Predict using the trained MLP
    mlp.eval()
    with torch.no_grad():
        outputs = mlp(X_encoded)
        _, predicted_labels = torch.max(outputs, 1)

    return predicted_labels.numpy()


# Example usage
if __name__ == "__main__":
    # Generate some synthetic data for demonstration purposes
    np.random.seed(42)
    X = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, size=(1000,))

    hidden_dims_sae = [16, 8]
    hidden_dims_mlp = [8, 4]
    output_dim = 1  # Change to the number of classes for classification problems

    model = fit_sae_mlp(X, y, hidden_dims_sae, hidden_dims_mlp, output_dim)
