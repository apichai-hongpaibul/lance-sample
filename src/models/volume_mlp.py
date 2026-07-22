"""PyTorch MLP Volume Model — regression (predict delivery volume)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class VolumeMLP(nn.Module):
    """Simple 2-layer MLP for volume prediction."""

    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    dataloader: DataLoader,
    input_dim: int = 5,
    epochs: int = 5,
    lr: float = 1e-3,
) -> VolumeMLP:
    """Train the MLP model for a fixed number of epochs.

    Args:
        dataloader: PyTorch DataLoader yielding (features, targets) batches.
        input_dim: Number of input features.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Trained VolumeMLP model.
    """
    device = torch.device("cpu")  # CPU only for this benchmark
    model = VolumeMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    return model


def create_dataloader_from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """Create a standard PyTorch DataLoader from numpy arrays.

    This is the Parquet approach: load everything to memory, wrap in TensorDataset.
    """
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
