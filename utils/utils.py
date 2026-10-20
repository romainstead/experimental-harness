import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset


def seed_everything(seed=0, num_threads=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def configure_mlp(n_layers: int, hidden: int, dropout: float):
    return torchvision.ops.MLP(
        in_channels=64,
        hidden_channels=[hidden] * n_layers + [10],
        activation_layer=torch.nn.ReLU,
        dropout=dropout,
    )


def train_model(model, num_epochs: int, lr: float, train_loader, val_loader, device="cpu"):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for digits, labels in train_loader:
            digits = digits.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(digits)
            _loss = loss(outputs, labels)
            _loss.backward()
            optimizer.step()
            train_loss += _loss.item()
        train_loss /= len(train_loader)
        model.eval()

        val_loss = 0
        for digits, labels in val_loader:
            digits = digits.to(device)
            labels = labels.to(device)
            outputs = model(digits)
            val_loss += loss(outputs, labels).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}\n train loss={train_loss:.4f}\n val loss={val_loss:.4f}')
    return model


def make_dataset(X, y, idxs):
    Xt = torch.tensor(X[idxs] / 16.0, dtype=torch.float32)
    yt = torch.tensor(y[idxs], dtype=torch.long)
    return TensorDataset(Xt, yt)
