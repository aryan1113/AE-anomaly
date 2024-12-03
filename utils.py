import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def calculate_reconstruction_error(autoencoder, dataloader):
    autoencoder.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for batch in dataloader:
            reconstructed = autoencoder(batch)
            errors = torch.mean((batch - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(errors.tolist())
    return reconstruction_errors

def anomaly_detection(autoencoder, test_data, threshold):
    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder(test_data)
        errors = torch.mean((test_data - reconstructed) ** 2, dim=1)
    anomalies = errors > threshold
    confidence_scores = 1 / (1 + torch.exp(-(errors - threshold)))
    return anomalies, confidence_scores



def train_autoencoder(autoencoder, dataloader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

