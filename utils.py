import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def compute_anomaly_confidence(autoencoder, sample, max_reconstruction_error):
    with torch.no_grad():
        reconstruction = autoencoder(sample)
        reconstruction_error = torch.mean(torch.pow(sample - reconstruction, 2))
    
    confidence_score = torch.sigmoid(reconstruction_error - max_reconstruction_error)

    is_anomaly = reconstruction_error > max_reconstruction_error
    
    if not is_anomaly:
        confidence_score = 1 - confidence_score
    
    return confidence_score.item(), is_anomaly.item()

def train_anomaly_detection_autoencoder(train_loader, test_loader, input_dim, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = AutoencoderWithAttention(input_dim).to(device)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        autoencoder.train()
        train_loss = 0
        reconstruction_errors = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = autoencoder(batch)
            batch_errors = torch.mean(criterion(outputs, batch), dim=1)
            reconstruction_errors.extend(batch_errors.detach().cpu().numpy())
            
            loss = torch.mean(batch_errors)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        max_reconstruction_error = np.max(reconstruction_errors)
        
        autoencoder.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in test_loader:
                batch = batch.to(device)
                outputs = autoencoder(batch)
                val_loss += torch.mean(criterion(outputs, batch)).item()
            
            val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Max Reconstruction Error: {max_reconstruction_error:.4f}')
    
    return autoencoder, max_reconstruction_error
