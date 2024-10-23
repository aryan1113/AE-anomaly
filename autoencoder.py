# dataset curation / collection is in process
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # data is in input layer, enters into the Encoder

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  
        )
        
        # data is in Latent Space, enters into Decoder henceforth

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim), 
            nn.Sigmoid()  
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, dataloader, num_epochs=50, learning_rate=0.001):

    ''' 
    fiddle with criterion 

    '''
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            inputs = data[0]
            
            # reset gradients 
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # loss for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

# Sample Usage
if __name__ == "__main__":
    # will change as per data
    input_dim = 10
    
    # dummy
    X_train = torch.rand(1000, input_dim) 

    # Prepare the dataset and DataLoader
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Instantiate the model
    autoencoder = Autoencoder(input_dim=input_dim)

    train_autoencoder(autoencoder, dataloader)
