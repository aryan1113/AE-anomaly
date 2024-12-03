import torch
import torch.nn as nn
import torch.optim as optim

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class AutoencoderWithAttention(nn.Module):
    def __init__(self, input_dim, architecture='1-layer'):
        super(AutoencoderWithAttention, self).__init__()
        self.architecture = architecture

        if architecture == '1-layer':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                AttentionBlock(input_dim // 2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )
        elif architecture == '3-layer':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                AttentionBlock(input_dim // 2),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),
                AttentionBlock(input_dim // 4)
            )
            self.decoder = nn.Sequential(
                nn.Linear(input_dim // 4, input_dim // 2),
                nn.ReLU(),
                AttentionBlock(input_dim // 2),
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )
        elif architecture == '5-layer':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 3 * input_dim // 4),
                nn.ReLU(),
                AttentionBlock(3 * input_dim // 4),
                nn.Linear(3 * input_dim // 4, input_dim // 2),
                nn.ReLU(),
                AttentionBlock(input_dim // 2),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),
                AttentionBlock(input_dim // 4)
            )
            self.decoder = nn.Sequential(
                nn.Linear(input_dim // 4, input_dim // 2),
                nn.ReLU(),
                AttentionBlock(input_dim // 2),
                nn.Linear(input_dim // 2, 3 * input_dim // 4),
                nn.ReLU(),
                AttentionBlock(3 * input_dim // 4),
                nn.Linear(3 * input_dim // 4, input_dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_dim = 20
architecture = '3-layer'
autoencoder = AutoencoderWithAttention(input_dim, architecture)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()



