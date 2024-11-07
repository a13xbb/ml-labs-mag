import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    
class Conv1DNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Conv1DNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            
            nn.Flatten(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    
class LSTM(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_classes: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, num_classes)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_batch) -> torch.Tensor:
        embeddings = self.embedding(input_batch)

        output, _ = self.lstm(embeddings)
        output = output.max(dim=1)[0]
        output = self.activation(output)
    
        output = self.linear_1(output)
        output = self.dropout(output)
        output = self.activation(output)

        output = self.linear_2(output)
        return output
