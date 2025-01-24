
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, d_model, num_neurons):
        super(Embedding, self).__init__()
        # self.embedding = nn.Linear(1,d_model)
        self.embedding_layers = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_neurons)])

    def forward(self, x):
        # x = self.embedding(x)
        x = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.embedding_layers)], dim=1)
        return x

