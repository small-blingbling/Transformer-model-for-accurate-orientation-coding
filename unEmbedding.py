import torch
import torch.nn as nn

class unEmbedding(nn.Module):
    def __init__(self, d_model,num_neurons):

        super(unEmbedding, self).__init__()
        # self.embedding = nn.Linear(d_model,1)
        self.embedding_layers = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_neurons)])


    def forward(self, x):
        # x = self.embedding(x)
        x = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.embedding_layers)], dim=1)
        return x


