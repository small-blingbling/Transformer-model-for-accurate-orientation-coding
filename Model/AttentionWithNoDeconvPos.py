
import torch.nn as nn
import math


from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from Embedding import Embedding
from unEmbedding import unEmbedding

class AttentionWithNoDeconvPos(nn.Module):
    def __init__(self, input_size, d_model , n_head, max_len, num_neurons, device, hidden):
        super(AttentionWithNoDeconvPos, self).__init__()
        self.embedding = Embedding(d_model,num_neurons)
        self.pos = PositionalEncoding(d_model, max_len, device)
        self.attention = MultiHeadAttention(d_model,n_head)
        self.ff = unEmbedding(d_model,num_neurons)
        self.ff1 = nn.Linear(input_size, hidden)


    def forward(self, x,  threshold, is_greater):

        x_pos = self.pos(x)
        x = x.unsqueeze(2)
        x = self.embedding(x)

        x = x + x_pos

        x , atten = self.attention(x,x,x, threshold = threshold, is_greater = is_greater)
        x = self.ff(x)
        x = x.squeeze()
        x_atten = x
        x = self.ff1(x)
        attention_weights = atten.squeeze()
        height = x.size(1)
        h = math.sqrt(height)
        x = x.reshape(-1, 1, int(h), int(h))

        return x, attention_weights, x_atten

