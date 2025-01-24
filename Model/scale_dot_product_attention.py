
import math
import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12, threshold = 0.00001, is_greater = True):

        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)

        col_means = torch.mean(score, dim=2)
        if is_greater:
            cols_to_keep = col_means > threshold
        else:
            cols_to_keep = col_means < threshold

        for i in range(score.size(0)):
            for j in range(score.size(1)):
                score[i, j, :, ~cols_to_keep[i, j]] = 0

        v = score @ v

        return v, score
