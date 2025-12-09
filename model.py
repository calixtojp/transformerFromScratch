import torch
from torch import nn
import math

class InputEmbeddings(nn.Module):
    def __init__(
            self,
            d_model: int, # dimensão do modelo ()
            voc_size: int # qtd de palavras no meu vocabulário
    ):
        self.d_model = d_model
        self.voc_size = voc_size
        self.embedding = nn.Embedding(voc_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(seld.d_model)

print("tá funcionando")