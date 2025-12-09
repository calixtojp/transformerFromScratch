import torch
from torch import nn
import math

class InputEmbeddings(nn.Module):
    def __init__(
            self,
            d_model: int, # dimensão do modelo ()
            voc_size: int # qtd de palavras no meu vocabulário
    ):
        super().__init__()
        self.d_model = d_model
        self.voc_size = voc_size
        self.embedding = nn.Embedding(voc_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(seld.d_model)

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,  # dimensão do modelo
        seq_len: int, # tamanho máximo da sequência
        dropout: float, # dropout para configurar o grau de overfitting
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # criar uma matriz no formato (seq_len, d_model)
        PE = torch.zeros(seq_len, d_model)

        # criar um vetor do tamanho da sequência
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1) # -> shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model)

        # aplicar seno para as posições pares e cosseno para as ímpares
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)

        PE = PE.unsqueeze(0) # (1, seq_len, d_model)

        # gravar esse tensor no buffer do modelo
        self.register_buffer('PE', PE)

    def forward(self, x):
        # aplicar a função para que calcula o positional encodding 
        # para cada token da sequência
        x = x + (
            self.PE[:, :x.shape[1], :]
        ).requires_grad_(False) # O PE vai ser fixo, não vai ser aprendido
        # no processo de treinamento

        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = self.Parameter(torch.ones(1)) # multiplicador aprendido
        self.bias = self.Parameter(torch.zeros(1)) # viés aprendido

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) 
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



print("tá funcionando")