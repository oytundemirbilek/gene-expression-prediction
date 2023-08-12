import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int = 5, dropout: float = 0.1, max_len: int = 112
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [112,1] = 0,2,...111
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [5] = 0,2,4
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # [112,30]
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # [112,30]

        pe = torch.permute(pe, (1, 2, 0))
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # pe = torch.transpose(self.pe, 1, 2) [1024, 5, 512] [1, 512, 5]
        x = x + self.pe
        return self.dropout(x)


class PromoterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=5, out_channels=32, kernel_size=16, stride=1
        )  # Feature Maps: (97, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (48, 64)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=12, stride=1
        )  # Feature Maps: (37, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (18, 128)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=8, stride=1
        )  # Feature Maps: (11, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Feature Maps: (5, 256)

        # d_model = size of input embeddings, nhead = the number of parallel heads, dk = dv = d_model / nhead
        # One sub_attention_fc module takes an input of NxLxD_model shape and produces output of NxLxD_model shape
        # Input(sub_attention_fc) = Nx5x256, Output(sub_attention_fc) = Nx5x256
        # Input(transformer_encoder) = Nx5x256, Output(transformer_encoder) = Nx5x256
        self.pe = PositionalEncoding(d_model=128, dropout=0.1, max_len=33)
        sub_attention_fc = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(sub_attention_fc, num_layers=3)

        self.flatten = nn.Flatten()  # Feature Vectors: (33 * 64, 1) = (2112, 1)
        # self.fc1 = nn.Linear(1280, 1)
        # self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(1280, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x = F.relu(self.conv1(x))#.transpose(2,1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # feature_maps_all.shape = Batch x channel x signal
        # transformer input format = Batch x signal x channel
        # WARNING: feature_maps_all should pass through positional encoding first
        x = torch.permute(x, dims=(0, 2, 1))
        # x = self.pe(x).transpose(2,1)
        x = self.transformer_encoder(x)

        batch_size = x.shape[0]  # 1024
        half = int(batch_size / 2)
        x_org, x_comp = x[:half], x[half:]
        x_all = torch.cat((x_org, x_comp), dim=1)  # 512
        # transformer output format = Batch x signal x channel
        # global max pooling input format = Batch x channel x signal
        x_all = self.flatten(x_all)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x_all = self.fc3(x_all)
        return torch.squeeze(x_all)
