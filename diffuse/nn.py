from typing import List, Optional, Type
import torch
from torch import nn
from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential, BatchNorm1d
from diffuse.transformer import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
import numpy as np
import torch.nn.functional as F


class DiffAtt(Module):
    def __init__(self, n_nodes: int,small_layer: int) -> None:
        super().__init__()
        big_layer=2048
        num_head=32
        self.main_block = nn.Sequential(
            nn.Linear(n_nodes + 1, small_layer, bias=False),
            nn.LeakyReLU(),
            nn.LayerNorm([small_layer]),
            nn.Dropout(0.2)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=small_layer, num_heads=num_head, dropout=0.2)

        self.transformer_decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=small_layer, nhead=num_head, dim_feedforward=big_layer),
            num_layers=4
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv1d(in_channels=small_layer, out_channels=small_layer, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=small_layer, out_channels=small_layer, kernel_size=5, padding=2)
        )

        self.output_layer = nn.Linear(small_layer, n_nodes)

        self.timeWeight = nn.Sequential(
            nn.Linear(1, small_layer, bias=False),
            nn.LeakyReLU(),
            nn.Linear(small_layer, small_layer, bias=False)
        )

        self.fusion_layers = nn.ModuleList([nn.Linear(small_layer, small_layer) for _ in range(3)])

    def forward(self, x_t, t, condition, t_condition):
        x_tT = torch.cat([x_t, t.unsqueeze(1)], axis=1)
        conditionT = torch.cat([condition, t_condition.unsqueeze(1)], axis=1)

        output_x = self.main_block(x_tT)
        output_condition = self.main_block(conditionT)

        globalFeature = self.self_attention(output_x, output_x, output_x)[0]

        localFeature = output_x.unsqueeze(-1)
        localFeature = self.conv_decoder(localFeature)
        localFeature = localFeature.squeeze(-1)

        timeWeight = self.timeWeight(t.unsqueeze(1))
        globalWeight = torch.sigmoid(timeWeight)*3
        globalWeight = torch.clamp(globalWeight, max=1.0)
        localWeight = 1 - globalWeight

        for fusion_layer in self.fusion_layers:
            fused_features = globalFeature * globalWeight + localFeature * localWeight
            fused_features = F.relu(fusion_layer(fused_features))
            localFeature = fused_features

        output = self.transformer_decoder(fused_features, output_condition)[0]

        output = self.output_layer(output)

        return output

