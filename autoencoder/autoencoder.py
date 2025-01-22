# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import sys
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff):
        super(TransformerModel, self).__init__()

        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

        self.encoder_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads) for _ in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads) for _ in range(num_layers)]
        )

        self.encoder_layers2 = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.LayerNorm(dff)) for _ in range(num_layers)]
        )

        self.decoder_layers2 = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.LayerNorm(dff)) for _ in range(num_layers)]
        )
        self.encoder_ffns = nn.ModuleList(
            [nn.Sequential(nn.Linear(dff, d_model)) for _ in range(num_layers)]
        )

        self.decoder_ffns = nn.ModuleList(
            [nn.Sequential(nn.Linear(dff, d_model)) for _ in range(num_layers)]
        )

    def forward(self, x):

        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x_fc = F.relu(self.fc1(lstm_out))
        mask_params = torch.sigmoid(self.fc2(x_fc))

        tau = 1
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(mask_params.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + mask_params) / tau
        mask = torch.sigmoid(gate_inputs).squeeze()

        x = x.squeeze(1) * mask


        for attn, attn2, ffn in zip(self.encoder_layers, self.encoder_layers2, self.encoder_ffns):
            x1 = attn(x, x, x)[0]
            x2 = attn2(x1)
            x = ffn(x2) + x
        # [0]表示取加权值，[1]表示取注意力权重

        encoderout = x

        for attn, attn2, ffn in zip(self.decoder_layers, self.decoder_layers2, self.decoder_ffns):
            x1 = attn(x, x, x)[0]
            x2 = attn2(x1)
            x = ffn(x2) + x

        decoderout = x

        return encoderout, decoderout

class LossFn(nn.Module):
    def __init__(self):
        super(LossFn, self).__init__()

    def forward(self, y_true, y_pred):
        loss = torch.mean(torch.square(y_true - y_pred))
        return loss


def train_model(model, data_all, epochs, batch_size):

    model = model.to(device)
    data_all = data_all.to(device)

    criterion = LossFn()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        num_batches = len(data_all) // batch_size
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size

            optimizer.zero_grad()
            encoderout, decoderout = model(data_all[batch_start:batch_end])
            loss = criterion(decoderout, data_all[batch_start:batch_end])

            if torch.isnan(loss).any():
                print("Loss is NaN. Terminating the program.")
                break

            if best_loss > loss:
                best_loss = loss
                best_model_state = deepcopy(model.state_dict())
                best_model_state_epoch = epoch

            loss.backward()
            optimizer.step()

        print("Epoch:", epoch, "Train Loss:", loss.item())

    print(f"Best model at epoch {best_model_state_epoch} with loss {best_loss}")
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        encoderout, decoderout = model(data_all[:4000])
        encoderout = encoderout.cpu().detach().numpy()
        np.savetxt("encoderoutmdd_nor.txt", encoderout, fmt="%lf")




file_path = r"../simsTxt/sim1.txt"
data_all = np.loadtxt(file_path, delimiter=' ')
data_all = torch.from_numpy(data_all).float()
n_nodes = data_all.shape[1]

num_layers = 4
d_model = n_nodes
num_heads = n_nodes
dff = 64

model = TransformerModel(num_layers, d_model, num_heads, dff)
train_model(model, data_all,epochs=160, batch_size=32)

