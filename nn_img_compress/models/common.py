import torch
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class BCEFLLoss:

    def __init__(self, gamma=0, alpha=0.5):
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def __call__(self, outputs, targets):
        if self.alpha.type() != outputs.data.type():
            self.alpha = self.alpha.type_as(outputs.data)
        if self.gamma.type() != outputs.data.type():
            self.gamma = self.gamma.type_as(outputs.data)

        self.alpha = self.alpha.to(outputs.device)
        self.gamma = self.gamma.to(outputs.device)

        return -targets * self.alpha * (1 - outputs) ** self.gamma * torch.log(outputs) \
               - (1 - targets) * (1 - self.alpha) * outputs ** self.gamma * torch.log(1 - outputs)


class PositionalWiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=True):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output
