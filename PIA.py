import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List


class PIA(nn.Module):

    def __init__(self,
                number_of_signals=16,
                D_mean = [0.5, 1.2, 2.85],
                T2_mean = [45, 70, 750],
                D_delta = [0.2, 0.5, 0.15],
                T2_delta = [25, 30, 250],
                b_values = [0, 150, 1000, 1500],
                TE_values = [0, 13, 93, 143],
                hidden_dims: List = None,
                predictor_depth=1,
                device='cuda'):
        super(PIA, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.number_of_signals = number_of_signals
        self.number_of_compartments = 3
        self.D_mean = torch.from_numpy(np.asarray(D_mean)).to(device)
        self.T2_mean = torch.from_numpy(np.asarray(T2_mean)).to(device)
        self.D_delta = torch.from_numpy(np.asarray(D_delta)).to(device)
        self.T2_delta = torch.from_numpy(np.asarray(T2_delta)).to(device)
        self.b_values = b_values
        self.TE_values = TE_values
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.device = device

        

        modules = []
        # Build Encoder
        in_channels = number_of_signals
        for h_dim in hidden_dims:
            
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules).to(device)

        D_predictor = []
        for _ in range(predictor_depth):
            
            D_predictor.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                    nn.LeakyReLU())
            )
        D_predictor.append(nn.Linear(hidden_dims[-1], self.number_of_compartments))
        self.D_predictor = nn.Sequential(*D_predictor).to(device)


        T2_predictor = []
        for _ in range(predictor_depth):
            
            T2_predictor.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                    nn.LeakyReLU())
            )
        T2_predictor.append(nn.Linear(hidden_dims[-1], self.number_of_compartments))
        self.T2_predictor = nn.Sequential(*T2_predictor).to(device)

        v_predictor = []
        for _ in range(predictor_depth):
            
            v_predictor.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                    nn.LeakyReLU())
            )
        v_predictor.append(nn.Linear(hidden_dims[-1], self.number_of_compartments))
        self.v_predictor = nn.Sequential(*v_predictor).to(device)
        


    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        
        D_var = self.D_delta*torch.tanh(self.D_predictor(result))
        T2_var = self.T2_delta*torch.tanh(self.T2_predictor(result))
        v = self.softmax(self.v_predictor(result))
        
        return [self.D_mean + D_var, self.T2_mean + T2_var, v]

    def decode(self, D, T2, v):
        """
            Maps the given latent codes onto the signal space.
            param D: [D_ep, D_st, D_lu]
            param T2: [T2_ep, T2_st, T2_lu]
            param v: [v_ep, v_st, v_lu]
            return: (Tensor) signal estimate
        """
        signal = torch.zeros((D.shape[0], self.number_of_signals))
        D, T2, v = D.T, T2.T, v.T
        ctr = 0
        for b in self.b_values:
            for TE in self.TE_values:
                S_ep = v[0]*torch.exp(-b/1000*D[0])*torch.exp(-TE/T2[0])
                S_st = v[1]*torch.exp(-b/1000*D[1])*torch.exp(-TE/T2[1])
                S_lu = v[2]*torch.exp(-b/1000*D[2])*torch.exp(-TE/T2[2])
                signal[:, ctr] = S_ep + S_st + S_lu
                ctr += 1
        return (1000*signal).to(self.device)
    
    
    def forward(self, x):
        D, T2, v = self.encode(x)
        return  [self.decode(D, T2, v), x, D, T2, v]


    def loss_function(self, pred_signal, true_signal, weights=None):

        if weights is not None:
            loss = torch.mean(weights * (pred_signal - true_signal) ** 2)
        else:
            loss = F.mse_loss(pred_signal, true_signal)

        return loss
