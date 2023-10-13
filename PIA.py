import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import time
from tqdm import tqdm
from scipy.optimize import curve_fit



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
                predictor_depth=1):
        super(PIA, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        return 1000*signal
    
    
    def forward(self, x):
        D, T2, v = self.encode(x)
        return  [self.decode(D, T2, v), x, D, T2, v]


    def loss_function(self, recons, x, PIDS, tissue_available=False):
        if tissue_available:
            pred_signal, pred_D, pred_T2, pred_v = recons
            true_signal, true_D, true_T2, true_v = x

            loss_signal = F.mse_loss(pred_signal, true_signal)
            loss_D = F.mse_loss(pred_D, true_D)
            loss_T2 = F.mse_loss(pred_T2, true_T2)
            loss_v = F.kl_div(pred_v, true_v)
            loss = loss_signal + loss_D + 0.0001*loss_T2 + 0.2*loss_v

        else:
            pred_signal = recons
            true_signal = x
            #loss = F.mse_loss(pred_signal, true_signal)
            loss = torch.mean(PIDS * (pred_signal - true_signal) ** 2)

        return loss

def ADC_slice(bvalues, slicedata):
    min_adc = 0
    max_adc = 3.0
    eps = 1e-7
    numrows, numcols, numbvalues = slicedata.shape
    adc_map = np.zeros((numrows, numcols))
    for row in range(numrows):
        for col in range(numcols):
            ydata = np.squeeze(slicedata[row,col,:])
            adc = np.polyfit(bvalues.flatten()/1000, np.log(ydata + eps), 1)
            adc = -adc[0]
            adc_map[row, col] =  max(min(adc, max_adc), min_adc)
    return adc_map

def get_batch(batch_size=16, noise_sdt=0.1):

    b_values = [0, 150, 1000, 1500]
    TE_values = [0, 13, 93, 143]

    b_TE = []
    for b in b_values:
        for TE in TE_values:
            b_TE.append((b,TE))
    
    D_ep = np.random.uniform(0.3, 0.7, batch_size)
    D_st = np.random.uniform(0.7, 1.7, batch_size)
    D_lu = np.random.uniform(2.7, 3, batch_size)
    T2_ep = np.random.uniform(20, 70, batch_size)
    T2_st = np.random.uniform(40, 100, batch_size)
    T2_lu = np.random.uniform(500, 1000, batch_size)
    
    v_ep = np.random.uniform(0, 1, batch_size)
    v_st = np.random.uniform(0, 1, batch_size)
    v_lu = np.random.uniform(0, 1, batch_size)
    
    sum_abc = v_ep + v_st + v_lu
    
    v_ep = v_ep/sum_abc
    v_st = v_st/sum_abc
    v_lu = v_lu/sum_abc
     

    signal = np.zeros((batch_size, len(b_TE)), dtype=float)
    for sample in range(batch_size):
        for ctr, (b, TE) in enumerate(b_TE):
            S_ep = v_ep[sample]*np.exp(-b/1000*D_ep[sample])*np.exp(-TE/T2_ep[sample])
            S_st = v_st[sample]*np.exp(-b/1000*D_st[sample])*np.exp(-TE/T2_st[sample])
            S_lu = v_lu[sample]*np.exp(-b/1000*D_lu[sample])*np.exp(-TE/T2_lu[sample])
            signal[sample, ctr] = S_ep + S_st + S_lu

    D = np.asarray([D_ep, D_st, D_lu])
    T2 = np.asarray([T2_ep, T2_st, T2_lu])
    v = np.asarray([np.asarray(v_ep), np.asarray(v_st), np.asarray(v_lu)])
    noise = np.random.normal(0, noise_sdt, signal.shape)
    
    #return 1000*torch.from_numpy(signal*(1+noise)).float(),torch.from_numpy(D.T).float(), torch.from_numpy(T2.T).float(), torch.from_numpy(v.T).float(), 1000*torch.from_numpy(signal).float()
    return 1000*torch.from_numpy(signal+noise).float(),torch.from_numpy(D.T).float(), torch.from_numpy(T2.T).float(), torch.from_numpy(v.T).float(), 1000*torch.from_numpy(signal).float()


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    #cbar.ax.set_ylabel('Density')


def three_compartment_fit(M, D_ep, D_st, D_lu, T2_ep,  T2_st, T2_lu, V_ep, V_st):
    """
    
    Three-compartment fit for Hybrid estimation
    
    """
    b, TE = M
    S_ep = V_ep*np.exp(-b/1000*D_ep)*np.exp(-TE/T2_ep)
    S_st = V_st*np.exp(-b/1000*D_st)*np.exp(-TE/T2_st)
    S_lu =(1 - V_ep - V_st)*np.exp(-b/1000*D_lu)*np.exp(-TE/T2_lu)
    
    return 1000*(S_ep + S_st + S_lu)

def hybrid_fit(signals):
    bvals = [0, 150, 1000, 1500]
    normTE = [0, 13, 93, 143]
    eps = 1e-7;
    numcols, acquisitions = signals.shape
    D = np.zeros((numcols, 3))
    T2 = np.zeros((numcols, 3))
    v = np.zeros((numcols, 3))
    for col in tqdm(range(numcols)):
        voxel = signals[col]
        X, Y = np.meshgrid(normTE, bvals)
        xdata = np.vstack((Y.ravel(), X.ravel()))
        ydata = voxel.ravel()
        try:
            fitdata_, _  = curve_fit(three_compartment_fit, 
                                       xdata,
                                       ydata,
                                       p0 = [0.55, 1.3, 2.8, 50,  70, 750, 0.3, 0.4],
                                       check_finite=True,
                                       bounds=([0.3, 0.7, 2.7, 20,  40, 500, 0, 0],
                                               [0.7,  1.7, 3.0, 70,  100, 1000,1, 1]),
                                      method='trf',
                                      maxfev=5000)
        except RuntimeError:
            fitdata_ = [0.55,  1.3, 2.8, 50,  70, 750, 0.3, 0.4]
        coeffs = fitdata_
        D[col, :] = coeffs[0:3]
        T2[col, :] = coeffs[3:6]
        v[col, 0:2] = coeffs[6:]
        v[col, 2]  = 1 - coeffs[6] - coeffs[7]
    return D, T2, v


def detect_PIDS_slice(b, S):
    """ Inputs: b - diffusion weight values used in image
                S - Hybrid Multi-dimensional image
        Outputs:
                PIDS_ADC1 : Binary Map with voxels ADC > 3 (could mean motion induced signal loss at high-b)
                PIDS_ADC2 : Binary Map with voxels ADC < 0 (could mean the voxel is below the noise level)
                PIDS_b_decay : Binary Map with voxels disobeying decay rule along b direction
                PIDS_TE_decay : Binary Map with voxels disobeying decay rule along TE direction
    """
    
    eps = 1e-7
    localize = np.eye(4)
    num_rows, num_cols, num_bvalues, num_TEs = S.shape
    PIDS_ADC1 = np.zeros((num_rows, num_cols))
    PIDS_ADC2 = np.zeros((num_rows, num_cols))
    PIDS_b_decay = np.zeros((num_rows, num_cols, num_TEs, 3))
    PIDS_TE_decay = np.zeros((num_rows, num_cols, num_bvalues, 3))
    for row in tqdm(range(num_rows)):
         for col in range(num_cols):
            te0 = np.squeeze(S[row,col, :, 0])
            adc = np.polyfit(b.flatten()/1000, np.log(te0 + eps), 1)
            adc = -adc[0]    
            PIDS_ADC1[row, col] = int(adc > 3)
            PIDS_ADC2[row, col] = int(adc < 0)
            for _b in range(num_bvalues):
                signals_along_te = np.squeeze(S[row,col, _b, :])
                to_compare = signals_along_te.copy().astype(int)
                to_compare[1:] = signals_along_te[:3]
                is_pids = signals_along_te - to_compare
                for local in range(3):
                    is_pids_ = int(is_pids[local + 1]>=0)
                    PIDS_TE_decay[row, col, _b, local] = is_pids_
            for _te in range(num_TEs):
                signals_along_b = np.squeeze(S[row,col, :, _te])
                to_compare = signals_along_b.copy().astype(int)
                to_compare[1:] = signals_along_b[:3]
                is_pids = signals_along_b - to_compare
                for local in range(3):
                    is_pids_ = int(is_pids[local + 1]>=0)
                    PIDS_b_decay[row, col, _te, local] = is_pids_

    return PIDS_ADC1, PIDS_ADC2, PIDS_b_decay, PIDS_TE_decay

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'