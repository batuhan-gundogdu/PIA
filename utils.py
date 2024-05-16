import torch
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

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


def three_compartment_fit(M, D_ep, D_st, D_lu, T2_ep,  T2_st, T2_lu, V_ep, V_st):
    """
    
    Three-compartment fit for Hybrid estimation
    
    """
    b, TE = M
    S_ep = V_ep*np.exp(-b/1000*D_ep)*np.exp(-TE/T2_ep)
    S_st = V_st*np.exp(-b/1000*D_st)*np.exp(-TE/T2_st)
    S_lu =(1 - V_ep - V_st)*np.exp(-b/1000*D_lu)*np.exp(-TE/T2_lu)
    
    return 1000*(S_ep + S_st + S_lu)

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


