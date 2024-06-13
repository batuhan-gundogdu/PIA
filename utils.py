import torch
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import stats
import random
import os

def get_batch(batch_size=16, noise_sdt=0.1, b_values=[0, 150, 1000, 1500], TE_values=[0, 13, 93, 143]):

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
    # TODO: make v, D, T2 not a torch tensor, becuase they are never used in model training
    #return 1000*torch.from_numpy(signal*(1+noise)).float(),torch.from_numpy(D.T).float(), torch.from_numpy(T2.T).float(), torch.from_numpy(v.T).float(), 1000*torch.from_numpy(signal).float()
    return 1000*torch.from_numpy(signal+noise).float(),torch.from_numpy(D.T).float(), torch.from_numpy(T2.T).float(), torch.from_numpy(v.T).float(), 1000*torch.from_numpy(signal).float()


def hybrid_fit(signals, bvals = [0, 150, 1000, 1500], normTE = [0, 13, 93, 143]):
    eps = 1e-7;
    numcols, acquisitions = signals.shape
    D = np.zeros((numcols, 3))
    T2 = np.zeros((numcols, 3))
    v = np.zeros((numcols, 3))
    for col in range(numcols):
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

def get_scores(true, pred):
    v_test, D_test, T2_test = true
    v, D, T2 = pred
    print(f'\tCorr\tMAE\tBias\tStddev')  
    for r in range(3):
        if r==0:
            x_image, y_image = v_test, v
            title = ['V_ep', 'V st', 'V lu']
        elif r==1:
            x_image, y_image = D_test, D
            title = ['D ep', 'D st', 'D lu']
        elif r==2:
            x_image, y_image = T2_test, T2
            title = ['T2 ep', 'T2 st', 'T2 lu']
        for c in range(3):

            if isinstance(x_image, torch.Tensor):
                x = x_image.detach().cpu().numpy()[:,c]
            else:
                x = x_image[:, c]
            if isinstance(y_image, torch.Tensor):
                y = y_image[:, c].detach().cpu().numpy()
            else: 
                y = y_image[:, c]

            corr = round(spearmanr(y, x).statistic, 2)
            mae = np.mean(np.abs(y - x))
            bias = np.mean((y - x))
            std = np.std((y - x))
            print(f'{title[c]}\t{corr:.2f}\t{mae:.2f}\t{bias:.2f}\t{std:.2f}')   



def plot_results(true, pred, method='NLLS'):
    v_test, D_test, T2_test = true
    v, D, T2 = pred
    fig, ax = plt.subplots(3,3, figsize=(25,25))

    for r in range(3):
        if r==0:
            x_image, y_image = v_test, v
            title = ['V_ep', 'V st', 'V lu']
            ylims = [(0,1), (0,1), (0,1)]
        elif r==1:
            x_image, y_image = D_test, D
            title = ['D ep', 'D st', 'D lu']
            ylims = [(0.3, 0.7), (0.7, 1.7), (2.7, 3)]
        elif r==2:
            x_image, y_image = T2_test, T2
            title = ['T2 ep', 'T2 st', 'T2 lu']
            ylims = [(20, 70), (40, 100), (500, 1000)]
        for c in range(3):
            if isinstance(x_image, torch.Tensor):
                x = x_image.detach().cpu().numpy()[:,c]
            else:
                x = x_image[:, c]
            if isinstance(y_image, torch.Tensor):
                y = y_image[:, c].detach().cpu().numpy()
            else: 
                y = y_image[:, c]
            nbins=300
            k = gaussian_kde([x,y])
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            ax[r,c].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="hot", shading='auto')
            ax[r,c].scatter(x,y, color='white', s=4, alpha=0.5)

            ax[r,c].xaxis.set_tick_params(labelsize=20)
            ax[r,c].yaxis.set_tick_params(labelsize=20)
            ax[r,c].set_title(fr'{title[c]}', fontsize=24)
            ax[r,c].set_xlabel('true', fontsize=24)
            ax[r,c].set_ylabel('predicted', fontsize=24)
    plt.savefig(f'plots/scatter_{method}.png')
    plt.close(fig)

def get_scores2(test, pia, hybrid, inx):    
    x1 = test.detach().cpu().numpy()[:,inx] 
    y1 = pia[:, inx].detach().cpu().numpy()
    corr1 = round(spearmanr(y1, x1).statistic, 2)
    mae1 = np.mean(np.abs(y1 - x1))
    bias1 = np.mean((y1 - x1))
    std1 = np.std((y1 - x1))
    x2 = test.detach().cpu().numpy()[:, inx] 
    y2 = hybrid[:, inx]
    corr2 = round(spearmanr(y2, x2).statistic, 2)
    mae2 = np.mean(np.abs(y2 - x2))
    bias2 = np.mean((y2 - x2))
    std2 = np.std((y2 - x2))
    return  (corr1, corr2), (mae1,mae2), (bias1,bias2), (std1,std2)

def steigers_z_test(r1, r2, n1, n2):
    """
    Performs Steiger's Z-test for two dependent correlation coefficients sharing one variable in common.

    Args:
    r1 (float): Pearson correlation coefficient for the first comparison.
    r2 (float): Pearson correlation coefficient for the second comparison.
    n1 (int): Sample size for the first comparison.
    n2 (int): Sample size for the second comparison.

    Returns:
    float: Z-score indicating the difference between the two correlation coefficients.
    float: p-value assessing the significance of the Z-score.
    """
    # Fisher Z transformation for each correlation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error for each transformed correlation
    se1 = 1 / np.sqrt(n1 - 3)
    se2 = 1 / np.sqrt(n2 - 3)

    # Standard error of the difference
    sed = np.sqrt(se1**2 + se2**2)

    # Z-score
    z = (z1 - z2) / sed

    # Two-tailed p-value
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    return z, p

def calculate_mae_bias_variance(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MSE) and Bias.

    Args:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted or estimated values.

    """
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean((y_pred - y_true))
    variance = np.std((y_pred - y_true))**2

    return mae, bias, variance

def compare_mae(y_true, y_pred1, y_pred2):

    mae1, bias1, variance1 = calculate_mae_bias_variance(y_true, y_pred1)
    mae2, bias2, variance2 = calculate_mae_bias_variance(y_true, y_pred2)

    # Perform paired t-test
    _, p_value = stats.ttest_rel(np.abs(y_true - y_pred1), np.abs(y_true - y_pred2))
    _, p_value2 = stats.ttest_rel(y_true - y_pred1, y_true - y_pred2)

    # Calculate variances
    var_a = variance1**2
    var_b = variance2**2

    # Calculate F statistic
    F = var_a / var_b
    df1 = len(y_true) - 1  # degrees of freedom for sample 1
    df2 = len(y_true) - 1  # degrees of freedom for sample 2

    # Calculate p-value
    p_value3 = 1 - stats.f.cdf(F, df1, df2) if var_a > var_b else stats.f.cdf(F, df1, df2)

    return p_value, p_value2, p_value3

def calculate_stats(test, pia, NLLS):
    v_test, D_test, T2_test = test
    v, D, T2 = pia
    v_NLLS, D_NLLS, T2_NLLS = NLLS
    tissue = ['ep', 'st', 'lu']
    print(f'\tCorr\tMAE\tBias\tStddev')  
    for inx in range(3):
        x1 = v_test[:,inx].detach().cpu().numpy()
        y1 = v[:, inx].detach().cpu().numpy()
        y2 = v_NLLS[:, inx]
        r_pia = spearmanr(x1, y1).statistic
        r_nlls = spearmanr(x1, y2).statistic
        p_mae, p_bias, p_var  = compare_mae(x1, y1, y2)
        p_corr = steigers_z_test(r_nlls, r_pia, v_NLLS.shape[0], v_NLLS.shape[0])[1]
        print(f'v_{tissue[inx]}\t{p_corr:.5f}\t{p_mae:.5f}\t{p_bias:.5f}\t{p_var:.5f}')
    for inx in range(3):
        x1 = D_test[:,inx].detach().cpu().numpy()
        y1 = D[:, inx].detach().cpu().numpy()
        y2 = D_NLLS[:, inx]
        r_pia = spearmanr(x1, y1).statistic
        r_nlls = spearmanr(x1, y2).statistic
        p_mae, p_bias, p_var  = compare_mae(x1, y1, y2)
        p_corr = steigers_z_test(r_nlls, r_pia, v_NLLS.shape[0], v_NLLS.shape[0])[1]
        print(f'D_{tissue[inx]}\t{p_corr:.5f}\t{p_mae:.5f}\t{p_bias:.5f}\t{p_var:.5f}')
    for inx in range(3):
        x1 = T2_test[:,inx].detach().cpu().numpy()
        y1 = T2[:, inx].detach().cpu().numpy()
        y2 = T2_NLLS[:, inx]
        r_pia = spearmanr(x1, y1).statistic
        r_nlls = spearmanr(x1, y2).statistic
        p_mae, p_bias, p_var  = compare_mae(x1, y1, y2)
        p_corr = steigers_z_test(r_nlls, r_pia, v_NLLS.shape[0], v_NLLS.shape[0])[1]
        print(f'T2_{tissue[inx]}\t{p_corr:.5f}\t{p_mae:.5f}\t{p_bias:.5f}\t{p_var:.5f}')

def get_batch_2compartment(batch_size=16, noise_sdt=0.1, b_values=[0, 150, 1000, 1500], TE_values=[0, 13, 93, 143]):
    
    " Sample from 2-compartment model with 2 tissues"

    b_TE = []
    for b in b_values:
        for TE in TE_values:
            b_TE.append((b,TE))
    
    D_low = np.random.uniform(0.3, 0.7, batch_size)
    D_high = np.random.uniform(2.7, 3, batch_size)
    T2_low = np.random.uniform(20, 70, batch_size)
    T2_high = np.random.uniform(500, 1000, batch_size)
    
    v_low = np.random.uniform(0, 1, batch_size)
    v_high = np.random.uniform(0, 1, batch_size)
    
    sum_abc = v_low + v_high
    
    v_low = v_low/sum_abc
    v_high = v_high/sum_abc
     

    signal = np.zeros((batch_size, len(b_TE)), dtype=float)
    for sample in range(batch_size):
        for ctr, (b, TE) in enumerate(b_TE):
            S_low = v_low[sample]*np.exp(-b/1000*D_low[sample])*np.exp(-TE/T2_low[sample])
            S_high = v_high[sample]*np.exp(-b/1000*D_high[sample])*np.exp(-TE/T2_high[sample])
            signal[sample, ctr] = S_low + S_high

    D = np.asarray([D_low, D_high])
    T2 = np.asarray([T2_low, T2_high])
    v = np.asarray([np.asarray(v_low), np.asarray(v_high)])
    noise = np.random.normal(0, noise_sdt, signal.shape)
    # TODO: make v, D, T2 not a torch tensor, becuase they are never used in model training
    #return 1000*torch.from_numpy(signal*(1+noise)).float(),torch.from_numpy(D.T).float(), torch.from_numpy(T2.T).float(), torch.from_numpy(v.T).float(), 1000*torch.from_numpy(signal).float()
    return 1000*torch.from_numpy(signal+noise).float(),torch.from_numpy(D.T).float(), torch.from_numpy(T2.T).float(), torch.from_numpy(v.T).float(), 1000*torch.from_numpy(signal).float()

def set_seed(seed):
    # TODO add mps seed setting
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc