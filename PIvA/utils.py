import numpy as np
import torch
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.stats import gaussian_kde

def generate(D, T2, 
             b_values = [0, 150, 1000, 1500], 
             TE_values = [0, 13, 93, 143],
             noise_std=0.05):
    

    b_TE = []
    for b in b_values:
        for TE in TE_values:
            b_TE.append((b,TE))

    S = np.zeros((len(b_values)*len(TE_values)))

    for j, (b, TE) in enumerate(b_TE):
        S[j] = np.exp(-b/1000 * D) * np.exp(-TE / (T2*100))

    noise_im = np.random.normal(0, noise_std, S.shape)
    noise_re = np.random.normal(0, noise_std, S.shape)
    noisy = np.sqrt((S + noise_re)**2 + (noise_im)**2)

    return torch.from_numpy(noisy).float(), torch.from_numpy(S).float()

def get_batch_mono2D(batch_size=16, 
                     b_values = [0, 150, 1000, 1500],
                     TE_values = [0, 13, 93, 143],
                     noise_std=0.1):
 
    b_TE = []
    for b in b_values:
        for TE in TE_values:
            b_TE.append((b,TE))

    D = np.random.uniform(0.3, 3.0, size=batch_size)
    T2 = np.random.uniform(0.2, 7.5, size=batch_size)

    S = np.zeros((batch_size, len(b_values)*len(TE_values)))

    for i in range(batch_size):
        for j, (b, TE) in enumerate(b_TE):
            S[i, j] = np.exp(-b/1000 * D[i]) * np.exp(-TE / (T2[i]*100))

    # Rician Noise
    noise_im = np.random.normal(0, noise_std, S.shape)
    noise_re = np.random.normal(0, noise_std, S.shape)
    noisy = np.sqrt((S + noise_re)**2 + (noise_im)**2)

    return torch.from_numpy(noisy).float(), torch.from_numpy(D).float(), torch.from_numpy(T2).float(), torch.from_numpy(S).float()


def hybrid_fit_mono2D(signals, bvals=[0, 150, 1000, 1500], normTE=[0, 13, 93, 143]):

    eps = 1e-7;
    numcols, acquisitions = signals.shape
    D = np.zeros((numcols, 1))
    T2 = np.zeros((numcols, 1))
    for col in tqdm(range(numcols)):
        voxel = signals[col]
        X, Y = np.meshgrid(normTE, bvals)
        xdata = np.vstack((Y.ravel(), X.ravel()))
        ydata = voxel.ravel()
        try:
            fitdata_, _  = curve_fit(one_compartment_fit2D, 
                                       xdata,
                                       ydata,
                                       p0 = [1, 3],
                                       check_finite=True,
                                       bounds=([0.0, 0.0],
                                               [5, 10]),
                                      method='trf',
                                      maxfev=5000)
        except RuntimeError:
            fitdata_ = [1, 3]
        coeffs = fitdata_
        D[col] = coeffs[0]
        T2[col] = coeffs[1]
    return D, T2

def one_compartment_fit2D(M, D, T2):

    b, TE = M
    S = np.exp(-b/1000*D)*np.exp(-TE/(T2*100))   
    return S

def scatterplot(D, T2, D_true, T2_true, f, ax, log=True):

    stats = {}
    for r in range(2):
        if r==0:
            x_image, y_image = D_true, D
            title = 'ADC'
            ylims = [(0.3, 3)]
        elif r==1:
            x_image, y_image = T2_true*100, T2*100
            title = 'T2'
            ylims = [(20, 750)]
        x = x_image.detach().cpu().numpy()
        if isinstance(y_image, torch.Tensor):
            y = y_image.detach().cpu().numpy()[:, 0]
        else:
            y = y_image[:, 0]
        nbins=300
        k = gaussian_kde([x,y])
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        ax[r].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="hot", shading='auto')
        err = np.mean(np.abs(x - y))
        corr = np.corrcoef(x, y)[0,1]
        stats[title] = (err, corr)
        if log:     
            f.write(f'NLLS: {title} MAE = {err:.3f}, rho = {corr:.3f}\n')
        ax[r].scatter(x,y, color='white', s=8, alpha=0.5)
        ax[r].set_title(fr'{title}, MAE = {err:.3f}, $\rho$ = {corr:.3f}')
        ax[r].set_xlabel('true')
        ax[r].set_ylabel('predicted')
    return stats, ax
