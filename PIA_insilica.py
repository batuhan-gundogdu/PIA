'''
PIA Non-commercial License
© 2023-2025 The University of Chicago.
Author: Batuhan Gundogdu


Redistribution and use for noncommercial purposes in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. The software is used solely for noncommercial purposes. It may not be used indirectly for commercial use, such as on a website that accepts advertising money for content. Noncommercial use does include use by a for-profit company in its research. For commercial use rights, contact The University of Chicago, Polsky Center for Entrepreneurship, and Innovation, at polskylicensing@uchicago.edu or call 773-702-1692 and inquire about Tech ID XX-X-XXX project.
2. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
3. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
4. Neither the name of The University of Chicago nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF CHICAGO AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF CHICAGO OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import argparse
import os
from utils import set_seed, get_batch, get_batch_2compartment, hybrid_fit, get_scores, get_scores2, plot_results, calculate_stats
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIA import PIA
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser(description='Monte Carlo Experiments of PIA for diffusion-relaxation mode')
parser.add_argument('--sample_size', type=int, default=2500, help='Sample size for in-silica experiments')
parser.add_argument('--noise_std', type=float, default=0.02, help='Depth of the predictor network')


def main(args):

    set_seed(16)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    print('Starting the In-silica Experiments for Physics-Informed Autoencoder (PIA) for diffusion-relaxation model...')
    test_tensor, D_test, T2_test, v_test, clean = get_batch(args.sample_size, noise_sdt=args.noise_std)
    test = test_tensor.detach().cpu().numpy()
    start = time.time()
    print('Calculating the NLLS fit - this may take a while...(but rejoice, we are solving it with PIA!)')
    D_NLLS, T2_NLLS, v_NLLS = hybrid_fit(test)
    end = time.time()
    print(f'NLLS solution takes {end - start} seconds for {args.sample_size} samples')
    get_scores((v_test, D_test, T2_test), (v_NLLS, D_NLLS, T2_NLLS))
    print('Plotting the results...')
    plot_results((v_test, D_test, T2_test), (v_NLLS, D_NLLS, T2_NLLS), method='NLLS')
    print('Now Testing with the PIA model...')
    PATH = 'pia_model.pt'
    model = torch.load(PATH)
    test_tensor = test_tensor.cuda()
    start = time.time()
    D, T2, v = model.encode(test_tensor)
    end = time.time()
    print(f'PIA takes {end - start} seconds for {args.sample_size} samples')
    print('Plotting the results for PIA...')
    get_scores((v_test, D_test, T2_test), (v, D, T2))
    plot_results((v_test, D_test, T2_test), (v, D, T2), method='PIA')
    print('Comparing the results of PIA and NLLS (p-values)...')
    calculate_stats((v_test, D_test, T2_test), (v, D, T2), (v_NLLS, D_NLLS, T2_NLLS))
    print('Conducting Speed Tests')
    test_tensor, D_test2, T2_test2, v_test, clean = get_batch(20000, noise_sdt=args.noise_std)
    test = test_tensor.detach().cpu().numpy()
    start = time.time()
    D, T2, v = hybrid_fit(test)
    end = time.time()
    print(f'NLLS solution took {end - start} seconds for 20,000 samples')
    test_tensor = test_tensor.cuda()
    start = time.time()
    D, T2, v = model.encode(test_tensor)
    end = time.time()
    print(f'PIA took {end - start} seconds for 20,000 samples')
    print('Now conducting Robustness Tests of PIA...')
    print('Stage 1: Testing with different levels of noise')
    
    noise_levels = [2*0.0001, 5*0.0001, 7*0.0001, 0.001, 2*0.001, 5*0.001, 7*0.001, 0.01, 0.02, 0.05, 0.07, 0.1]
    v_ep, v_st, v_lu, D_ep, D_st, D_lu, T2_ep, T2_st, T2_lu = [], [], [], [], [], [], [], [], []
    results_dict = {
        'v': [v_ep, v_st, v_lu], 
        'D': [D_ep, D_st, D_lu], 
        'T2': [T2_ep, T2_st, T2_lu]
    }

    B = 100 # number of samples used in stress testing
    for N in tqdm(noise_levels):  
        _, D_test2, T2_test2, v_test, clean = get_batch(B, noise_sdt=0)
        clean = clean.detach().cpu().numpy()/1000
        noise_im = np.random.normal(0, N, (B, 16))
        noise_re = np.random.normal(0, N, (B, 16))
        noisy = np.sqrt((clean + noise_im)**2 + (clean + noise_re)**2) / np.sqrt(2)

        test_tensor = 1000 * torch.from_numpy(noisy).float()
        test_tensor = test_tensor.cuda()
        D_pia, T2_pia, v_pia = model.encode(test_tensor)
        test = test_tensor.detach().cpu().numpy()
        D_hybrid, T2_hybrid, v_hybrid = hybrid_fit(test)
        
        test_cases = [
            (v_test, v_pia, v_hybrid, 'v'),
            (D_test2, D_pia, D_hybrid, 'D'),
            (T2_test2, T2_pia, T2_hybrid, 'T2')
        ]
        
        for test_data, pia, hybrid, key in test_cases:
            for i, result_list in enumerate(results_dict[key]):
                corr, mae, bias, std = get_scores2(test_data, pia, hybrid, i)
                result_list.append({'corr': corr, 'mae': mae, 'bias': bias, 'std': std})

    sequence_names = ['Ep. Vol.', 'St. Vol.', 'Lu. Vol' , 'Ep. D.', 'St. D.', 'Lu. D', 'Ep. T2.', 'St. T2.', 'Lu. T2']
    measure_names = ['Spearman R', 'MAE', 'Bias', 'Std.Dev']
    for name, sq in enumerate([v_ep, v_st, v_lu, D_ep, D_st, D_lu, T2_ep, T2_st, T2_lu]):
        fig, ax = plt.subplots(1,4, figsize=(20,5))
        for m, kw in enumerate(['corr', 'mae', 'bias', 'std']):       
            P = [x[kw][0] for x in sq]
            H = [x[kw][1] for x in sq]
            line1 = ax[m].plot(noise_levels, P, color='skyblue', lw=2, marker='o', label='PIA')
            line2 = ax[m].plot(noise_levels, H, color='salmon', lw=2, marker='s', label ='NLLS')
            ax[m].set_xscale('log')
            ax[m].set_xlabel('Noise Std. Dev.')
            ax[m].set_title(f'{sequence_names[name]} {measure_names[m]}')
            ax[m].legend(['PIA', 'NLLS'])
            ax[m].grid(True)
        plt.savefig(f'plots/noise_response_{sequence_names[name]}.png')
        plt.close(fig)
    
    print('Plots saved in plots/ directory.')
    print('Stage 2: Testing with a different protocol')
    new_protocol_b_values = [0, 300, 600, 900]
    new_protocol_TE_values = [0, 5, 100, 150]
    new_protocol_PIA = PIA(b_values = new_protocol_b_values, TE_values = new_protocol_TE_values, predictor_depth=2)
    new_protocol_PIA.load_state_dict(model.state_dict())
    test_tensor, D_test, T2_test, v_test, _ = get_batch(50)
    test_tensor = test_tensor.cuda()
    D_pia, T2_pia, v_pia = new_protocol_PIA.encode(test_tensor)

    # Now, get a batch from the Out-of-Domain distribution (new protocol)
    test_tensor, D_test2, T2_test2, v_test2, _ = get_batch(50, b_values=new_protocol_b_values, TE_values=new_protocol_TE_values)
    test_tensor = test_tensor.cuda()

    params = new_protocol_PIA.parameters()
    lr = 3e-4
    optimizer = optim.Adam(params, lr=lr)
    for ep in range(1): # fine tuning for 1 epoch
        for i in range(test_tensor.shape[0]):
            x , y = test_tensor[i].unsqueeze(dim=0).cuda(), test_tensor[i].unsqueeze(dim=0).cuda()
            optimizer.zero_grad()
            D, T2, v = new_protocol_PIA.encode(x)  
            recon = new_protocol_PIA.decode(D, T2, v)
            loss = new_protocol_PIA.loss_function(recon, y)
            loss.backward()
            optimizer.step()
    tissues = ['Epithelium', 'Stroma', 'Lumen']
    D_pia3, T2_pia3, v_pia3 = new_protocol_PIA.encode(test_tensor)
    for t in range(3):
        error_D_id = np.abs(D_pia[:, t].detach().cpu().numpy() - D_test[:, t].detach().cpu().numpy())
        error_T2_id = np.abs(T2_pia[:, t].detach().cpu().numpy() - T2_test[:, t].detach().cpu().numpy())
        error_v_id = np.abs(v_pia[:, t].detach().cpu().numpy() - v_test[:, t].detach().cpu().numpy())
        error_D_ood = np.abs(D_pia3[:, t].detach().cpu().numpy() - D_test2[:, t].detach().cpu().numpy())
        error_T2_ood = np.abs(T2_pia3[:, t].detach().cpu().numpy() - T2_test2[:, t].detach().cpu().numpy())
        error_v_ood = np.abs(v_pia3[:, t].detach().cpu().numpy() - v_test2[:, t].detach().cpu().numpy())
        print(f'{tissues[t]} vol: {ttest_ind(error_v_id, error_v_ood)}')
        print(f'{tissues[t]} D: {ttest_ind(error_D_id, error_D_ood)}')
        print(f'{tissues[t]} T2: {ttest_ind(error_T2_id, error_T2_ood)}')

    print('Stage 3: Testing with a different tissue model')

    model2 = PIA(predictor_depth=2)
    model2.load_state_dict(model.state_dict())
    test_tensor, D_test, T2_test, v_test, clean = get_batch(50)
    test_tensor = test_tensor.cuda()
    D_pia, T2_pia, v_pia = model.encode(test_tensor)

    test_tensor, D_test2, T2_test2, v_test2, clean = get_batch_2compartment(50)
    test_tensor = test_tensor.cuda()

    params = model.parameters()
    lr = 3e-4
    optimizer = optim.Adam(params, lr=lr)
    for ep in range(1): # fine tuning for 1 epoch
        for i in range(test_tensor.shape[0]):
            x , y = test_tensor[i].unsqueeze(dim=0).cuda(), test_tensor[i].unsqueeze(dim=0).cuda()
            optimizer.zero_grad()
            D, T2, v = model2.encode(x)  
            recon = model2.decode(D, T2, v)
            loss = model2.loss_function(recon, y)
            loss.backward()
            optimizer.step()

    D_pia3, T2_pia3, v_pia3 = model2.encode(test_tensor)

    error_D_id_low = np.abs(D_pia[:, 0].detach().cpu().numpy() - D_test[:, 0].detach().cpu().numpy())
    error_T2_id_low = np.abs(T2_pia[:, 0].detach().cpu().numpy() - T2_test[:, 0].detach().cpu().numpy())
    error_v_id_low = np.abs(v_pia[:, 0].detach().cpu().numpy() - v_test[:, 0].detach().cpu().numpy())

    error_D_ood_low = np.abs(D_pia3[:, 0].detach().cpu().numpy() - D_test2[:, 0].detach().cpu().numpy())
    error_T2_ood_low = np.abs(T2_pia3[:, 0].detach().cpu().numpy() - T2_test2[:, 0].detach().cpu().numpy())
    error_v_ood_low = np.abs(v_pia3[:, 0].detach().cpu().numpy() - v_test2[:, 0].detach().cpu().numpy())

    error_D_id_high = np.abs(D_pia[:, 2].detach().cpu().numpy() - D_test[:, 2].detach().cpu().numpy())
    error_T2_id_high = np.abs(T2_pia[:, 2].detach().cpu().numpy() - T2_test[:, 2].detach().cpu().numpy())
    error_v_id_high = np.abs(v_pia[:, 2].detach().cpu().numpy() - v_test[:, 2].detach().cpu().numpy())

    error_D_ood_high = np.abs(D_pia3[:, 2].detach().cpu().numpy() - D_test2[:, 1].detach().cpu().numpy())
    error_T2_ood_high = np.abs(T2_pia3[:, 2].detach().cpu().numpy() - T2_test2[:, 1].detach().cpu().numpy())
    error_v_ood_high = np.abs(v_pia3[:, 2].detach().cpu().numpy() - v_test2[:, 1].detach().cpu().numpy())

    print(f'Low D: {ttest_ind(error_D_id_low, error_D_ood_low)}')
    print(f'Low T2: {ttest_ind(error_T2_id_low, error_T2_ood_low)}')
    print(f'Low v: {ttest_ind(error_v_id_low, error_v_ood_low)}')
    print(f'High D: {ttest_ind(error_D_id_high, error_D_ood_high)}')
    print(f'High T2: {ttest_ind(error_T2_id_high, error_T2_ood_high)}')
    print(f'High v: {ttest_ind(error_v_id_high, error_v_ood_high)}')


    print('Done')

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
