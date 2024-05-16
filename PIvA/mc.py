'''
Mono-exponential Physics-Informed Variational Autoencoder (PI-VAE) for parameter estimation MRI data
Author: Batuhan Gundogdu (2024)
University of Chicago

'''
import argparse
import torch
from models import PIvA_mono
from torch.optim import Adam
from utils import get_batch_mono2D, generate, hybrid_fit_mono2D, scatterplot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from colorama import Fore

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prior_D_mean = torch.tensor(1.6) # prior for diffusion coefficient
    prior_D_var = torch.tensor(((1.4)**2)) # TODO: priors will be updated into posteriors after in-vivo experiments

    prior_T2_mean = torch.tensor(3.65) # prior for T2 relaxation time (normalized by a factor of 100 for computational stability)
    prior_T2_var = torch.tensor(((3.35)**2))

    prior_mean = [prior_D_mean, prior_T2_mean]
    prior_var = [prior_D_var, prior_T2_var]

    # define model
    model = PIvA_mono().to(device)
    params = model.parameters()
    optimizer = Adam(params, lr=args.lr)

    # First test set is the randomly selected 2500 datapoints
    test_tensor, D_true, T2_true, clean = get_batch_mono2D(2500, noise_std=0.05)
    test = test_tensor.detach().cpu().numpy()
    D, T2 = hybrid_fit_mono2D(test)

    fig, ax = plt.subplots(2,1, figsize=(10,20))
    fig.suptitle('NLLS fit')
    f = open(f'results/{args.experiment}.txt', 'w')
    NLLS, ax = scatterplot(D, T2, D_true, T2_true, f, ax)
    plt.savefig('plots/NLLS_MC.png')
    plt.close()
    f.write('\n')

    # Now PIvA
    test_tensor = test_tensor.to(device)

    # generate systematic test set
    granularity = 500
    D_all = np.linspace(0.3, 3, granularity)
    T2_all = np.linspace(0.2, 7.5, granularity)
    signal = torch.zeros(D_all.shape[0], T2_all.shape[0], 16).cuda()

    for i, d in enumerate(D_all):
        for j, t2 in enumerate(T2_all):
            signal[i, j], _ = generate(d, t2, noise_std=args.noise)

    # TODO: different noise levels will be investigated in the future

    D_confidence = np.zeros((granularity,granularity))
    T2_confidence = np.zeros((granularity, granularity))
    D_error = np.zeros((granularity, granularity))
    T2_error = np.zeros((granularity, granularity))


    with open(f'results/{args.experiment}_epochs.txt', 'a') as ff:
        ff.write(f"Epoch, D_MAE, D_corr, T2_MAE, T2_corr, D_confidence, T2_confidence\n")

    for epoch in range(args.epochs):
        x, _, _, y = get_batch_mono2D(1, noise_std=args.noise)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        D_logmean, D_logvar, T2_logmean, T2_logvar = model.encode(x)

        # In training, we just sample from the distribution (instead of using the direct values)
        D = model.reparameterize(D_logmean, D_logvar)
        T2 = model.reparameterize(T2_logmean, T2_logvar)

        recon = model.decode(D, T2).to(device)

        loss, mse, kl = model.loss_function(recon, y, [D_logmean, T2_logmean], [D_logvar, T2_logvar], prior_mean, prior_var, args.alpha)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:

            # TODO: also add the systematic evaluation
            D_logmean, D_logvar, T2_logmean, T2_logvar = model.encode(test_tensor)
            D = torch.exp(D_logmean)
            T2 = torch.exp(T2_logmean)

            mae_D = np.mean(np.abs(D_true.detach().cpu().numpy() - D.detach().cpu().numpy()[:, 0]))
            corr_D = np.corrcoef(D_true.detach().cpu().numpy(), D.detach().cpu().numpy()[:, 0])[0,1]
            mae_T2 = np.mean(np.abs(100*T2_true.detach().cpu().numpy() - 100*T2.detach().cpu().numpy()[:, 0]))
            corr_T2 = np.corrcoef(100*T2_true.detach().cpu().numpy(), 100*T2.detach().cpu().numpy()[:, 0])[0,1]

            metrics = [mae_D, mae_T2, corr_D, corr_T2]
            baselines = [NLLS['ADC'][0], NLLS['T2'][0], NLLS['ADC'][1], NLLS['T2'][1]]
            colors1 = [Fore.RED if m > b else Fore.GREEN for m, b in zip(metrics[:2], baselines[:2])]
            colors2 = [Fore.RED if m < b else Fore.GREEN for m, b in zip(metrics[2:], baselines[2:])]
            colors = colors1 + colors2
            colored_metrics = [f'{color}{metric:.3f}{Fore.WHITE}' for metric, color in zip(metrics, colors)]
            colored_metrics += [f'{np.mean(np.exp(D_logvar.cpu().detach().numpy())):.3f}, {np.mean(np.exp(T2_logvar.cpu().detach().numpy())):.3f}']
            with open(f'results/{args.experiment}_epochs.txt', 'a') as ff:
                ff.write(f"{epoch}, {', '.join(colored_metrics)}\n")
            if epoch % 1000 == 0:
                fig, ax = plt.subplots(2,1, figsize=(10,20))
                fig.suptitle('PIvA fit')
                _ , ax = scatterplot(D, T2, D_true, T2_true, f, ax, log=False)


                plt.savefig(f'plots/{args.experiment}_scatter_{epoch:05}.png')
                plt.close()

                for i, d in enumerate(D_all):
                    for j, t2 in enumerate(T2_all):
                        x = signal[i, j]
                        logmean_of_D, logvar_of_D, logmean_of_T2, logvar_of_T2 = model.encode(x)
                        D_confidence[i, j] = np.exp(logvar_of_D.item())
                        T2_confidence[i, j] = np.exp(logvar_of_T2.item())
                        D_error[i, j] = np.abs(torch.exp(logmean_of_D).detach().cpu().numpy() - d)
                        T2_error[i, j] = np.abs(torch.exp(logmean_of_T2).detach().cpu().numpy() - t2)

                fig, ax = plt.subplots(2,2, figsize=(20,20))
                fig.suptitle('PIvA Confidence and Error')
                images = [[D_confidence, T2_confidence], [D_error, T2_error]]
                titles = [['ADC Confidence of PIvA', 'T2 Confidence of PIvA'], ['ADC Error', 'T2 Error']]
                ticks_loc = np.linspace(0, granularity, 5)
                yticks_labels = np.linspace(0.3, 3.0, 5)
                xticks_labels = np.linspace(20, 750, 5)
                for i in range(2):
                    for j in range(2):
                        im = ax[i, j].imshow(images[i][j], cmap='inferno')
                        ax[i][j].set_title(titles[i][j])
                        ax[i][j].set_xlabel('T2')
                        ax[i][j].set_ylabel('ADC')                                          
                        ax[i][j].set_yticks(ticks_loc)
                        ax[i][j].set_yticklabels([f"{label:.1f}" for label in yticks_labels])
                        ax[i][j].set_xticks(ticks_loc)
                        ax[i][j].set_xticklabels([f"{label:.1f}" for label in xticks_labels])
                        fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)

                plt.savefig(f'plots/{args.experiment}_confidence_error_{epoch:05}.png')
                plt.close()
                # save model
                torch.save(model.state_dict(), f'checkpoints/{args.experiment}_{epoch:05}.pt')
    f.close()








if __name__ == "__main__":
    parser = argparse.ArgumentParser('Mono-exponential Physics-Informed Variational Autoencoder (PI-VAE) for parameter estimation MRI data')
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200000, help="number of epochs")
    parser.add_argument("--alpha", type=float, default=1e-5, help="alpha to mix the KL divergence with the MSE loss")
    parser.add_argument("--noise", type=float, default=0.05, help="noise std used in training and MC simulations")
    parser.add_argument("--experiment", type=str, default='exp1', help="experiment name")
    args = parser.parse_args()
    main(args)