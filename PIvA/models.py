import torch
from torch import nn

class PIvA_mono(nn.Module):
    '''
    Physics-Informed Variational Autoencoder for monoexponential decay
    '''
    def __init__(self, 
                b_values = [0, 150, 1000, 1500],
                TE_values = [0, 13, 93, 143],
                hidden_dims = [32, 64, 128, 256, 512],
                predictor_depth=2):
        super(PIvA_mono, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        self.number_of_signals = len(b_values) * len(TE_values)
        self.b_values = b_values
        self.TE_values = TE_values
        self.relu = nn.ReLU()

        modules = []
        # Build Encoder
        in_channels = self.number_of_signals
        for h_dim in hidden_dims:
            
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.LeakyReLU())
                
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules).to(self.device)

        # Now we will the mean and variance of the latent space for D and T2 separately

        def build_predictor(hidden_dims, predictor_depth):
            predictor = []
            for _ in range(predictor_depth):
                predictor.append(
                    nn.Sequential(
                        nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                        nn.LeakyReLU())
                )
            predictor.append(nn.Linear(hidden_dims[-1], 1))
            return nn.Sequential(*predictor).to(self.device)

        self.D_logmean = build_predictor(hidden_dims, predictor_depth)
        self.D_logvar = build_predictor(hidden_dims, predictor_depth)
        self.T2_logmean = build_predictor(hidden_dims, predictor_depth)
        self.T2_logvar = build_predictor(hidden_dims, predictor_depth)

        # The following lines initialization is key because we want the variances to be positive and start from sigma=1
        torch.nn.init.zeros_(self.D_logvar[-1].weight) 
        torch.nn.init.zeros_(self.D_logvar[-1].bias) 
        torch.nn.init.zeros_(self.T2_logvar[-1].weight)
        torch.nn.init.zeros_(self.T2_logvar[-1].bias)

    def encode(self, x):

        x = self.encoder(x)
        D_logmean = self.D_logmean(x) # We want the mean to be positive, this is why we use the exponential
        D_logvar = self.D_logvar(x)

        T2_logmean = self.T2_logmean(x)
        T2_logvar = self.T2_logvar(x)

        return D_logmean, D_logvar, T2_logmean, T2_logvar
    
    def reparameterize(self, logmean, logvar):

        std = torch.exp(0.5*logvar)
        mu = torch.exp(logmean)
        
        while True:
            eps = torch.randn_like(std)
            if torch.all(mu + eps*std > 0):
                break
            
        return mu + eps*std
    
    def decode(self, D, T2):

        signal = torch.zeros(D.shape[0], self.number_of_signals).to(self.device)
        D, T2 = D.T, T2.T
        ctr = 0
        for b in self.b_values:
            for TE in self.TE_values:
                signal[:, ctr] = torch.exp(-b/1000 * D) * torch.exp(-TE / (T2*100))
                ctr += 1

        return signal
    
    def calculate_KL_loss(self, logmean, logvar, prior_mean, prior_var):

        mean = torch.exp(logmean)
        var = torch.exp(logvar)

        kl_div = 0.5 * (torch.log(prior_var) - logvar + (var + (mean - prior_mean)**2) / prior_var - 1)

        return kl_div.sum()
    
    def total_kl_loss(self, all_logmeans, all_logvars, all_prior_means, all_prior_vars):

        kl_loss = 0
        for logmean, logvar, prior_mean, prior_var in zip(all_logmeans, all_logvars, all_prior_means, all_prior_vars):

            kl_loss += self.calculate_KL_loss(logmean, logvar, prior_mean, prior_var)

        return kl_loss/len(all_logmeans)
    
    def loss_function(self, pred_signal, true_signal, logmeans, logvars, prior_means, prior_vars, alpha):
            
            mse = nn.functional.mse_loss(pred_signal, true_signal)
            kl = self.total_kl_loss(logmeans, logvars, prior_means, prior_vars)
            
            return mse + alpha*kl, mse, kl







        








