import numpy as np 
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as tdist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE2(nn.Module):
    def __init__(self, hidden_dims=[500, 500, 2, 500, 500], data_dim=784):
        super().__init__()
        self.data_dim = data_dim
        self.device = device
        # define IO
        self.in_layer = nn.Linear(data_dim, hidden_dims[0])
        self.out_layer = nn.Linear(hidden_dims[-1], data_dim)
        # hidden layer
        self.enc_h = nn.Linear(hidden_dims[0], hidden_dims[1])
        # define hidden and latent
        self.enc_mu = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc_sigma = nn.Linear(hidden_dims[1], hidden_dims[2])
        # hidden layer decoder
        self.dec_h = nn.Linear(hidden_dims[2], hidden_dims[-2])
        self.dec_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.to(device)

    def encode(self, x: torch.Tensor):
        h1 = F.dropout(F.relu(self.in_layer(x)), p=0.1)
        h2 = F.dropout(F.relu(self.enc_h(h1)), p=0.1)
        return self.enc_mu(h2), self.enc_sigma(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.dropout(F.relu(self.dec_h(z)), p=0.1)
        h4 = F.dropout(F.relu(self.dec_layer(h3)), p=0.1)
        return torch.sigmoid(self.out_layer(h4))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE20(nn.Module):
    def __init__(self, hidden_dims=[500, 500, 20, 500, 500], data_dim=784):
        super().__init__()
        self.data_dim = data_dim
        self.device = device
        # define IO
        self.in_layer = nn.Linear(data_dim, hidden_dims[0])
        self.out_layer = nn.Linear(hidden_dims[-1], data_dim)
        # hidden layer
        self.enc_h = nn.Linear(hidden_dims[0], hidden_dims[1])
        # define hidden and latent
        self.enc_mu = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc_sigma = nn.Linear(hidden_dims[1], hidden_dims[2])
        # hidden layer decoder
        self.dec_h = nn.Linear(hidden_dims[2], hidden_dims[-2])
        self.dec_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.to(device)

    def encode(self, x: torch.Tensor):
        h1 = F.dropout(F.relu(self.in_layer(x)), p=0.1)
        h2 = F.dropout(F.relu(self.enc_h(h1)), p=0.1)
        return self.enc_mu(h2), self.enc_sigma(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.dropout(F.relu(self.dec_h(z)), p=0.1)
        h4 = F.dropout(F.relu(self.dec_layer(h3)), p=0.1)
        return torch.sigmoid(self.out_layer(h4))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CBVAE_Mean(nn.Module):
    def __init__(self, hidden_dims=[500, 500, 20, 500, 500], data_dim=784):
        super().__init__()
        self.data_dim = data_dim
        self.device = device
        # define IO
        self.in_layer = nn.Linear(data_dim, hidden_dims[0])
        self.out_layer = nn.Linear(hidden_dims[-1], data_dim)
        # hidden layer
        self.enc_h = nn.Linear(hidden_dims[0], hidden_dims[1])
        # define hidden and latent
        self.enc_mu = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc_sigma = nn.Linear(hidden_dims[1], hidden_dims[2])
        # hidden layer decoder
        self.dec_h = nn.Linear(hidden_dims[2], hidden_dims[-2])
        self.dec_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.to(device)

    def encode(self, x: torch.Tensor):
        h1 = F.dropout(F.relu(self.in_layer(x)), p=0.1)
        h2 = F.dropout(F.relu(self.enc_h(h1)), p=0.1)
        return self.enc_mu(h2), self.enc_sigma(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.dropout(F.relu(self.dec_h(z)), p=0.1)
        h4 = F.dropout(F.relu(self.dec_layer(h3)), p=0.1)
        temp = torch.sigmoid(self.out_layer(h4))
        temp = tdist.ContinuousBernoulli(probs=temp) 
        return temp.mean

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
class CBVAE_Lambda(nn.Module):
    def __init__(self, hidden_dims=[500, 500, 20, 500, 500], data_dim=784):
        super().__init__()
        self.data_dim = data_dim
        self.device = device
        # define IO
        self.in_layer = nn.Linear(data_dim, hidden_dims[0])
        self.out_layer = nn.Linear(hidden_dims[-1], data_dim)
        # hidden layer
        self.enc_h = nn.Linear(hidden_dims[0], hidden_dims[1])
        # define hidden and latent
        self.enc_mu = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc_sigma = nn.Linear(hidden_dims[1], hidden_dims[2])
        # hidden layer decoder
        self.dec_h = nn.Linear(hidden_dims[2], hidden_dims[-2])
        self.dec_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.to(device)

    def encode(self, x: torch.Tensor):
        h1 = F.dropout(F.relu(self.in_layer(x)), p=0.1)
        h2 = F.dropout(F.relu(self.enc_h(h1)), p=0.1)
        return self.enc_mu(h2), self.enc_sigma(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.dropout(F.relu(self.dec_h(z)), p=0.1)
        h4 = F.dropout(F.relu(self.dec_layer(h3)), p=0.1)
        return torch.sigmoid(self.out_layer(h4))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

class BetaVAE(nn.Module):
    def __init__(self, hidden_dims=[500, 500, 20, 500, 500], data_dim=784):
        super().__init__()
        self.data_dim = data_dim
        self.device = device
        
        # self.beta_reg = nn.Parameter(torch.ones(1))
        # define IO
        self.in_layer = nn.Linear(data_dim, hidden_dims[0])
        self.out_layer = nn.Linear(hidden_dims[-1], 2*data_dim)
        # self.out_layer_alpha = nn.Linear(hidden_dims[-1], data_dim)
        # self.out_layer_beta = nn.Linear(hidden_dims[-1], data_dim)
        # hidden layer
        self.enc_h = nn.Linear(hidden_dims[0], hidden_dims[1])
        # define hidden and latent
        self.enc_mu = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc_sigma = nn.Linear(hidden_dims[1], hidden_dims[2])
        # hidden layer decoder
        self.dec_h = nn.Linear(hidden_dims[2], hidden_dims[-2])
        self.dec_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.to(device)
        
    def encode(self, x: torch.Tensor):
        h1 = F.dropout(F.relu(self.in_layer(x)), p=0.1)
        h2 = F.dropout(F.relu(self.enc_h(h1)), p=0.1)
        return self.enc_mu(h2), self.enc_sigma(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.dropout(F.relu(self.dec_h(z)), p=0.1)
        h4 = F.dropout(F.relu(self.dec_layer(h3)), p=0.1)
        beta_params = self.out_layer(h4)
        alphas = 1e-6 + F.softmax(beta_params[:, :self.data_dim])
        betas = 1e-6 + F.softmax(beta_params[:, self.data_dim:])
        # alphas = 1e-6 + F.relu(beta_params[:, :self.data_dim])
        # betas = 1e-6 + F.relu(beta_params[:, self.data_dim:])
        return alphas, betas

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        alphas, betas = self.decode(z)
        return alphas, betas, mu, logvar

