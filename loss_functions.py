import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as tdist

# https://github.com/Robert-Aduviri/Continuous-Bernoulli-VAE
def sumlogC(x , eps = 1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation
        
    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    x = torch.clamp(x, eps, 1.-eps) 
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    return far_values.sum() + close_values.sum()


# this can be used for VAE2, VAE20, CBVAE_Mean
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    logC = sumlogC(recon_x)
    return BCE, KLD, logC


def cb_lambda_loss(recon_x, x, mu, logvar):
    tmp = tdist.ContinuousBernoulli(probs=recon_x)
    recon_x = tmp.mean
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    logC = sumlogC(recon_x)
    return BCE, KLD, logC


def beta_loss(alphas, betas, x, mu, logvar, beta_reg):
    x = x.view(-1, 784) 
    recon_dist = tdist.Beta(alphas, betas)
    recon_x = recon_dist.mean
    recon_x = recon_x.view(-1, 784)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD #+ logC