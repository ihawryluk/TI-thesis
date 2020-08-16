# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 10:34:30 2020

@author: iwona
"""


import pandas as pd
import pystan
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

MEAN = [5, 6, 6.5, 7, 8]

ITER = 2000
SEED = 1234
CHAINS = 4
OUT_PATH = ''

def get_GI(mean):
    return 1 / (2 * (np.sqrt(2/np.pi) * mean) **2)

SKdata = pd.read_csv('SouthKoreaData.csv')


stan_code = """
data {
  int<lower=1> N; // days of observed data for country m. each entry must be <= N
  int cases[N]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  real x[N]; // x
  real GI;
}
parameters {
  real<lower=0> phi;
  vector[N+1] weekly_effect;
  real<lower=0, upper=1> weekly_rho;
  real<lower=0, upper=1> weekly_rho1;
  real<lower=0> weekly_sd;
  //real<lower=0> GI;
  real<lower=0> R0;
}

transformed parameters {
  vector[N] prediction=rep_vector(1e-5,N);
  vector<lower=0>[N] Rt;
  vector[N] SI_rev; // SI in reverse order
  {
   
   vector[N] SI;
   
  //real GI = 0.01;
    SI[1] = exp(-0.5*GI*1.5^2);  // is that needed? we're overwriting in the next line
    for(i in 1:N){
      SI[i] = exp(-(i-0.5)^2*GI) - exp(-(i+0.5)^2*GI); 
    }
    for(i in 1:N){
      SI_rev[i] = SI[N-i+1];
    }
    
    Rt[1:N] = exp(weekly_effect[1:N]);
    for (i in 2:N) {
      real convolution = dot_product(prediction[1:(i-1)], tail(SI_rev, i-1));
      prediction[i] = prediction[i] + Rt[i] * convolution;
    }
    
  }
}
model {
  weekly_sd ~ normal(0,0.2);
  weekly_rho ~ normal(0.8, 0.05);
  weekly_rho1 ~ normal(0.1, 0.05);
  phi ~ normal(0,5);
  //GI ~ normal(0.01,0.001);
  R0 ~ normal(3.28, 1); // citation: https://academic.oup.com/jtm/article/27/2/taaa021/5735319
  
  weekly_effect[3:(N+1)] ~ normal( weekly_effect[2:N]* weekly_rho + weekly_effect[1:(N-1)]* weekly_rho1, 
  weekly_sd *sqrt(1-pow(weekly_rho,2)-pow(weekly_rho1,2) - 2 * pow(weekly_rho,2) * weekly_rho1/(1-weekly_rho1)));
  weekly_effect[2] ~ normal(-1,weekly_sd *sqrt(1-pow(weekly_rho,2)-pow(weekly_rho1,2) - 2 * pow(weekly_rho,2) * weekly_rho1/(1-weekly_rho1)));
  weekly_effect[1] ~ normal(-1, 0.1);
 
  cases ~ neg_binomial_2(prediction,phi);
}
"""


BH_model = pystan.StanModel(model_code=stan_code)

def get_posterior(GI):
    stan_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 
                 'x': np.arange(1,SKdata.shape[0]+1), 'GI': GI}
    fit = BH_model.sampling(data=stan_data, iter=ITER, seed=SEED, chains=CHAINS,
                            n_jobs=-1, control={'adapt_delta': 0.8})
    
    all_samples = fit.to_dataframe()

    weekly_effect = all_samples[[col for col in all_samples if col.startswith('weekly_effect')]]
    weekly_effect_means = list(weekly_effect.mean(axis=0))
    
    samples = all_samples[['phi', 'weekly_rho', 'weekly_rho1', 'weekly_sd', 'R0']]
    samples = pd.concat([samples,weekly_effect], axis=1)
    cov = np.cov(np.array(samples), rowvar = False)
    theta_mean = samples.mean(axis = 0)
    c = np.linalg.inv(np.linalg.cholesky(cov))
    covinv = np.dot(c.T,c)
    return {'theta' : theta_mean, 'cov': cov, 'covinv': covinv}



cases = SKdata.cases.values


def convert_params(mu, phi):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    r = phi
    p = mu / (mu+r)
    return r, 1 - p



def Logf(theta_mean, GI):
    
    N = len(cases)
        
    # here we're extracting the parameters estimates from the theta vector
    phi = theta_mean[0]
    weekly_rho = theta_mean[1]
    weekly_rho1 = theta_mean[2]
    weekly_sd = theta_mean[3]    
    R0 = theta_mean[4]
    weekly_effect = theta_mean[5:].values
    
    log_prior = 0
    log_likelihood = 0
    
    # log-prior
    log_prior += scipy.stats.norm(0,5).logpdf(phi)
    log_prior += scipy.stats.norm(3.28,1).logpdf(R0)
    log_prior += scipy.stats.norm(0,0.2).logpdf(weekly_sd)
    log_prior += scipy.stats.norm(0.8,0.05).logpdf(weekly_rho)
    log_prior += scipy.stats.norm(0.1,0.05).logpdf(weekly_rho1)


    log_prior += scipy.stats.norm(-1, 0.1).logpdf(weekly_effect[0])
    log_prior += scipy.stats.norm(-1,
                                  weekly_sd * np.sqrt(1-weekly_rho**2-weekly_rho1**2 - 2 * weekly_rho**2 * weekly_rho1/(1-weekly_rho1))).logpdf(weekly_effect[1])
    
    we3_mu = weekly_effect[1:N] * weekly_rho + weekly_effect[0:(N-1)]* weekly_rho1
    we3_sig = weekly_sd * np.sqrt(1-weekly_rho**2-weekly_rho1**2 - 2 * weekly_rho**2* weekly_rho1/(1-weekly_rho1))
    for i in range(2,N+1):
        log_prior += scipy.stats.norm(we3_mu[i-2], we3_sig).logpdf(weekly_effect[i])


    # SI
    SI = np.ones(N)
    for i in range(N):
        SI[i] = np.exp(-((i+1-0.5)**2)*GI) - np.exp(-((i+1+0.5)**2)*GI)
        
    SI_rev = SI[::-1] # SI in reverse order
    
    # Rt
    Rt = np.exp(weekly_effect)[:-1] # remove last element, weekly_effect has length N+1

    # prediction
    prediction = np.ones(N) * 1e-5

    for i in range(1,N):
        convolution = np.dot(prediction[:i], SI_rev[-i:])
        prediction[i] = prediction[i] +  Rt[i] * convolution
    
    # log-likelihood
    for i in range(N):
        r, p = convert_params(prediction[i], phi)
        log_likelihood += scipy.stats.nbinom(n=r, p=p).logpmf(cases[i])
    return log_likelihood + log_prior    


def LogLaplaceCovariance(theta_mean, cov, GI):
     result = 1/2 * len(theta_mean) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(cov))
     result += Logf(theta_mean, GI)
     return result


def get_all_model_fitting(m):
    print(m)
    GI = get_GI(m)
    posterior = get_posterior(GI)
    posterior.update({'GI': m})
    posterior.update({'Logf': Logf(posterior['theta'], GI)})
    posterior.update({'LogLaplaceCovariance': LogLaplaceCovariance(posterior['theta'], 0.9 * posterior['cov'], GI)})
    return posterior

all_posteriors = {}
for m in MEAN:
    all_posteriors.update({m: get_all_model_fitting(m)})

laplaces = pd.read_csv('Laplace.csv')
laplaces = laplaces.append({'Logf': Logf(theta_mean, GI), 
                            'LogLaplaceCovariance': LogLaplaceCovariance(theta_mean, 0.9*cov, GI),
                            'GI': 40 }, ignore_index=True)
laplaces.to_csv('Laplace.csv', index=False)


model_fits = pd.DataFrame(columns = ['Logf', 'LogLaplaceCovariance', 'GI'])
for k in all_posteriors.keys():
    d = {'Logf': all_posteriors[k]['Logf'],
         'LogLaplaceCovariance': all_posteriors[k]['LogLaplaceCovariance'],
         'GI': all_posteriors[k]['GI']}
    model_fits = model_fits.append(d, ignore_index=True)
    
model_fits.to_csv('fixed_GI_Laplace.csv', index = False)