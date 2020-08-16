# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:24:46 2020

@author: iwona
"""


import pandas as pd
import pystan
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

ITER = 2000
SEED = 1234
CHAINS = 4

SKdata = pd.read_csv('C:/Users/iwona/Desktop/TI/bellman_harris/SouthKoreaData.csv')
stan_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1)}

stan_code = """
data {
  int<lower=1> N; // days of observed data for country m. each entry must be <= N
  int cases[N]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  real x[N]; // x
}
parameters {
  real<lower=0> phi;
  vector[N+1] weekly_effect;
  real<lower=0, upper=1> weekly_rho;
  real<lower=0, upper=1> weekly_rho1;
  real<lower=0, upper=1> weekly_rho2;
  real<lower=0, upper=1> weekly_rho3;
  real<lower=0> weekly_sd;
  real<lower=0> GI;
  real<lower=0> R0;
}

transformed parameters {
  vector[N] prediction=rep_vector(1e-5,N);
  vector[N] SI_rev; // SI in reverse order
  vector[N] SI; // SI in reverse order
  vector<lower=0>[N] Rt;
  
  {
    SI[1] = exp(-0.5*GI*1.5^2);
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
  weekly_rho1 ~ normal(0.05, 0.05);
  weekly_rho2 ~ normal(0.05, 0.05);
  weekly_rho3 ~ normal(0.05, 0.05);
  phi ~ normal(0,5);
  GI ~ normal(0.01,0.001);
  R0 ~ normal(3.28, 1); // citation: https://academic.oup.com/jtm/article/27/2/taaa021/5735319
  
  weekly_effect[5:(N+1)] ~ normal(weekly_effect[4:N]* weekly_rho3 + weekly_effect[3:N-1]* weekly_rho + weekly_effect[2:(N-2)]* weekly_rho1 + weekly_effect[1:(N-3)]* weekly_rho2, weekly_sd);
  
  weekly_effect[2:4] ~ normal(-1,weekly_sd);
  weekly_effect[1] ~ normal(-1, 0.1);
 
  cases ~ neg_binomial_2(prediction,phi);
}
"""


BH_model = pystan.StanModel(model_code=stan_code)

fit = BH_model.sampling(data=stan_data, iter=ITER, seed=SEED, chains=CHAINS,
                     n_jobs=-1, control={'adapt_delta': 0.8})

all_samples = fit.to_dataframe()
all_samples.to_csv('C:/Users/iwona/Desktop/TI/bellman_harris//models_fits/' + 'AR4' + '.csv', index=False)

# prediction = all_samples[[col for col in all_samples if col.startswith('prediction')]]
# prediction_means = list(prediction.mean(axis=0))
# predictions_df = pd.read_csv('predictions_cases.csv')
# predictions_df['AR4'] = prediction_means
# predictions_df.to_csv('predictions_cases.csv', index = False)

# plt.plot(stan_data['x'], stan_data['cases'], label = 'data')
# plt.plot(stan_data['x'], prediction_means, label = 'model', ls = '-')
# plt.legend()
# plt.xlabel('Days')
# plt.ylabel('Infections')
# plt.show()

rt = all_samples[[col for col in all_samples if col.startswith('Rt')]]
rt_means = list(rt.mean(axis=0))
rt_df = pd.read_csv('Rt_estimated.csv')
rt_df['AR4'] = rt_means
rt_df.to_csv('Rt_estimated.csv', index = False)


weekly_effect = all_samples[[col for col in all_samples if col.startswith('weekly_effect')]]
weekly_effect_means = list(weekly_effect.mean(axis=0))

samples = all_samples[['phi', 'weekly_rho', 'weekly_rho1', 'weekly_rho2', 'weekly_rho3', 'weekly_sd', 'GI', 'R0']]
samples = pd.concat([samples,weekly_effect], axis=1)
cov = np.cov(np.array(samples), rowvar = False)
theta_mean = samples.mean(axis = 0)
c = np.linalg.inv(np.linalg.cholesky(cov))
covinv = np.dot(c.T,c)

cases = SKdata.cases.values

sns.pairplot(all_samples[['Rt[1]','Rt[50]','Rt[100]','Rt[150]']])
plt.title('pairplot for the model fit')
plt.show()


def convert_params(mu, phi):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    r = phi
    p = mu / (mu+r)
    return r, 1 - p



def Logf(theta_mean):
    
    N = len(cases)
        
    # here we're extracting the parameters estimates from the theta vector
    phi = theta_mean[0]
    weekly_rho = theta_mean[1]
    weekly_rho1 = theta_mean[2]
    weekly_rho2 = theta_mean[3]
    weekly_rho3 = theta_mean[4]
    weekly_sd = theta_mean[5]    
    GI = theta_mean[6]
    R0 = theta_mean[7]
    weekly_effect = theta_mean[8:].values
    
    log_prior = 0
    log_likelihood = 0
    
    # log-prior
    log_prior += scipy.stats.norm(0,5).logpdf(phi)
    log_prior += scipy.stats.norm(0.01,0.001).logpdf(GI)
    log_prior += scipy.stats.norm(3.28,1).logpdf(R0)
    log_prior += scipy.stats.norm(0,0.2).logpdf(weekly_sd)
    log_prior += scipy.stats.norm(0.8,0.05).logpdf(weekly_rho)
    log_prior += scipy.stats.norm(0.05,0.05).logpdf(weekly_rho1)
    log_prior += scipy.stats.norm(0.05,0.05).logpdf(weekly_rho2)
    log_prior += scipy.stats.norm(0.05,0.05).logpdf(weekly_rho3)


    log_prior += scipy.stats.norm(-1, 0.1).logpdf(weekly_effect[0])
    log_prior += scipy.stats.norm(-1, weekly_sd).logpdf(weekly_effect[1])
    log_prior += scipy.stats.norm(-1, weekly_sd).logpdf(weekly_effect[2])
    log_prior += scipy.stats.norm(-1, weekly_sd).logpdf(weekly_effect[3])

    we4_mu = weekly_effect[3:N] * weekly_rho3 + weekly_effect[2:N-1] * weekly_rho + weekly_effect[1:(N-2)]* weekly_rho1 + weekly_effect[0:(N-3)]* weekly_rho2
    we4_sig = weekly_sd
    for i in range(4,N+1):
        log_prior += scipy.stats.norm(we4_mu[i-4], we4_sig).logpdf(weekly_effect[i])


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

def LogLaplaceCovariance(theta_mean, cov):
     result = 1/2 * len(theta_mean) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(cov))
     result += Logf(theta_mean)
     return result


laplaces = pd.read_csv('Laplace.csv')
laplaces = laplaces.append({'Logf': Logf(theta_mean), 
                            'LogLaplaceCovariance': LogLaplaceCovariance(theta_mean, 0.9*cov),
                            'GI': 'AR4'}, ignore_index=True)
laplaces.to_csv('Laplace.csv', index=False)

##############################################################################
# TI part

TI_code = """
functions {
    
    real lf(vector theta, int N, int[] cases)
    {
     real Lprior = 0;
     real log_lik = 0;
     
     real phi = theta[1];
     real weekly_rho = theta[2];
     real weekly_rho1 = theta[3];
     real weekly_rho2 = theta[4];
     real weekly_rho3 = theta[5];
     real weekly_sd = theta[6];
     real GI = theta[7];
     real R0 = theta[8];
     vector[N+1] weekly_effect = theta[9:(8+N+1)];
     
    // prepare the remaining parameters
    vector[N-3] we4_mu = weekly_effect[4:N]* weekly_rho3 + weekly_effect[3:N-1]* weekly_rho + weekly_effect[2:(N-2)]* weekly_rho1 + weekly_effect[1:(N-3)]* weekly_rho2;
    real we4_sigma = weekly_sd;

    vector[N] prediction=rep_vector(1e-5,N);
    vector[N] Rt;
    vector[N] SI_rev; // SI in reverse order
    vector[N] SI;
       
       
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
        
    
     // priors
    for (i in 5:(N+1)){
     Lprior += normal_lpdf(weekly_effect[i] | we4_mu[i-4], we4_sigma);
    }
    Lprior += normal_lpdf(weekly_effect[4] | -1, weekly_sd);
    Lprior += normal_lpdf(weekly_effect[3] | -1, weekly_sd);
    Lprior += normal_lpdf(weekly_effect[2] | -1, weekly_sd);
    Lprior += normal_lpdf(weekly_effect[1] | -1, 0.1);
 
    Lprior += normal_lpdf(weekly_sd|0, 0.2);
    Lprior += normal_lpdf(weekly_rho|0.8, 0.05);
    Lprior += normal_lpdf(weekly_rho1|0.05, 0.05);
    Lprior += normal_lpdf(weekly_rho2|0.05, 0.05);
    Lprior += normal_lpdf(weekly_rho3|0.05, 0.05);
    Lprior += normal_lpdf(phi|0,5);
    Lprior += normal_lpdf(GI|0.01,0.001);
    Lprior += normal_lpdf(R0|3.28, 1);
 
    // likelihood
    for (i in 1:N) {        
     log_lik += neg_binomial_2_lpmf(cases[i] | prediction[i],phi);
     
      }
    
    return log_lik + Lprior;   
    }            
        
    real lfref(vector theta, real f0, vector peak, matrix C)
    {
     return f0 - 0.5 * (theta-peak)' * C * (theta-peak);
     }
    
    real path_lpdf(vector theta, real f0, vector peak, matrix C, int N, int[] cases, real lambda)
    {
    return lambda * lf(theta, N, cases) + (1-lambda) * lfref(theta, f0, peak, C);
    } 
    }
                                       
data {  
      int<lower=1> N; // days of observed data for country m. each entry must be <= N
      int cases[N]; // reported deaths -- the rows with i > N contain -1 and should be ignored
      real x[N]; // x
      matrix[8 + N + 1,8 + N + 1] C;
      vector[8 + N + 1] peak; // mean parameters estimate
      real f0; // log(f(peak))
      real<lower=0,upper=1> lambda; // used for creating a path between f and f_ref
      }

transformed data {
   }

parameters {
  real<lower=0> phi;
  vector[N+1] weekly_effect;
  real<lower=0, upper=1> weekly_rho;
  real<lower=0, upper=1> weekly_rho1;
  real<lower=0, upper=1> weekly_rho2;
  real<lower=0, upper=1> weekly_rho3;
  real<lower=0> weekly_sd;
  real<lower=0> GI;
  real<lower=0> R0;
    
}

transformed parameters {
    vector[8 + N + 1] theta;
    theta[1] = phi;
    theta[2] = weekly_rho;
    theta[3] = weekly_rho1;
    theta[4] = weekly_rho2;
    theta[5] = weekly_rho3;
    theta[6] = weekly_sd;
    theta[7] = GI;
    theta[8] = R0;
    theta[9:(8+N+1)] = weekly_effect;

    }
model {
    theta ~ path(f0, peak, C, N, cases, lambda);
}

generated quantities {
    real logf = lf(theta, N, cases);
    real logfref = lfref(theta, f0, peak, C);
    real diff = logf - logfref;
    }
"""

TI_BH_model = pystan.StanModel(model_code=TI_code)

def get_expect_for_lambda(lam, data, n_iter=ITER):
    fit = TI_BH_model.sampling(data=data, iter=n_iter, n_jobs=-1, control = {'adapt_delta': 0.8}, seed=SEED)
    vals = fit.extract()['diff']
    expects = vals.mean()
    df = pd.DataFrame(columns = ['diff', 'mean'])
    df['diff'] = fit.extract('diff')['diff']
    df['mean'] = df['diff'].cumsum() / (df.index + 1)
    df.to_csv('' + 'GI_est_AR4' + '_' + str(lam) + '.csv', index=False)
    print(lam, ', expectation = ', expects)
    return {'expects': expects, 'vals': vals}

def MCMC_for_all_lambdas(lambdaVals, data):
    """Execute TI MCMC for multiple lambdas.
    lambdaVals: list of values of lambdas"""
    lambdaOutput = {}
    for l in lambdaVals:
        lam = round(l,1)
        data.update({'lambda': lam})
        lambdaOutput.update({lam: get_expect_for_lambda(lam, data)})
    return lambdaOutput

def get_TI_dict(cov_frac = 1.0):
    TI_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1),
           'C': (1/cov_frac) * covinv, 'f0': Logf(theta_mean), 'peak': theta_mean}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, TI_data)
    return lambda_dict

# TI_dict = get_TI_dict(1.0)
TI_dict_90_cov = get_TI_dict(0.9)
# TI_dict_75_cov = get_TI_dict(0.75)
# TI_dict_50_cov = get_TI_dict(0.5)
# TI_dict_25_cov = get_TI_dict(0.25)

all_TI = {0.9 : TI_dict_90_cov}