# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:40:00 2020

@author: iwona
"""


import pandas as pd
import pystan
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

MEAN = 7

ITER = 2000
SEED = 1234
CHAINS = 4
OUT_PATH = ''

GI = 1 / (2 * (np.sqrt(2/np.pi) * MEAN) **2)

SKdata = pd.read_csv('SouthKoreaData.csv')
stan_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1),
             'GI': GI}

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

fit = BH_model.sampling(data=stan_data, iter=ITER, seed=SEED, chains=CHAINS,
                     n_jobs=-1, control={'adapt_delta': 0.8})
print('Model fitting outputs')
print(fit)

all_samples = fit.to_dataframe()

weekly_effect = all_samples[[col for col in all_samples if col.startswith('weekly_effect')]]
weekly_effect_means = list(weekly_effect.mean(axis=0))

samples = all_samples[['phi', 'weekly_rho', 'weekly_rho1', 'weekly_sd', 'R0']]
samples = pd.concat([samples,weekly_effect], axis=1)
cov = np.cov(np.array(samples), rowvar = False)
theta_mean = samples.mean(axis = 0)
c = np.linalg.inv(np.linalg.cholesky(cov))
covinv = np.dot(c.T,c)

cases = SKdata.cases.values


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

def LogLaplaceCovariance(theta_mean, cov):
     result = 1/2 * len(theta_mean) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(cov))
     result += Logf(theta_mean)
     return result


##############################################################################
# TI part

TI_code = """
functions {
    
    real lf(vector theta, int N, int[] cases, real GI)
    {
     real Lprior = 0;
     real log_lik = 0;
     
     real phi = theta[1];
     real weekly_rho = theta[2];
     real weekly_rho1 = theta[3];
     real weekly_sd = theta[4];
     //real GI = theta[5];
     real R0 = theta[5];
     vector[N+1] weekly_effect = theta[6:(5+N+1)];
     
     
    // prepare the remaining parameters
    vector[N-1] we3_mu = weekly_effect[2:N]* weekly_rho + weekly_effect[1:(N-1)]* weekly_rho1;
    real we3_sigma = weekly_sd *sqrt(1-pow(weekly_rho,2)-pow(weekly_rho1,2) - 2 * pow(weekly_rho,2) * weekly_rho1/(1-weekly_rho1));

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
    for (i in 3:(N+1)){
     Lprior += normal_lpdf(weekly_effect[i] | we3_mu[i-2], we3_sigma);
    }
    Lprior += normal_lpdf(weekly_effect[2] | -1,weekly_sd *sqrt(1-pow(weekly_rho,2)-pow(weekly_rho1,2) - 2 * pow(weekly_rho,2) * weekly_rho1/(1-weekly_rho1)));
    Lprior += normal_lpdf(weekly_effect[1] | -1, 0.1);
 
    Lprior += normal_lpdf(weekly_sd|0,0.2);
    Lprior += normal_lpdf(weekly_rho|0.8, 0.05);
    Lprior += normal_lpdf(weekly_rho1|0.1, 0.05);
    Lprior += normal_lpdf(phi|0,5);
    //Lprior += normal_lpdf(GI|0.01,0.001);
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
    
    real path_lpdf(vector theta, real f0, vector peak, matrix C, int N, int[] cases, real GI, real lambda)
    {
    return lambda * lf(theta, N, cases, GI) + (1-lambda) * lfref(theta, f0, peak, C);
    } 
    }
                                       
data {  
      int<lower=1> N; // days of observed data for country m. each entry must be <= N
      int cases[N]; // reported deaths -- the rows with i > N contain -1 and should be ignored
      real x[N]; // x
      real GI;
      matrix[5 + N + 1,5 + N + 1] C;
      vector[5 + N + 1] peak; // mean parameters estimate
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
  real<lower=0> weekly_sd;
  //real<lower=0> GI;
  real<lower=0> R0;
    //vector<lower=0>[6 + N + 1] theta;
    
}

transformed parameters {
    vector[5 + N + 1] theta;
    theta[1] = phi;
    theta[2] = weekly_rho;
    theta[3] = weekly_rho1;
    theta[4] = weekly_sd;
    //theta[5] = GI;
    theta[5] = R0;
    theta[6:(5+N+1)] = weekly_effect;

    }
model {
    theta ~ path(f0, peak, C, N, cases, GI, lambda);
}

generated quantities {
    real logf = lf(theta, N, cases, GI);
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
    df.to_csv(OUT_PATH + str(MEAN) + '_' + str(lam) + '.csv', index=False)
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
           'C': (1/cov_frac) * covinv, 'f0': Logf(theta_mean), 'peak': theta_mean, 'GI' : GI}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, TI_data)
    return lambda_dict

# TI_dict = get_TI_dict(1.0)
TI_dict_90_cov = get_TI_dict(0.90)
# TI_dict_75_cov = get_TI_dict(0.75)
# TI_dict_50_cov = get_TI_dict(0.5)
# TI_dict_25_cov = get_TI_dict(0.25)

# all_TI = {0.9 : TI_dict_90_cov}
# all_TI = {1.0: TI_dict, 0.9: TI_dict_90_cov, 0.75: TI_dict_75_cov},
#           0.5: TI_dict_50_cov, 0.25: TI_dict_25_cov}


##############################################################################


# def add_rolling_stats(values):
#     df = pd.DataFrame(columns = ['values', 'roll_mean', 'roll_std'])
#     df['values'] = values
#     df['roll_mean'] = df['values'].rolling(len(df),min_periods=1).mean()    
#     df['roll_std'] = df['values'].rolling(len(df),min_periods=1).std()
#     df['roll_sdErr'] = df['roll_std'] / np.sqrt(df.index)
#     return df

# def get_all_vals_rolling(lambda_dict):
#     lambdaVals = list(lambda_dict.keys())
#     df_rolling_means = pd.DataFrame(columns = lambdaVals)
#     df_rolling_se = pd.DataFrame(columns = lambdaVals)
#     df_rolling_std = pd.DataFrame(columns = lambdaVals)

#     for lam in lambdaVals:
#         df = add_rolling_stats(lambda_dict[lam]['vals'])
#         df_rolling_means.loc[:,lam] = df['roll_mean'].values
#         df_rolling_se.loc[:,lam] = df['roll_sdErr'].values
#         df_rolling_std.loc[:,lam] = df['roll_std'].values

#     df_rolling_means = df_rolling_means.dropna()
#     df_rolling_se = df_rolling_se.dropna()
#     df_rolling_std = df_rolling_se.dropna()

#     return {'means': df_rolling_means, 'se': df_rolling_se,
#             'std': df_rolling_std}

# all_rolling = {}
# for k in all_TI.keys():
#     all_rolling.update({k: get_all_vals_rolling(all_TI[k])})


# def plot_per_iter_lambdaZero(meas, all_rolling_dict):
#     for k in all_rolling_dict.keys():
#         rolling_dict = all_rolling_dict[k]
#         if meas == 'mean':
#             df = rolling_dict['means']
#             ylab = 'rolling mean'
#         elif meas == 'std':
#             df = rolling_dict['std']
#             ylab = 'standard deviation'
#         else:
#             df = rolling_dict['se']
#             ylab = 'standard error'

#         plt.plot(df.index, df[0.0].values, label=str(k) + '* covariance')
#     plt.xlabel('iteration')
#     plt.ylabel(ylab)
#     plt.title('Expectation per iteration for $\lambda=0.0$')
#     plt.legend()
#     plt.show()
    
# plot_per_iter_lambdaZero('mean', all_rolling)

# def integral_value(lambdaVals, values):
#     assert len(values) == len(lambdaVals)
#     if lambdaVals[0] != 0.0:
#         lambdaVals = [0.0] + lambdaVals
#         values = [1.0] + list(values)
#     tck = scipy.interpolate.splrep(lambdaVals, values, s=0)
#     logzExtra_mean = scipy.interpolate.splint(lambdaVals[0], lambdaVals[-1], tck, full_output=0)
#     return logzExtra_mean


# def get_integrals(rolling_dict):
#     df = rolling_dict['means']
#     lambdaVals = list(df.columns)
#     if lambdaVals[-1] == 'integral':
#         lambdaVals = lambdaVals[:-1]
#     df['integral'] = df[lambdaVals].apply(lambda x: integral_value(lambdaVals, x), axis=1)
#     return df['integral'].values

# integrals_TI = {}
# for k in all_TI.keys():
#     integrals_TI_tmp = get_integrals(all_rolling[k])
#     integrals_TI_tmp += LogLaplaceCovariance(theta_mean, k * cov)
#     integrals_TI.update({k: integrals_TI_tmp})
    

# for k in all_TI.keys():
#     ar = np.arange(len(integrals_TI[k]))
#     plt.plot(ar, integrals_TI[k], label=str(k) + '* covariance')
# plt.legend()
# plt.xlabel('iteration')
# plt.ylabel('evidence value')
# plt.show()


# def get_logZextra(all_TI_dict):
#     """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
#     And plots it too
#     """
#     for k in all_TI_dict:
#         lambda_dict = all_TI_dict[k]
#         lambdaVals = list(lambda_dict.keys())
#         expectsPerLambda = []
#         for lam in lambdaVals:
#             expectsPerLambda.append(lambda_dict[lam]['vals'].mean())
#         tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
#         xnew = np.linspace(0, 1)
#         ynew = scipy.interpolate.splev(xnew, tck, der=0)
#         plt.plot(xnew,ynew, label = str(k) + '* cov')
#         plt.scatter(lambdaVals, expectsPerLambda)
#     plt.xlim([0,1])
#     plt.xlabel('lambda')
#     plt.ylabel('expectation')
#     plt.legend()
#     plt.show()

# get_logZextra(all_TI)

# print('Integral distribution mean:', integrals_TI.mean())
# print('Integral distribution final value:',integrals_TI[-1])
# print('Integral distribution sd:', integrals_TI.std())
# print('Integral distribution standard error:', integrals_TI.std()/len(integrals_TI))
# print('Integral distribution CI95:', np.quantile(integrals_TI, [0.025, 0.975], axis = 0))

