# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:27:58 2020

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
  real<lower=0> weekly_sd;
  real<lower=0> GI;
  real<lower=0> R0;
}

transformed parameters {
  vector[N] prediction=rep_vector(1e-5,N);
  vector<lower=0>[N] Rt;
  vector[N] SI_rev; // SI in reverse order
  {
   
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
    
  }
}
model {
  weekly_sd ~ normal(0,0.2);
  weekly_rho ~ normal(0.8, 0.05);
  weekly_rho1 ~ normal(0.1, 0.05);
  phi ~ normal(0,5);
  GI ~ normal(0.01,0.001);
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

all_samples = fit.to_dataframe()
all_samples.to_csv('C:/Users/iwona/Desktop/TI/bellman_harris//models_fits/' + 'AR2' + '.csv', index=False)

# prediction = all_samples[[col for col in all_samples if col.startswith('prediction')]]
# prediction_means = list(prediction.mean(axis=0))
# predictions_df = pd.read_csv('predictions_cases.csv')
# predictions_df['AR2'] = prediction_means
# predictions_df.to_csv('predictions_cases.csv', index = False)

sns.pairplot(all_samples[['phi','weekly_sd','weekly_rho','weekly_rho1', 'GI', 'R0']])
plt.title('pairplot for the model fit')
plt.show()

weekly_effect = all_samples[[col for col in all_samples if col.startswith('weekly_effect')]]
weekly_effect_means = list(weekly_effect.mean(axis=0))

prediction = all_samples[[col for col in all_samples if col.startswith('prediction')]] # for debugging
prediction_means = list(prediction.mean(axis=0))

rt = all_samples[[col for col in all_samples if col.startswith('Rt')]]
rt_means = list(rt.mean(axis=0))
rt_df = pd.read_csv('Rt_estimated.csv')
rt_df['AR2'] = rt_means
rt_df.to_csv('Rt_estimated.csv', index = False)

Rt = all_samples[[col for col in all_samples if col.startswith('Rt')]] # for debugging
RT_mean = list(Rt.mean(axis=0))

R0 = all_samples[[col for col in all_samples if col.startswith('R0')]] # for debugging
R0_mean = list(R0.mean(axis=0))
sns.distplot(R0.values, kde=False)
plt.title('R0')

phi = all_samples[[col for col in all_samples if col.startswith('phi')]] # for debugging
phi_mean = list(phi.mean(axis=0))
sns.distplot(phi.values, kde=False)
plt.title('phi')


SI_rev = all_samples[[col for col in all_samples if col.startswith('SI')]] # for debugging
SI_rev_means = list(SI_rev.mean(axis=0))

samples = all_samples[['phi', 'weekly_rho', 'weekly_rho1', 'weekly_sd', 'GI', 'R0', 'mu']]
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
    GI = theta_mean[4]
    R0 = theta_mean[5]
    weekly_effect = theta_mean[6:].values
    
    log_prior = 0
    log_likelihood = 0
    
    # log-prior
    log_prior += scipy.stats.norm(0,5).logpdf(phi)
    log_prior += scipy.stats.norm(0.01,0.001).logpdf(GI)
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

def LogLaplaceCovariance(theta_mean, cov):
     cov = np.diag(np.diag(cov))
     result = 1/2 * len(theta_mean) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(cov))
     result += Logf(theta_mean)
     # and now correction for the constraint
     # correction = 0
     # for i in range(6):
     #     sigma = np.sqrt(cov[i,i])
     #     mu = theta_mean[i]
     #     # print(round(mu,2), round(sigma,2), (1 + scipy.special.erf(mu/(np.sqrt(2)*sigma))) / 2)
     #     correction += np.log((1 + scipy.special.erf(mu/(np.sqrt(2)*sigma))) / 2)
     #     print(correction)
     # print(result, correction)
     return result# + correction
 

def LogLaplaceCovariance_plot(theta, cov, peak=theta_mean):
    f0 = Logf(theta_mean)
    # C = np.diag(np.diag(cov))
    C = cov
    tmp = np.dot(C, theta-peak)
    tmp1 = np.dot(theta-peak, tmp)
    return f0 - 0.5 * tmp1;

    
# plt.plot(np.arange(183), SI_rev_means)
# plt.plot(np.arange(183), SI_rev)

# # this is peffect
# plt.plot(np.arange(184), weekly_effect_means)
# plt.plot(np.arange(184), weekly_effect)

# # this is slightly lower
# plt.plot(np.arange(183), RT_mean)
# plt.plot(np.arange(183), Rt)


# plt.plot(np.arange(183), prediction_means, label = 'stan model')
# plt.plot(np.arange(183), prediction, label='mine')
# plt.legend()

    
def plot_along_one_dim(dim, xlabel):
    # theta_cp = theta_mean.copy()
    theta_cp = samples.iloc[2000, :]

    old_value = theta_cp[dim]
    
    x_dim = np.linspace(samples.iloc[:,dim].min()-0.5, samples.iloc[:,dim].max()+0.5, 100)
    # x_dim = np.linspace(0.1,10, 100)

    y_true = []
    y_laplace = []
    for x in x_dim:
        theta_cp[dim] = x
        y_true.append(Logf(theta_cp))
        y_laplace.append(LogLaplaceCovariance_plot(theta_cp, 1*cov))
    plt.plot(x_dim, y_laplace, label='reference')
    plt.scatter(x_dim, y_true, label='true posterior')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('posterior value')
    
plt.figure(figsize=(15,15))
plt.subplot(3,3,1)
plot_along_one_dim(0, 'phi')
plt.subplot(3,3,2)
plot_along_one_dim(1, 'weekly_rho')
plt.subplot(3,3,3)
plot_along_one_dim(2, 'weekly_rho1')
plt.subplot(3,3,4)
plot_along_one_dim(3, 'weekly_sd')
plt.subplot(3,3,5)
plot_along_one_dim(4, 'GI')
plt.subplot(3,3,6)
plot_along_one_dim(5, 'R0')
plt.subplot(3,3,7)
plot_along_one_dim(6+15-1, 'weekly_effect[15]')
plt.subplot(3,3,8)
plot_along_one_dim(6+60-1, 'weekly_effect[60]')
plt.subplot(3,3,9)
plot_along_one_dim(6+170-1, 'weekly_effect[170]')
plt.tight_layout()

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
     real weekly_sd = theta[4];
     real GI = theta[5];
     real R0 = theta[6];
     vector[N+1] weekly_effect = theta[7:(6+N+1)];
     
     
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
      matrix[6 + N + 1,6 + N + 1] C;
      vector[6 + N + 1] peak; // mean parameters estimate
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
  real<lower=0> GI;
  real<lower=0> R0;
    //vector<lower=0>[6 + N + 1] theta;
    
}

transformed parameters {
    vector[6 + N + 1] theta;
    theta[1] = phi;
    theta[2] = weekly_rho;
    theta[3] = weekly_rho1;
    theta[4] = weekly_sd;
    theta[5] = GI;
    theta[6] = R0;
    theta[7:(6+N+1)] = weekly_effect;

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

##############################################################################
# test for one lambda
TI_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1),
           'C': covinv, 'f0': Logf(theta_mean), 'peak': theta_mean, 'lambda': 0.0}

fit_TI = TI_BH_model.sampling(data=TI_data, iter=2000, seed=SEED, chains=CHAINS,
                     n_jobs=-1, control={'adapt_delta': 0.8})

print(fit_TI)
ti_df = fit_TI.to_dataframe()
# sns.pairplot(ti_df[['theta[1]','theta[2]','theta[3]','theta[4]','theta[5]', 'theta[6]']])

sns.pairplot(ti_df[['theta[1]','theta[2]','theta[3]','theta[4]','theta[5]',
                    'theta[6]', 'theta[15]', 'theta[60]', 'theta[170]']])

TI_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1),
           'C': covinv, 'f0': Logf(theta_mean), 'peak': theta_mean, 'lambda': 1.0}

fit_TI_1 = TI_BH_model.sampling(data=TI_data, iter=2000, seed=SEED, chains=CHAINS,
                     n_jobs=-1, control={'adapt_delta': 0.8})

ti_df_1 = fit_TI_1.to_dataframe()
sns.pairplot(ti_df_1[['theta[1]','theta[2]','theta[3]','theta[4]','theta[5]',
                    'theta[6]', 'theta[15]', 'theta[60]', 'theta[170]']])
##############################################################################



def get_expect_for_lambda(lam, data, n_iter=ITER):
    fit = TI_BH_model.sampling(data=data, iter=n_iter, n_jobs=-1, 
                                control = {'adapt_delta': 0.8}, seed=SEED)
    vals = fit.extract()['diff']
    expects = vals.mean()
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

def get_TI_dict():
    TI_data = {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1),
           'C': 2 * covinv, 'f0': Logf(theta_mean), 'peak': theta_mean}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, TI_data)
    return lambda_dict

TI_dict = get_TI_dict()

# TI_dict.update({0.0: get_expect_for_lambda(0.0, {'N': SKdata.shape[0], 'cases': SKdata.cases.values, 'x': np.arange(1,SKdata.shape[0]+1),
#            'C': covinv, 'f0': Logf(theta_mean), 'peak': theta_mean, 'lambda': 0.0}, n_iter=ITER)})

def get_logZextra(lambda_dict):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['vals'].mean())
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    plt.plot(xnew,ynew)
    plt.scatter(lambdaVals, expectsPerLambda)
    plt.xlim([0,1])
    plt.xlabel('lambda')
    plt.ylabel('expectation')
    plt.show()
    # calculate the integral of the interpolated line
    logzExtra = scipy.interpolate.splint(0, 1, tck, full_output=0)
    return logzExtra

Zlaplace = LogLaplaceCovariance(theta_mean, cov)
Zextra = get_logZextra(TI_dict)
ZTI = Zlaplace + Zextra

print('Laplace approximation:', Zlaplace)
print('TI result:', ZTI)


def add_rolling_stats(values):
    df = pd.DataFrame(columns = ['values', 'roll_mean', 'roll_std'])
    df['values'] = values
    df['roll_mean'] = df['values'].rolling(len(df),min_periods=1).mean()    
    df['roll_std'] = df['values'].rolling(len(df),min_periods=1).std()
    df['roll_sdErr'] = df['roll_std'] / np.sqrt(df.index)
    return df

def get_all_vals_rolling(lambda_dict):
    lambdaVals = list(lambda_dict.keys())
    df_rolling_means = pd.DataFrame(columns = lambdaVals)
    df_rolling_se = pd.DataFrame(columns = lambdaVals)
    df_rolling_std = pd.DataFrame(columns = lambdaVals)

    for lam in lambdaVals:
        df = add_rolling_stats(lambda_dict[lam]['vals'])
        df_rolling_means.loc[:,lam] = df['roll_mean'].values
        df_rolling_se.loc[:,lam] = df['roll_sdErr'].values
        df_rolling_std.loc[:,lam] = df['roll_std'].values

    df_rolling_means = df_rolling_means.dropna()
    df_rolling_se = df_rolling_se.dropna()
    df_rolling_std = df_rolling_se.dropna()

    return {'means': df_rolling_means, 'se': df_rolling_se,
            'std': df_rolling_std}
    
def plot_per_iter(meas, rolling_dict):
    if meas == 'mean':
        df = rolling_dict['means']
        ylab = 'rolling mean'
    elif meas == 'std':
        df = rolling_dict['std']
        ylab = 'standard deviation'
    else:
        df = rolling_dict['se']
        ylab = 'standard error'
    lambdaVals = list(df.columns)
    if lambdaVals[-1] == 'integral':
        lambdaVals = lambdaVals[:-1]
    for l in lambdaVals:
        plt.plot(df.index, df[l].values, label=round(l,1))
    if meas == 'se':
        plt.ylim([0, 0.5])
    plt.xlabel('iteration')
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    
    
TI_rolling = get_all_vals_rolling(TI_dict)
plot_per_iter('mean', TI_rolling)
plot_per_iter('se', TI_rolling)
plot_per_iter('std', TI_rolling)

def integral_value(lambdaVals, values):
    assert len(values) == len(lambdaVals)
    if lambdaVals[0] != 0.0:
        lambdaVals = [0.0] + lambdaVals
        values = [1.0] + list(values)
    tck = scipy.interpolate.splrep(lambdaVals, values, s=0)
    logzExtra_mean = scipy.interpolate.splint(lambdaVals[0], lambdaVals[-1], tck, full_output=0)
    return logzExtra_mean


def get_integrals(rolling_dict):
    df = rolling_dict['means']
    lambdaVals = list(df.columns)
    if lambdaVals[-1] == 'integral':
        lambdaVals = lambdaVals[:-1]
    df['integral'] = df[lambdaVals].apply(lambda x: integral_value(lambdaVals, x), axis=1)
    return df['integral'].values

integrals_TI = get_integrals(TI_rolling)
integrals_TI += Zlaplace

ar = np.arange(len(integrals_TI))
plt.plot(ar, integrals_TI)
# plt.fill_between(ar, integrals_TI - TI_rolling['std'].values,
#                  integrals_TI + TI_rolling['std'].values,
#                  alpha = 0.3, color='b', linewidth=0.1)
plt.xlabel('iteration')
plt.ylabel('evidence value')
plt.show()

print('Integral distribution mean:', integrals_TI.mean())
print('Integral distribution final value:',integrals_TI[-1])
print('Integral distribution sd:', integrals_TI.std())
print('Integral distribution standard error:', integrals_TI.std()/len(integrals_TI))
print('Integral distribution CI95:', np.quantile(integrals_TI, [0.025, 0.975], axis = 0))

