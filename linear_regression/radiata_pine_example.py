# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:37:51 2020

@author: ieh19
"""


""""This script demonstrates the use of TI on the Radiata Pine dataset.
The model follows Friel and Wyse, 2011.
For this model, we can calculate the evidence analytically to get an exact solution.
"""

import pandas as pd
import pystan
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

sns.set()

ITER = 2000
SEED = 1234

######################################################
# load data from the Friel example

PineData = pd.read_csv('C:/Users/iwona/Desktop/TI/data_examples/radiata_friel_data.csv')
PineData.drop(columns=['id'], inplace = True)
y = PineData.loc[:,'y'].values.astype(int)
model1 = PineData.loc[:,'x'].values
model2 = PineData.loc[:,'z'].values
mean_x = np.mean(model1)
mean_z = np.mean(model2)
model1 = model1 - mean_x
model2 = model2 - mean_z

print("Data loaded")

######################################################
#### Stan model to get the posterior distribution    
code = """
data {
    int N;
    vector[N] X;
    vector[N] y;
    matrix[2,2] C;
    vector[2] mu0;
    
}
parameters {
    vector<lower=0>[2] beta;
    real<lower=0> tau;
}
transformed parameters {
    real tauinvers = 1/tau;  // risk when tau = 0
    real sqrttauinvers = sqrt(tauinvers);
    }
model {
    tau ~ gamma(3, 2 * 300*300);
    beta ~ multi_normal(mu0, tauinvers * C);
    y ~ normal(beta[1] + beta[2] * X, sqrttauinvers);
}
"""

model = pystan.StanModel(model_code=code)
print('Stan model compiled')

def fit_posterior(data_x, data_y):
    data = {'N': data_x.shape[0], 'y': data_y, 'X': data_x,
            'C': np.array([[1/0.06,0],[0,1/6.0]]),'mu0': [3000,185]}
    fit = model.sampling(data=data, iter=10000, n_jobs=-1, seed = SEED)
    print(fit)
    df = fit.to_dataframe()
    samples = df.loc[:, ['beta[1]', 'beta[2]', 'tau']].values
    cov = np.cov(np.array(samples[:,[0,1,2]]), rowvar = False)
    theta_mean = samples.mean(axis = 0)
    c = np.linalg.inv(np.linalg.cholesky(cov))
    covinv = np.dot(c.T,c)
    print(np.log(np.linalg.det(cov)))
    print(np.log(np.linalg.det(covinv)))

    return {'mu': theta_mean, 'cov': cov, 'covinv': covinv,
            'x': data_x, 'y': data_y}
    
posterior1 = fit_posterior(model1, y)
posterior2 = fit_posterior(model2, y)


print("Both posteriors fitted")

def Logf(posterior):
    theta = posterior['mu']
    alpha, beta, tau = theta
    vecpar = [alpha, beta]
    mu0 = [3000, 185]
    cov = 1/tau * np.array([[1/0.06,0],[0,1/6.0]])
    x = posterior['x']
    y = posterior['y']
    N = len(y)
    # log prior = sum of priors for each parameter
    log_prior = scipy.stats.gamma(a = 6, scale = 1/(4 * 300*300)).logpdf(tau)
    log_prior += scipy.stats.multivariate_normal.logpdf(vecpar,mu0, cov)

    # log likelihood = sum of log likelihoods
    lp = ((y - (alpha + beta * x))**2).sum()
    log_likelihood = - N/2 * np.log(2*np.pi)
    log_likelihood += N/2 * np.log(tau)
    log_likelihood += -1/2 * tau * lp

    return log_likelihood + log_prior                                                                

def LogLaplaceCovariance(posterior):
    
     result = 1/2 * len(posterior['mu']) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(posterior['cov']))
     result += Logf(posterior)
     return result

print('log zRef model 1 = ', LogLaplaceCovariance(posterior1))
print('log zRef model 2 = ', LogLaplaceCovariance(posterior2))
print('Analytical solution: -310.1280 for model 1, -301.7046 for model 2')

x = model1
plt.scatter(x,y)
plt.plot(x,posterior1['mu'][0] + posterior1['mu'][1]*x)
plt.show()

x = model2
plt.scatter(x,y)
plt.plot(x,posterior2['mu'][0] + posterior2['mu'][1]*x)
plt.show()

###### TI ##########################################
stan_TI_code = """
functions {
    
    real lf(vector theta, real[] Y, vector X, int K, vector mu0, matrix cov_prior)
    {
    real log_lik = normal_lpdf(Y | theta[1] + theta[2] * X, sqrt(1/theta[3]));
     
    real Lprior = 0;
    Lprior += gamma_lpdf(theta[3] | 3, 2 * 300 * 300);
    Lprior +=  multi_normal_lpdf([theta[1], theta[2]] | mu0, 1/theta[3] * cov_prior);

    return log_lik + Lprior;
    }
                     
    real lfref(vector theta, real f0, vector peak, matrix C)
    {
     return f0 - 0.5 * (theta-peak)' * C * (theta-peak);
     }
    
    real path_lpdf(vector theta, real f0, vector peak, matrix C, real[] Y, vector X, int K, real lambda, matrix cov_prior, vector mu0)
    {
    return lambda * lf(theta, Y, X, K, mu0, cov_prior) + (1-lambda) * lfref(theta, f0, peak, C);
    } 
    }
                                       
data {
      int K; // number of variables
      matrix[K, K] C; // inverse of the covariates matrix
      vector[K] peak; // typically mean or mode
      real f0; // log(f(peak))
      real lambda; // used for creating a path between f and f_ref
      int N; //the number of  observations                                                       
      real Y[N]; //the response 
      vector[N] X; //the model matrix (all observations)
      matrix[2,2] cov_prior;
      vector[2] mu0;
      }

transformed data {}

parameters {
    vector<lower=0>[3] theta;
}

transformed parameters {}

model {
       theta ~ path(f0, peak, C, Y, X, K, lambda, cov_prior, mu0);
}

generated quantities {
    real logf = lf( theta, Y, X, K, mu0, cov_prior);
    real logfref = lfref(theta, f0, peak, C);
    real diff = logf - logfref;
    }
"""


t0 = time.time()
m = pystan.StanModel(model_code=stan_TI_code)
print(round(time.time() - t0,2), 'seconds elapsed for the stan model compilation')

# # testing for one lambda
# posterior = posterior1
# data = {'K': len(posterior['mu']),'C': posterior['covinv'], 
#             'peak': posterior['mu'], 'f0': Logf(posterior), 
#             'N': posterior['x'].shape[0], 'Y': posterior['y'], 
#             'X': posterior['x'], 'cov_prior': np.array([[1/0.06,0],[0,1/6.0]]), 'mu0': [3000,185]}
# lam = 0.0
# data.update({'lambda': lam})
# fit = m.sampling(data=data, iter=10000, n_jobs=-1)
# fit.plot()
# df = fit.to_dataframe()
# vals = fit.extract()['diff']
# vals.mean()

def get_expect_for_lambda(lam, data, n_iter):
    n_iter = n_iter
    data.update({'lambda': lam})
    fit = m.sampling(data=data, iter=n_iter, n_jobs=-1, seed = SEED)
    vals = fit.extract()['diff']
    expects = vals.mean()
    # quantiles_list = [0.025, 0.25, 0.75, 0.975]
    # quantiles = np.quantile(vals, quantiles_list, axis = 0)
    print(lam, ', expectation = ', expects)
    # variance = vals.var()
    # fitsum = fit.summary()
    # n_eff = fitsum['summary'][-1,-2] # n_eff for lp__
    # sdErrs = np.sqrt(variance/n_eff)
    return {'expects': expects, 'vals': vals}
    # return {'expects': expects, 'variance': variance, 
    #                 'sdErrs': sdErrs, 'steps': n_iter,
    #                 'values_used': len(vals),
    #                 'n_eff': n_eff,
    #                 'quantiles': quantiles}

def MCMC_for_all_lambdas(lambdaVals, data, n_iter):
    """Execute TI MCMC for multiple lambdas.
    lambdaVals: list of values of lambdas"""
    t0 = time.time()
    lambdaOutput = {}
    for l in lambdaVals:
        lam = round(l,5)
        lambdaOutput.update({lam: get_expect_for_lambda(lam, data, n_iter)})
    print(time.time() - t0, 'seconds elapsed')
    return lambdaOutput

# this still doesn't work but is not really necessary anymore
# lambdaVals = np.arange(0,1,0.1)
# lambdaOutput={}
# def MCMC_for_one_lambdas(lam):
#     lam = round(lam,1)
#     lambdaOutput.update({lam: get_expect_for_lambda(lam)})
#     return lambdaOutput

# def get_lambda_dict(MCMC_for_one_lambdas, lambdaVals):
#     pool=multiprocessing.Pool(8)
#     poolout=pool.map(MCMC_for_one_lambdas, lambdaVals)
#     lambda_dict=dict((key,d[key]) for d in poolout for key in d)
#     pool.close()
#     return lambda_dict


# lambda_dict=get_lambda_dict(MCMC_for_one_lambdas, lambdaVals)


def get_logZextra(lambda_dict):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    # sdErrsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['expects'])
        # sdErrsPerLambda.append(lambda_dict[lam]['sdErrs'])
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    # sdErLow = [a_i - b_i for a_i, b_i in zip(expectsPerLambda, sdErrsPerLambda)]
    # tck_low = scipy.interpolate.splrep(lambdaVals, sdErLow, s=0)
    # sdErHigh = [a_i + b_i for a_i, b_i in zip(expectsPerLambda, sdErrsPerLambda)]
    # tck_high = scipy.interpolate.splrep(lambdaVals, sdErHigh, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    # ylow = scipy.interpolate.splev(xnew, tck_low, der=0)
    # yhigh = scipy.interpolate.splev(xnew, tck_high, der=0)
    plt.plot(xnew,ynew)
    plt.scatter(lambdaVals, expectsPerLambda)
    # plt.fill_between(xnew,ylow,yhigh, alpha = 0.5)
    plt.xlim([0,1])
    plt.xlabel('lambda')
    plt.ylabel('expectation')
    plt.show()
    # calculate exponent of the integral of the interpolated line
    logzExtra = scipy.interpolate.splint(lambdaVals[0], lambdaVals[-1], tck, full_output=0)
    return logzExtra

def plot_logZextra_quantiles(lambda_dict):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    lambdaVals = list(lambda_dict.keys())
    quantiles_list = [0.025, 0.25, 0.75, 0.975]
    quantilesPerLambda = []
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['expects'])
        quantilesPerLambda.append(lambda_dict[lam]['quantiles'])
    allQuantiles = np.array(quantilesPerLambda)
    l = ['95% CI', '50% CI']
    for i in range(int(len(quantiles_list)/2)):
        vals_low = allQuantiles[:,i]
        vals_high = allQuantiles[:,-i-1]
        tck_low = scipy.interpolate.splrep(lambdaVals, vals_low, s=0)
        tck_high = scipy.interpolate.splrep(lambdaVals, vals_high, s=0)
        xnew = np.linspace(0, 1)
        ynew_low = scipy.interpolate.splev(xnew, tck_low, der=0)
        ynew_high = scipy.interpolate.splev(xnew, tck_high, der=0)
        plt.fill_between(xnew,ynew_low,ynew_high, alpha = 0.5, label = l[i])
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    plt.plot(xnew,ynew, label='mean')
    plt.scatter(lambdaVals, expectsPerLambda)
    plt.xlim([0,1])
    # plt.ylim([-0.07,0.05])
    plt.xlabel('lambda')
    plt.ylabel('expectation')
    plt.legend()
    plt.show()


def TI_dict(posterior, n_iters):
    data = {'K': len(posterior['mu']),'C': posterior['covinv'], 
            'peak': posterior['mu'], 'f0': Logf(posterior), 
            'N': posterior['x'].shape[0], 'Y': posterior['y'], 
            'X': posterior['x'], 'cov_prior': np.array([[1/0.06,0],[0,1/6.0]]), 'mu0': [3000,185]}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, data, n_iters)
    return lambda_dict

### Run TI and print the results

TI_dict_1 = TI_dict(posterior1, ITER)
print("TI for model 1 done")
TI_dict_2 = TI_dict(posterior2, ITER)
print("TI for model 2 done")


logzExtra_covariance1 = get_logZextra(TI_dict_1)
# plot_logZextra_quantiles(TI_dict_1)
logzExtra_covariance2 = get_logZextra(TI_dict_2)
# plot_logZextra_quantiles(TI_dict_2)

logzRefCovariance1 = LogLaplaceCovariance(posterior1)
logzRefCovariance2 = LogLaplaceCovariance(posterior2)

logzTI_covariance1 = logzExtra_covariance1 + logzRefCovariance1
logzTI_covariance2 = logzExtra_covariance2 + logzRefCovariance2

BF12_LaplaceCovariance = np.exp(logzRefCovariance1 - logzRefCovariance2)
BF12_TICovariance = np.exp(logzTI_covariance1 - logzTI_covariance2)

BF21_LaplaceCovariance = np.exp(logzRefCovariance2 - logzRefCovariance1)
BF21_TICovariance = np.exp(logzTI_covariance2 - logzTI_covariance1)

results_Laplace = ['Laplace', logzRefCovariance1, logzRefCovariance2, BF21_LaplaceCovariance]
results_TI = ['TI', logzTI_covariance1, logzTI_covariance2, BF21_TICovariance]
print(tabulate([results_Laplace, results_TI], headers=['Method', 'log post M1', 'Log post M2', 'BF21']))
print('Analytical solution: -310.1280 for model 1, -301.7046 for model 2')

evidence_dict_M1 = {'refTI': TI_dict_1}
evidence_dict_M2 = {'refTI': TI_dict_2}

###### TI with direct path between two models ################################
stan_TI_simult_code = """
functions {
    
    real lf(vector theta, real[] Y, vector X, int K, vector mu0, matrix cov_prior)
    {
    real log_lik = normal_lpdf(Y | theta[1] + theta[2] * X, sqrt(1/theta[3]));
     
    real Lprior = 0;
    Lprior += gamma_lpdf(theta[3] | 3, 2 * 300 * 300);
    Lprior +=  multi_normal_lpdf([theta[1], theta[2]] | mu0, 1/theta[3] * cov_prior);

    return log_lik + Lprior;
    }
                     
    
    real path_lpdf(vector theta, real[] Y, vector X1, vector X2, int K, real lambda, matrix cov_prior, vector mu0)
    {
    return lambda * lf(theta, Y, X2, K, mu0, cov_prior) + (1-lambda) * lf(theta, Y, X1, K, mu0, cov_prior);
    } 
    }
                                       
data {
      int K; // number of variables
      real lambda; // used for creating a path between f and f_ref
      int N; //the number of  observations                                                       
      real Y[N]; //the response 
      vector[N] X1; //the model matrix (all observations)
      vector[N] X2; //the model matrix (all observations)
      matrix[2,2] cov_prior;
      vector[2] mu0;
      }

transformed data {}

parameters {
    vector<lower=0>[3] theta;
}

transformed parameters {}

model {
       theta ~ path(Y, X1, X2, K, lambda, cov_prior, mu0);
}

generated quantities {
    real logf1 = lf( theta, Y, X1, K, mu0, cov_prior);
    real logf2 = lf( theta, Y, X2, K, mu0, cov_prior);
    real diff = logf2 - logf1;
    }
"""


t0 = time.time()
m_simul = pystan.StanModel(model_code=stan_TI_simult_code)
print(round(time.time() - t0,2), 'seconds elapsed for the stan model compilation')

def get_expect_for_lambda(lam, data, n_iter):
    n_iter = n_iter
    data.update({'lambda': lam})
    fit = m_simul.sampling(data=data, iter=n_iter, n_jobs=-1, seed = SEED)
    vals = fit.extract()['diff']
    expects = vals.mean()
    print(lam, ', expectation = ', expects)
    return {'expects': expects, 'vals': vals}

def TI_dict(n_iters):
    data = {'K': len(posterior1['mu']), 
            'N': posterior1['x'].shape[0], 'Y': posterior1['y'], 
            'X1': posterior1['x'], 'X2': posterior2['x'], 
            'cov_prior': np.array([[1/0.06,0],[0,1/6.0]]), 'mu0': [3000,185]}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, data, n_iters)
    return lambda_dict

### Run TI and print the results

TI_dict_simul = TI_dict(ITER)
print("TI for simultaneous TI done")
logzExtra_covariance_simul = get_logZextra(TI_dict_simul)

print('BF with path built between two models', np.exp(logzExtra_covariance_simul))
print('BF with path built with references   ', BF21_TICovariance)
############## POWER POSTERIOR #############################################
stan_PP_code = """
functions {
    real lf(vector theta, real[] Y, vector X)
    {
    return normal_lpdf(Y | theta[1] + theta[2] * X, sqrt(1/theta[3]));
    }

    real path_lpdf(vector theta, real[] Y, vector X, real lambda, vector mu0, matrix cov_prior)
    {
    real Lprior = 0;
    Lprior += gamma_lpdf(theta[3] | 3, 2 * 300 * 300);
    Lprior +=  multi_normal_lpdf([theta[1], theta[2]] | mu0, 1/theta[3] * cov_prior);

    return lambda * lf(theta, Y, X) + Lprior;
    } 
    }
                                       
data {
      int K; // number of variables
      real lambda; // used for creating a path between f and f_ref
      int N; //the number of  observations                                                       
      real Y[N]; //the response 
      vector[N] X; //the model matrix (all observations)
      matrix[2,2] cov_prior;
      vector[2] mu0;
      }

transformed data {}

parameters {
    vector<lower=0>[K] theta;
}

transformed parameters {}

model {
        theta ~ path(Y, X, lambda, mu0, cov_prior);
}

generated quantities {
    real logf = 0;
    {
     real Lprior = 0;
    Lprior += gamma_lpdf(theta[3] | 3, 2 * 300 * 300);
    Lprior +=  multi_normal_lpdf([theta[1], theta[2]] | mu0, 1/theta[3] * cov_prior);
    logf = lf(theta, Y, X);
     }
    }
"""

t0 = time.time()
mpp = pystan.StanModel(model_code=stan_PP_code)
print(round(time.time() - t0,2), 'seconds elapsed for the stan model compilation')


def get_expect_for_lambda_PP(lam, data, n_iter):
    n_iter = n_iter
    data.update({'lambda': lam})
    fit = mpp.sampling(data=data, iter=n_iter, n_jobs=-1, seed = SEED, warmup  = int(n_iter * 0.2))
    vals = fit.extract()['logf']
    expects = vals.mean()
    # print(lam, ', expectation = ', expects)
    return {'expects': expects, 'vals': vals}

def MCMC_for_all_lambdas_PP(lambdaVals, data, n_iter):
    """Execute TI MCMC for multiple lambdas.
    lambdaVals: list of values of lambdas"""
    t0 = time.time()
    lambdaOutput = {}
    for l in lambdaVals:
        lam = round(l,10)
        # lam = round(l,1)
        lambdaOutput.update({lam: get_expect_for_lambda_PP(lam, data, n_iter)})
    print(time.time() - t0, 'seconds elapsed')
    return lambdaOutput

def PP_dict(posterior, n_iters):
    data = {'K': len(posterior['mu']),
            'N': posterior['x'].shape[0], 'Y': posterior['y'], 
            'X': posterior['x'], 'cov_prior': np.array([[1/0.06,0],[0,1/6.0]]),
            'mu0': [3000,185]}
    # lambdaVals = np.arange(0.0,1.1,0.1)
    lambdaVals = np.arange(0.0,1.01,0.01)**5  # Friel&Wyse(2011)
    lambda_dict = MCMC_for_all_lambdas_PP(lambdaVals, data, n_iters)
    return lambda_dict


PP_dict_1 = PP_dict(posterior1, ITER)
print("PP for model 1 done")
PP_dict_2 = PP_dict(posterior2, ITER)
print("PP for model 2 done")

PP_dict_1_full_test = PP_dict(posterior1, 5000)
PP_dict_2_full_test = PP_dict(posterior2, 5000)

PP_dict_1_full = PP_dict(posterior1, ITER)
print("PP for model 1 done")
PP_dict_2_full = PP_dict(posterior2, ITER)
print("PP for model 2 done")

def get_logZextra_PP(lambda_dict):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    # sdErrsPerLambda = [0]
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['expects'])
        # sdErrsPerLambda.append(lambda_dict[lam]['sdErrs'])
    # print(sdErrsPerLambda)
    # lambdaVals = [0] + lambdaVals
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    # sdErLow = [a_i - b_i for a_i, b_i in zip(expectsPerLambda, sdErrsPerLambda)]
    # tck_low = scipy.interpolate.splrep(lambdaVals, sdErLow, s=0)
    # sdErHigh = [a_i + b_i for a_i, b_i in zip(expectsPerLambda, sdErrsPerLambda)]
    # tck_high = scipy.interpolate.splrep(lambdaVals, sdErHigh, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    # ylow = scipy.interpolate.splev(xnew, tck_low, der=0)
    # yhigh = scipy.interpolate.splev(xnew, tck_high, der=0)
    plt.plot(xnew,ynew)
    plt.scatter(lambdaVals, expectsPerLambda)
    # plt.fill_between(xnew,ylow,yhigh, alpha = 0.5)
    plt.xlim([0,1])
    plt.xlabel('lambda')
    plt.ylabel('expectation')
    plt.show()
    # calculate exponent of the integral of the interpolated line
    logzExtra = scipy.interpolate.splint(0, 1, tck, full_output=0)
    return logzExtra

logpp_1 = get_logZextra_PP(PP_dict_1)
logpp_2 = get_logZextra_PP(PP_dict_2)
BF12_PowerPosterior = np.exp(logpp_1) / np.exp(logpp_2)
BF21_PowerPosterior = np.exp(logpp_2) / np.exp(logpp_1)

print(logpp_1, logpp_2)
results_PP= ['Power Posterior', logpp_1, logpp_2, BF21_PowerPosterior]
print(tabulate([results_Laplace, results_TI, results_PP], headers=['Method', 'log post M1', 'Log post M2', 'BF21']))
print('Analytical solution: -310.1280 for model 1, -301.7046 for model 2')

############## Plots  #############################################
def plot_lambdaExp(lambda_dict, label, title):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['expects'])
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    plt.plot(xnew,ynew, label = label)
    plt.scatter(lambdaVals, expectsPerLambda)
    plt.xlim([-0.05,1.05])
    plt.xlabel('$\lambda$')
    plt.ylabel('Expectation')
    plt.legend(loc = 'lower right')
    # plt.title(title)

plt.figure()
plot_lambdaExp(TI_dict_1, '$M_1$', 'Referenced TI')
plot_lambdaExp(TI_dict_2, '$M_2$', 'Referenced TI')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataLambdasTI.pdf', format='pdf', bbox_inches = "tight")

plt.figure()
plot_lambdaExp(TI_dict_simul, 'path from $M_1$ to $M_2$', 'TI')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataLambdasTI_direct.pdf', format='pdf', bbox_inches = "tight")

plt.figure()
plot_lambdaExp(PP_dict_1, '$M_1$', 'PP$_{11}$')
plot_lambdaExp(PP_dict_2, '$M_2$', 'PP$_{11}$')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataLambdasPP11.pdf', format='pdf', bbox_inches = "tight")


plt.figure()
plot_lambdaExp(PP_dict_1_full, '$M_1$', 'PP$_{100}$')
plot_lambdaExp(PP_dict_2_full, '$M_2$', 'PP$_{100}$')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataLambdasPP100.pdf', format='pdf', bbox_inches = "tight")


############## NESTED SAMPLING #############################################
## pip install dynesty
import dynesty
# log-likelihood
x = posterior1['x']
y = posterior1['y']
N = len(y)

def loglike(theta):
    # print(theta)
    alpha, beta, tau = theta
    lp = ((y - (alpha + beta * x))**2).sum()
    log_likelihood = - N/2 * np.log(2*np.pi)
    log_likelihood += N/2 * np.log(tau)
    log_likelihood += -1/2 * tau * lp
    return log_likelihood

        
# these priors don't seem to work...
# prior transform
def prior_transform(utheta):
    xx = np.array(utheta)  # copy u
    theta = [] 
    alpha, beta, tau = utheta
    mu0 = np.array([3000, 185])
    tau_1 = scipy.stats.gamma(a = 6, scale = 1/(4 * 300*300)).ppf(tau)

    cov = 1/tau_1 * np.array([[1/0.06,0],[0,1/6.0]])

    # Bivariate Normal
    t = scipy.stats.norm.ppf(utheta[0:2])  # convert to standard normal
    xx[0:2] = np.dot(cov, t)  # correlate with appropriate covariance

    # xx[0:2] = np.dot(scipy.linalg.sqrtm(cov), t)  # correlate with appropriate covariance
    xx[0:2] += mu0  # add mean
    theta.append(xx[0])
    theta.append(xx[1])
    
    theta.append(tau_1)
    
    return theta

dsampler = dynesty.NestedSampler(loglike, prior_transform, ndim=3, nlive=1500)
dsampler.run_nested()
dres = dsampler.results
dres.summary()
logz1 = dres.logz[-1]
err1 = dres.logzerr[-1]

x = posterior2['x']
y = posterior2['y']
dsampler = dynesty.NestedSampler(loglike, prior_transform, ndim=3, nlive=1500)
dsampler.run_nested()
dres = dsampler.results
dres.summary()
logz2 = dres.logz[-1]
err2 = dres.logzerr[-1]

BF12_NestedSampling = np.exp(logz1) / np.exp(logz2)

results_Nested= ['Nested Sampling', logz1, logz2, BF12_NestedSampling]
print(tabulate([results_Laplace, results_TI, results_Nested], headers=['Method', 'log post M1', 'Log post M2', 'BF12']))

##############################################################################
evidence_dict = {'refTI_1': TI_dict_1, 'refTI_2': TI_dict_2, 
                    'refTIdirect': TI_dict_simul,
                    'PP_1': PP_dict_1, 'PP_2': PP_dict_2,
                    'PP_1_full': PP_dict_1_full, 'PP_2_full': PP_dict_2_full}



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

all_rolling = {}
for k in evidence_dict.keys():
    all_rolling.update({k: get_all_vals_rolling(evidence_dict[k])})
    
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

integrals_TI = {}
for k in evidence_dict.keys():
    print(k)
    integrals_TI_tmp = get_integrals(all_rolling[k])
    if k == 'refTI_1':
        integrals_TI_tmp += LogLaplaceCovariance(posterior1)
    if k == 'refTI_2':
        integrals_TI_tmp += LogLaplaceCovariance(posterior2)
    integrals_TI.update({k: integrals_TI_tmp})
    

for k in evidence_dict.keys():
    if k != 'refTIdirect' and k!= 'PP_1' and k!= 'PP_2':
        ls = '-'
        ar = np.arange(len(integrals_TI[k]))
        if k == 'refTI_1':
            label = 'ref TI $M_1$'
            ls = '--'
        elif k == 'refTI_2':
            label = 'ref TI $M_2$'
            ls = '--'
        elif k == 'PP_1_full':
            label = 'PP $M_1$'
        elif k == 'PP_2_full':
            label = 'PP $M_2$'
        plt.plot(ar, integrals_TI[k], label=label, ls=ls)
plt.hlines(-310.1280, ar[0], ar[-1], label = '$M_1$ exact', color = 'magenta')
plt.hlines(-301.7046, ar[0], ar[-1], label = '$M_2$ exact', color = 'black')
plt.legend(ncol = 2)
plt.xlabel('Iteration')
plt.ylabel('Log-evidence')
plt.xlim([0,100])
# plt.show()
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataEvidenceIter.pdf', format='pdf')

for k in evidence_dict.keys():
    if k != 'refTIdirect':
        ls = '-'
        ar = np.arange(len(integrals_TI[k]))
        if k == 'refTI_1':
            label = 'ref TI $M_1$'
            ls = '--'
        elif k == 'refTI_2':
            label = 'ref TI $M_2$'
            ls = '--'
        elif k == 'PP_1':
            label = 'PP $M_1$'
        elif k == 'PP_2': 
            label = 'PP $M_2$'
        elif k == 'PP_1_full':
            label = 'PP full $M_1$'
        elif k == 'PP_2_full': 
            label = 'PP full $M_2$'

k = 'refTI_1'
ar = np.arange(len(integrals_TI[k]))
plt.plot(ar, integrals_TI[k], label='ref TI $M_1$', ls=ls)
k = 'PP_1_full'
ar = np.arange(len(integrals_TI[k]))
plt.plot(ar, integrals_TI[k], label='PP$_{100}$ $M_1$', ls=ls)
plt.hlines(-310.1280, ar[0], ar[-1], label = '$M_1$ exact', color = 'tab:red', ls = '--')
plt.legend(ncol = 1, loc = 'lower right')
plt.xlabel('Iteration')
plt.ylabel('Log-evidence')
# plt.ylim([-310,-310.5])
plt.xlim([0,1500])
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataEvidenceIterM1.pdf', format='pdf', bbox_inches = "tight")

k = 'refTI_2'
ar = np.arange(len(integrals_TI[k]))
plt.plot(ar, integrals_TI[k], label='ref TI $M_2$', ls=ls)
k = 'PP_2_full'
ar = np.arange(len(integrals_TI[k]))
plt.plot(ar, integrals_TI[k], label='PP$_{100}$ $M_2$', ls=ls)
plt.hlines(-301.7046, ar[0], ar[-1], label = '$M_2$ exact', color = 'tab:red', ls = '--')
plt.legend(ncol = 1, loc = 'lower right')
# plt.ylim([-301,-302])
plt.xlim([0,1500])
plt.xlabel('Iteration')
plt.ylabel('Log-evidence')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataEvidenceIterM2.pdf', format='pdf', bbox_inches = "tight")

k = 'PP_1'
ar = np.arange(len(integrals_TI[k]))
plt.plot(ar, integrals_TI[k], label='PP$_{11}$ for $M_1$', ls=ls)
k = 'PP_2'
ar = np.arange(len(integrals_TI[k]))
plt.plot(ar, integrals_TI[k], label='PP$_{11}$ for $M_2$', ls=ls)
plt.hlines(-310.1280, ar[0], ar[-1], label = '$M_1$ exact', color = 'tab:blue', ls = '--')
plt.hlines(-301.7046, ar[0], ar[-1], label = '$M_2$ exact', color = 'tab:orange', ls = '--')
plt.legend(ncol = 2, loc = 'lower right')
plt.xlabel('Iteration')
plt.ylabel('Log-evidence')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataEvidenceIterPP11.pdf', format='pdf', bbox_inches = "tight")

##############################################################################
# standard errors plots
sdErrors_TI = {}
for k in integrals_TI.keys():
    df = pd.DataFrame(columns = ['integral', 'std', 'sdErr'])
    df['integral'] = integrals_TI[k]
    df['std'] = df['integral'].rolling(len(df['integral'].values),min_periods=1).std()
    sdErr = df['std'] / np.sqrt(df.index + 1)
    sdErrors_TI.update({k: sdErr[1:]})

x = np.arange(3999)
plt.plot(x, sdErrors_TI['refTI_1'])
plt.plot(x, sdErrors_TI['refTI_2'])

plt.plot(x, sdErrors_TI['PP_1_full'])
plt.plot(x, sdErrors_TI['PP_2_full'])

plt.xlim([0,100])

##############################################################################
# Bayes factors convergence

bf_TI = np.exp(integrals_TI['refTIdirect'])
bf_refTI = np.exp(integrals_TI['refTI_2'] - integrals_TI['refTI_1'])
bf_PP11 = np.exp(integrals_TI['PP_2'] - integrals_TI['PP_1'])
bf_PP100 = np.exp(integrals_TI['PP_2_full'] - integrals_TI['PP_1_full'])

BF_dict = {'TI': bf_TI, 'refTI': bf_refTI, 'PP11': bf_PP11, 'PP100': bf_PP100}

for k in BF_dict.keys():
    x = np.arange(ITER * 4 / 2)
    plt.plot(x, BF_dict[k], label=k)
    plt.legend()

for k in BF_dict.keys():
    print(k, round(BF_dict[k][-1],2))
    
BF_sdErr = {}
BF_std = {}

for k in BF_dict.keys():
    df = pd.DataFrame(columns = ['BF', 'std', 'sdErr'])
    df['BF'] = BF_dict[k]
    df['std'] = df['BF'].rolling(len(df['BF'].values),min_periods=1).std()
    sdErr = df['std'] / np.sqrt(df.index + 1)
    BF_sdErr.update({k: sdErr[1:]})
    BF_std.update({k: df['std'].values[1:]})

for k in BF_sdErr.keys():
    x = np.arange(ITER * 4 / 2 -1)
    plt.plot(x, BF_sdErr[k], label=k)
    plt.legend()

for k in BF_dict.keys():
    bf = BF_dict[k][-1]
    sdErr = BF_sdErr[k]
    conv = np.where(sdErr < bf * 0.005)[0][0]
    print(k, conv)