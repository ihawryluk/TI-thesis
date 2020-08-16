# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:48:33 2020

@author: iwona
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

def fit_posterior(data_x, data_y, s):
    data = {'N': data_x.shape[0], 'y': data_y, 'X': data_x,
            'C': np.array([[1/0.06,0],[0,1/6.0]]),'mu0': [3000,185]}
    fit = model.sampling(data=data, iter=ITER, n_jobs=-1, seed=SEED + s)
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

posterior1 = fit_posterior(model1, y, 0)
posterior2 = fit_posterior(model2, y, 0)


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

Lap_1 = []
Lap_2 = []

for i in range(15):
    posterior1 = fit_posterior(model1, y, i)
    posterior2 = fit_posterior(model2, y, i)
    Lap_1.append(LogLaplaceCovariance(posterior1))
    Lap_2.append(LogLaplaceCovariance(posterior2))

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


def get_expect_for_lambda(lam, data, n_iter, s):
    n_iter = n_iter
    data.update({'lambda': lam})
    fit = m.sampling(data=data, iter=n_iter, n_jobs=-1, seed = SEED + s)
    vals = fit.extract()['diff']
    expects = vals.mean()
    return {'expects': expects, 'vals': vals}

def MCMC_for_all_lambdas(lambdaVals, data, n_iter, s):
    """Execute TI MCMC for multiple lambdas.
    lambdaVals: list of values of lambdas"""
    t0 = time.time()
    lambdaOutput = {}
    for l in lambdaVals:
        lam = round(l,5)
        lambdaOutput.update({lam: get_expect_for_lambda(lam, data, n_iter, s)})
    print(time.time() - t0, 'seconds elapsed')
    return lambdaOutput


def get_logZextra(lambda_dict):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['expects'])
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    # calculate exponent of the integral of the interpolated line
    logzExtra = scipy.interpolate.splint(lambdaVals[0], lambdaVals[-1], tck, full_output=0)
    return logzExtra


def TI_dict(posterior, n_iters, s):
    data = {'K': len(posterior['mu']),'C': posterior['covinv'], 
            'peak': posterior['mu'], 'f0': Logf(posterior), 
            'N': posterior['x'].shape[0], 'Y': posterior['y'], 
            'X': posterior['x'], 'cov_prior': np.array([[1/0.06,0],[0,1/6.0]]), 'mu0': [3000,185]}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, data, n_iters, s)
    return lambda_dict

### Run TI and print the results
logzRefCovariance1 = LogLaplaceCovariance(posterior1)
logzRefCovariance2 = LogLaplaceCovariance(posterior2)

TI_ref1 = []
TI_ref2 = []
for i in range(15):
    print(i)
    TI_dict_1 = TI_dict(posterior1, ITER, i)
    TI_dict_2 = TI_dict(posterior2, ITER, i)
    logzExtra_covariance1 = get_logZextra(TI_dict_1)
    logzExtra_covariance2 = get_logZextra(TI_dict_2)
    TI_ref1.append(logzExtra_covariance1)
    TI_ref2.append(logzExtra_covariance2)


TI_ref1_Lap = TI_ref1 + logzRefCovariance1
TI_ref2_Lap = TI_ref2 + logzRefCovariance2


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


def get_expect_for_lambda_PP(lam, data, n_iter, s):
    n_iter = n_iter
    data.update({'lambda': lam})
    fit = mpp.sampling(data=data, iter=n_iter, n_jobs=-1, seed = SEED + s)
    vals = fit.extract()['logf']
    expects = vals.mean()
    # print(lam, ', expectation = ', expects)
    return {'expects': expects, 'vals': vals}

def MCMC_for_all_lambdas_PP(lambdaVals, data, n_iter, s):
    """Execute TI MCMC for multiple lambdas.
    lambdaVals: list of values of lambdas"""
    t0 = time.time()
    lambdaOutput = {}
    for l in lambdaVals:
        lam = round(l,10)
        lambdaOutput.update({lam: get_expect_for_lambda_PP(lam, data, n_iter, s)})
    print(time.time() - t0, 'seconds elapsed')
    return lambdaOutput

def PP_dict(posterior, n_iters, s):
    data = {'K': len(posterior['mu']),
            'N': posterior['x'].shape[0], 'Y': posterior['y'], 
            'X': posterior['x'], 'cov_prior': np.array([[1/0.06,0],[0,1/6.0]]),
            'mu0': [3000,185]}
    # lambdaVals = np.arange(0.0,1.1,0.1)
    lambdaVals = np.arange(0.0,1.01,0.01)**5  # Friel&Wyse(2011)
    lambda_dict = MCMC_for_all_lambdas_PP(lambdaVals, data, n_iters, s)
    return lambda_dict


PP_1 = []
PP_2 = []
for i in range(15):
    print(i)
    PP_dict_1 = PP_dict(posterior1, ITER, i)
    PP_dict_2 = PP_dict(posterior2, ITER, i)
    logpp_1 = get_logZextra(PP_dict_1)
    logpp_2 = get_logZextra(PP_dict_2)
    PP_1.append(logpp_1)
    PP_2.append(logpp_2)


###############################################################################
# all_evidence_1 = pd.DataFrame(columns = ['Laplace', 'Referenced TI', 'Power posterior'])
# all_evidence_2 = pd.DataFrame(columns = ['Laplace', 'Referenced TI', 'Power posterior'])
all_evidence_1 = pd.read_csv('C:/Users/iwona/Desktop/TI/data_examples/radiata15runsModel1.csv')
all_evidence_2 = pd.read_csv('C:/Users/iwona/Desktop/TI/data_examples/radiata15runsModel2.csv')

# all_evidence_1['Laplace'] = Lap_1
# all_evidence_2['Laplace'] = Lap_2
# all_evidence_1['Referenced TI'] = TI_ref1_Lap
# all_evidence_2['Referenced TI'] = TI_ref2_Lap
# all_evidence_1['PP 100 temperatures'] = PP_1
# all_evidence_2['PP 100 temperatures'] = PP_2

# all_evidence_1.to_csv('C:/Users/iwona/Desktop/TI/data_examples/radiata15runsModel1.csv', index = False)
# all_evidence_2.to_csv('C:/Users/iwona/Desktop/TI/data_examples/radiata15runsModel2.csv', index = False)

all_evidence_1_no_PP = all_evidence_1.drop(columns = ['Power posterior'])
all_evidence_2_no_PP = all_evidence_2.drop(columns = ['Power posterior'])

names = {'Referenced TI': 'Ref TI', 'PP 100 temperatures': 'PP$_{100}$'}
all_evidence_1_no_PP.rename(columns = names, inplace=True)
all_evidence_2_no_PP.rename(columns = names, inplace=True)

graph = sns.boxplot(x="variable", y="value", data=pd.melt(all_evidence_1_no_PP))
graph.axhline(-310.1280, linestyle=':')
plt.ylabel('Log-evidence')
plt.xlabel('Approximation method')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataBoxModel1.pdf', format='pdf', bbox_inches = "tight")

graph = sns.boxplot(x="variable", y="value", data=pd.melt(all_evidence_2_no_PP))
graph.axhline(-301.7046, linestyle=':')
plt.ylabel('Log-evidence')
plt.xlabel('Approximation method')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataBoxModel2.pdf', format='pdf', bbox_inches = "tight")

all_evidence_PP = pd.DataFrame(columns = ['$M_1$', '$M_2$'])
all_evidence_PP['$M_1$'] = all_evidence_1['Power posterior']
all_evidence_PP['$M_2$'] = all_evidence_2['Power posterior']
graph = sns.boxplot(x="variable", y="value", data=pd.melt(all_evidence_PP))
graph.axhline(-310.1280, linestyle=':', color = 'tab:blue', label = '$M_1$ exact')
graph.axhline(-301.7046, linestyle=':', color = 'tab:orange', label = '$M_2$ exact')
plt.legend()
plt.ylabel('Log-evidence')
plt.xlabel('')
plt.savefig('C:/Users/iwona/Desktop/TI_plots/' +  'RadiataBoxPPoff.pdf', format='pdf', bbox_inches = "tight")
