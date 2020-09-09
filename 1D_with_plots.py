# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:50:01 2020

@author: iwona
"""

#############################################################################
# This script demonstrates the method of calculating the model evidence 
# via referenced Thermodynamic Integration (TI) for 1d function
############################################################################

import numpy as np
import pystan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
sns.set()

SEED = 1234
ITER = 2000

###########################################################################
# 1. Define a 1D posterior distribution of interest

def f(x):
    tmp = 0.5 * np.sqrt(np.abs(x-4)) + 0.5 * (x-4)**4
    return np.exp(-1 * tmp)
            
x = np.linspace(0,8,100)
sns.lineplot(x, f(x))
plt.show()

###########################################################################
# 2. Sample from the posterior
f_stan = """
functions {
   real f_lpdf(real x)
  {
  real tmp = 0.5 * sqrt(fabs(x-4)) + 0.5 * pow(x-4, 4);
  return -1 * tmp;
  }
    }
"""

custom_pdf_code = f_stan + """
data {}
parameters {
    real y;
}
transformed parameters {}
model {
       y ~ f();
}
generated quantities {}
"""

m = pystan.StanModel(model_code=custom_pdf_code)
fit = m.sampling(iter=ITER*10, seed=SEED, control = {'adapt_delta': 0.8}, n_jobs=-1)
print(fit)

# from the sampler we need the mean and variance
sampled_x = fit.extract('y')['y']
mu = sampled_x.mean()
sigma = sampled_x.std() #/ np.sqrt(2)

###########################################################################
# 3. Build a reference function using Laplace approximation
def f_ref(x):
    return f(mu) * np.exp(-1/2 * ((x-mu)/sigma)**2)
x = np.linspace(0,8,500)
sns.lineplot(x, f(x), label='$q(\\theta)$')
sns.lineplot(x, f_ref(x), label='$q_{\mathrm{ref}}(\\theta)$')
plt.xlabel('$\\theta$')
plt.ylabel('Density')
plt.legend()
plt.show()

###########################################################################
# 4. Show the paths for power posterior and TI
lam = [0, 0.2, 0.5, 0.8, 1]

def path(fun, fun_ref, lam, x):
    return fun(x)**(lam) * fun_ref(x)**(1-lam)

for l in lam:
    sns.lineplot(x, path(f, f_ref, l, x), label = '$\lambda$=' + str(round(l,1)))
plt.xlim([2,6])
plt.xlabel('$\\theta$')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.show()

###########################################################################
# 5. Do the referenced TI
TI_stan_code_1d = """
functions {
  real lf(real x)
  {
  real tmp = 0.5 * sqrt(fabs(x-4)) + 0.5 * pow(x-4, 4);
  return -1 * tmp;  
   }
  real lfref(real x, real f0, real peak, real C)
  {
   return f0 - 0.5 * (x-peak) * C * (x-peak);
   }
  real path_lpdf(real x, real f0, real peak, real C, real lambda)
  {
  return lambda * lf(x) + (1-lambda) * lfref(x, f0, peak, C);
  }
    }
data {
      real C; // variance^-1
      real peak; // mu
      real f0; // true posterior(mu)
      real lambda;
      }
parameters {
    real theta;
}
model {
       theta ~ path(f0, peak, C, lambda);
}
generated quantities {
    real logf = lf(theta);
    real logfref = lfref(theta, f0, peak, C);
    real diff = logf - logfref;
    }
"""
model_TI = pystan.StanModel(model_code=TI_stan_code_1d)

def get_expect_for_lambda(lam, data, n_iter=ITER * 10, s=0):
    fit = model_TI.sampling(data=data, iter=n_iter, n_jobs=-1, 
                                control = {'adapt_delta': 0.8}, seed=SEED+s)
    vals = fit.extract()['diff']
    expects = vals.mean()
    print(lam, ', expectation = ', expects)
    return {'expects': expects, 'vals': vals}

def MCMC_for_all_lambdas(lambdaVals, data, s=0):
    """Execute TI MCMC for multiple lambdas.
    lambdaVals: list of values of lambdas"""
    lambdaOutput = {}
    for l in lambdaVals:
        lam = round(l,1)
        data.update({'lambda': lam})
        lambdaOutput.update({lam: get_expect_for_lambda(lam, data, s=s)})
    return lambdaOutput

def get_TI_dict(s=0):
    TI_data = {'C': 1/(sigma**2), 'peak': mu, 'f0': np.log(f(mu))}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, TI_data, s=s)
    return lambda_dict

TI_dict = get_TI_dict()

ti_15_runs = []
for i in range(15):
    ti_15_runs.append(get_TI_dict(s=i+1))


###########################################################################
# 1. Define a 1D posterior distribution of interest
def get_logZextra(lambda_dict, lambdaVals = 'all'):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    if lambdaVals == 'all':
        lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['vals'].mean())
        
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    plt.plot(xnew,ynew)
    plt.scatter(lambdaVals, expectsPerLambda)
    plt.xlim([-0.05,1.05])
    plt.xlabel('$\lambda$')
    plt.ylabel('Expectation')
    # calculate exponent of the integral of the interpolated line
    logzExtra = scipy.interpolate.splint(0, 1, tck, full_output=0)
    return logzExtra

def add_rolling_stats(values):
    df = pd.DataFrame(columns = ['values', 'roll_mean', 'roll_std'])
    df['values'] = values
    df['roll_mean'] = df['values'].rolling(len(df),min_periods=1).mean()    
    df['roll_std'] = df['values'].rolling(len(df),min_periods=1).std()
    df['roll_sdErr'] = df['roll_std'] / np.sqrt(df.index)
    return df

def get_all_vals_rolling(lambda_dict, lambdaVals = 'all'):
    if lambdaVals == 'all':
        lambdaVals = list(lambda_dict.keys())
    df_rolling_means = pd.DataFrame(columns = lambdaVals)
    df_rolling_se = pd.DataFrame(columns = lambdaVals)
    df_rolling_sd = pd.DataFrame(columns = lambdaVals)
    for lam in lambdaVals:
        df = add_rolling_stats(lambda_dict[lam]['vals'])
        df_rolling_means.loc[:,lam] = df['roll_mean'].values
        df_rolling_se.loc[:,lam] = df['roll_sdErr'].values
        df_rolling_sd.loc[:,lam] = df['roll_std'].values
    df_rolling_means = df_rolling_means.dropna()
    df_rolling_se = df_rolling_se.dropna()
    df_rolling_sd = df_rolling_sd.dropna()
    return {'means': df_rolling_means, 'se': df_rolling_se,
            'sd': df_rolling_sd}
    
def plot_per_iter(meas, rolling_dict, lambdaVals = 'all', xlim = False):
    if meas == 'mean':
        df = rolling_dict['means']
        ylab = 'Expectation per $\lambda$'
    else:
        df = rolling_dict['se']
        ylab = 'standard error'
    if lambdaVals == 'all':
        lambdaVals = list(df.columns)
    if lambdaVals[-1] == 'integral':
        lambdaVals = lambdaVals[:-1]
    for l in lambdaVals:
        plt.plot(df.index, df[l].values, label='$\lambda$=' + str(round(l,1)))
    if meas == 'se':
        plt.ylim([0, 0.5])
    plt.ylim([-0.085,0.00])
    plt.xlabel('Iteration')
    plt.ylabel(ylab)
    if xlim:
        plt.xlim(xlim)
    plt.legend(loc='lower right', ncol = 3)
    plt.show()


TIextra = get_logZextra(TI_dict, lambdaVals = [0, 0.2, 0.5, 0.8, 1])
TI_rolling = get_all_vals_rolling(TI_dict, lambdaVals = [0, 0.2, 0.5, 0.8, 1])

ti_15_runs_rolling = [get_all_vals_rolling(ti, lambdaVals = [0, 0.2, 0.5, 0.8, 1]) for ti in ti_15_runs]

plot_per_iter('mean', TI_rolling, lambdaVals = [0.0, 0.2, 0.5, 0.8, 1.0])
plot_per_iter('se', TI_rolling)

def integral_value(lambdaVals, values):
    assert len(values) == len(lambdaVals)
    if lambdaVals[0] != 0.0:
        lambdaVals = [0.0] + lambdaVals
        values = [1.0] + list(values)
    tck = scipy.interpolate.splrep(lambdaVals, values, s=0)
    logzExtra_mean = scipy.interpolate.splint(lambdaVals[0], lambdaVals[-1], tck, full_output=0)
    return logzExtra_mean


def get_integrals(rolling_dict, lambdaVals = 'all'):
    df = rolling_dict['means']
    if lambdaVals == 'all':
        lambdaVals = list(df.columns)
    if lambdaVals[-1] == 'integral':
        lambdaVals = lambdaVals[:-1]
    df['integral'] = df[lambdaVals].apply(lambda x: integral_value(lambdaVals, x), axis=1)
    return df['integral'].values

def LogLaplaceCovariance():
    return np.log(scipy.integrate.quad(lambda x: f_ref(x), 0,10)[0])

integrals_TI = get_integrals(TI_rolling, lambdaVals = [0, 0.2, 0.5, 0.8, 1])
integrals_TI += LogLaplaceCovariance()
ti_15_runs_integrals = [get_integrals(ti, lambdaVals = [0, 0.2, 0.5, 0.8, 1]) for ti in ti_15_runs_rolling]
ti_15_runs_integrals = [ti + LogLaplaceCovariance() for ti in ti_15_runs_integrals]

zExact = np.log(scipy.integrate.quad(lambda x: f(x), 0,10)[0])
zExact = np.ones(len(integrals_TI)) * zExact
print(zExact[0], LogLaplaceCovariance())


rolling_std = pd.Series(np.exp(integrals_TI)).rolling(len(integrals_TI),min_periods=1).std()

ar = np.arange(len(integrals_TI))
logZref = np.ones(len(integrals_TI)) * LogLaplaceCovariance()
plt.plot(ar, np.exp(logZref), label=r'$z_{ \mathrm{ref} }$', color='purple',ls = '--')
plt.plot(ar, np.exp(zExact), label='z', color='red', ls = '--')
plt.fill_between(ar, np.exp(zExact) - np.exp(zExact) * 0.01,
                  np.exp(zExact) + np.exp(zExact) * 0.01,
                  alpha = 0.2, color='green', linewidth=0.1, label=r'$z \pm 1\%$')
plt.fill_between(ar, np.exp(zExact) - np.exp(zExact) * 0.001,
                  np.exp(zExact) + np.exp(zExact) * 0.001,
                  alpha = 0.4, color='orange', linewidth=0.1, label=r'$z \pm 0.1\%$')
plt.plot(ar, np.exp(integrals_TI), label=r'$z_{ \mathrm{TI} }$')
plt.xlabel('Iteration')
plt.ylabel('Evidence')
plt.ylim([np.exp(zExact)[0]-0.023344311227239, np.exp(logZref)[0]+0.005])
plt.xlim([0,20000])
plt.legend(bbox_to_anchor=(0.70, 0.45))
plt.show()


print(zExact[0], integrals_TI[-1])
se = integrals_TI.std() # / np.sqrt(len(integrals_TI))
print(integrals_TI[-1] - se,  integrals_TI[-1] + se)

all_final_TI = [np.exp(integrals_TI[-1])]
for i in range(15):
    all_final_TI.append(np.exp(ti_15_runs_integrals[i][-1]))

print('zExact', np.exp(zExact[0]))
print('zRefTI mean', np.asarray(all_final_TI).mean())
print('zExact +- 0.01%', (np.exp(zExact[0]) - np.exp(zExact[0]) * 0.001,
                  np.exp(zExact[0]) + np.exp(zExact[0]) * 0.001))
print('zRefTI CrI', np.quantile(np.asarray(all_final_TI),[0.025, 0.975]))


def plot_logZextraAndExact(lambda_dict, exact):
    zref = LogLaplaceCovariance()
    lambdaVals = list(lambda_dict.keys())
    lambdaVals = [0.0, 0.2, 0.5, 0.8, 1.0]
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['vals'].mean()) 

    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    lineLab = 'Expectation'
    plt.plot(xnew,ynew, label = lineLab)
    plt.scatter(lambdaVals, expectsPerLambda)
    plt.xlim([-0.05,1.05])
    plt.xlabel('$\lambda$')
    plt.ylabel(lineLab)
    plt.gca()
    plt.twinx()
    plt.grid(None)
    plt.hlines(zref, -0.05, 1.05, ls = '--', color = 'purple', label = r'$\mathrm{log}(z_{\mathrm{ref}})$')
    plt.hlines(exact, -0.05, 1.05, ls = '--', color = 'red', label = r'$\mathrm{log}(z)$')
    plt.legend(loc = 'center right')
    plt.ylabel('Log-evidence')
    plt.show()

plot_logZextraAndExact(TI_dict, zExact)

def plot_expectationPerLambda(lambda_dict):
    lambdaVals = list(lambda_dict.keys())
    lambdaVals = [0.0, 0.2, 0.5, 0.8, 1.0]
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['vals'].mean()) 

    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    lineLab = r'$\mathrm{log}(z_{\mathrm{ref}}) + \mathbb{E}_{\theta \sim q(\lambda;\theta)}$'
    lineLab += r'$[\mathrm{log} \frac{q_{1}(\theta)}{q_{0}(\theta)}]$'
    plt.plot(xnew,ynew, label = lineLab)
    plt.scatter(lambdaVals, expectsPerLambda)

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.08,0.0])
    plt.xlabel('$\lambda$')
    plt.ylabel('Expectation')
    plt.show()

plot_expectationPerLambda(TI_dict)



# when do we get convergence?
def whenConverge(integrals, conv, log=True):
    if log == False:
        threshold = np.exp(zExact[0]) * conv
        integrals = np.exp(integrals)
        diff = [abs(i - np.exp(zExact[0])) for i in integrals]
    else:
        threshold = zExact[0] * conv
        diff = [abs(i - zExact[0]) for i in integrals]
    diffBool = list(diff < threshold)
    try:
        index_pos = len(diffBool) - diffBool[::-1].index(False) - 1
    except:
        return -1
    return index_pos + 1

print('Convergence within 1%, log-evidence: ', whenConverge(integrals_TI, 0.01))
print('Convergence within 0.1%,  log-evidence: ', whenConverge(integrals_TI, 0.001))

print('Convergence within 1%, evidence: ', whenConverge(integrals_TI, 0.01, False))
print('Convergence within 0.1%, evidence: ', whenConverge(integrals_TI, 0.001, False))

