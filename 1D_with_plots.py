# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:50:01 2020

@author: iwona
"""

#############################################################################
# This script demonstrates the method of calculating the model evidence 
# via Thermodynamic Integration (TI) with reference and compares it to the 
# Power Posterior method
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
    # tmp = 0.5*(x - 4)**2 + 0.25*(x - 4)**6 + 0.5*(x - 4)**10
    # tmp = (x-4)**2 + (x-4)**10
    # return 0.5*(x - 4)**2 + 0.25*(x - 4)**6
    # tmp =  np.sinh(x-4) + (x-4)**4
    tmp = 0.5 * np.sqrt(np.abs(x-4)) + 0.5 * (x-4)**4
    # return (x - 4)**2 * np.cos(16/(x - 4)**4) + (x - 4)**4
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
   //real tmp = (x-4)*(x-4) + pow(x-4,10);
   //real tmp =  0.5*pow((x - 4),2) + 0.25*pow(x - 4,6) + 0.5*pow(x - 4,10);
  real tmp = 0.5 * sqrt(fabs(x-4)) + 0.5 * pow(x-4, 4);
  //real tmp = sinh(x-4) + pow(x-4,4);
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
# fit.plot()
# print(fit)

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
plt.savefig('cusp_q_qref.pdf', format='pdf', bbox_inches = "tight")
# plt.savefig('div_narrow_q_ref.pdf', format='pdf', bbox_inches = "tight")

###########################################################################
# 4. Show the paths for power posterior and TI
# lam = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lam = [0, 0.2, 0.5, 0.8, 1]

def path(fun, fun_ref, lam, x):
    return fun(x)**(lam) * fun_ref(x)**(1-lam)

for l in lam:
    sns.lineplot(x, path(f, f_ref, l, x), label = '$\lambda$=' + str(round(l,1)))
plt.xlim([2,6])
plt.xlabel('$\\theta$')
plt.ylabel('Density')
plt.legend(loc='upper right')
# plt.title('Paths for the TI with reference method')
plt.savefig('cusp_paths.pdf', format='pdf', bbox_inches = "tight")
# plt.savefig('div_paths.pdf', format='pdf', bbox_inches = "tight")

# for l in lam:
#     sns.lineplot(x, path(f, lambda x: 1, l, x))
# plt.title('Paths for the PP method')
# plt.show()

###########################################################################
# 5. Do the TI and PP with lambdas [0,1]
TI_stan_code_1d = """
functions {
  real lf(real x)
  {
   //real tmp = (x-4)*(x-4) + pow(x-4,10);
  //real tmp =  0.5*pow((x - 4),2) + 0.25*pow(x - 4,6) + 0.5*pow(x - 4,10);
  real tmp = 0.5 * sqrt(fabs(x-4)) + 0.5 * pow(x-4, 4);
  //real tmp = sinh(x-4) + pow(x-4,4);
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

# PP_stan_code_1d = """
# functions {
#   real lf(real x)
#   {
#    real tmp =  0.5*pow((x - 4),2) + 0.25*pow(x - 4,6) + 0.5*pow(x - 4,10);
#   //real tmp = 0.5 * sqrt(abs(x-4)) + 0.5 * pow(x-4, 4);
#   return -1 * tmp;  
#    }
#   real path_lpdf(real x, real lambda)
#   {
#   return lambda * lf(x);
#   }
#     }
# data {
#       real lambda;
#       }
# parameters {
#     real theta;
# }
# model {
#        theta ~ path(lambda);
# }
# generated quantities {
#     real diff = lf(theta);
#     }
# """
# model_PP = pystan.StanModel(model_code=PP_stan_code_1d)

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
TI_dict_div = get_TI_dict()
TI_dict_div_narrow = get_TI_dict()

# PP_dict = get_TI_dict(TI=False)  

TI_dict_full = TI_dict.copy()

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
    # plt.ylim([-0.06,0])
    plt.xlabel('$\lambda$')
    plt.ylabel('Expectation')
    # plt.show()
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
    
def plot_per_iter(meas, rolling_dict, lambdaVals = 'all'):
    if meas == 'mean':
        df = rolling_dict['means']
        # ylab = r'$\mathbb{E}_{\theta \sim q(\lambda;\theta)}$'
        # ylab += r'$[\mathrm{log} \frac{q_{1}(\theta)}{q_{0}(\theta)}]$'
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
    plt.ylim([-0.1,0.00])
    plt.xlabel('Iteration')
    plt.ylabel(ylab)
    plt.legend(loc='lower right', ncol = 2, fontsize=10)
    # plt.show()

TIextra = get_logZextra(TI_dict, lambdaVals = [0, 0.2, 0.5, 0.8, 1])
# TIextra_div = get_logZextra(TI_dict_div, T=True)
# TIextra_div_narrow = get_logZextra(TI_dict_div_narrow, TI=True)
# plt.savefig('div_narrow_expect_lambda.pdf', format='pdf', bbox_inches = "tight")

# plt.savefig('div_expect_lambda.pdf', format='pdf', bbox_inches = "tight")
# plt.savefig('cusp_expect_lambda.pdf', format='pdf', bbox_inches = "tight")

# PPextra = get_logZextra(PP_dict, TI=False)
    
TI_rolling = get_all_vals_rolling(TI_dict, lambdaVals = [0, 0.2, 0.5, 0.8, 1])
TI_rolling_div = get_all_vals_rolling(TI_dict_div)
TI_rolling_div_narrow = get_all_vals_rolling(TI_dict_div_narrow)

ti_15_runs_rolling = [get_all_vals_rolling(ti, lambdaVals = [0, 0.2, 0.5, 0.8, 1]) for ti in ti_15_runs]
# PP_rolling = get_all_vals_rolling(PP_dict)

plot_per_iter('mean', TI_rolling, lambdaVals = [0.0, 0.2, 0.5, 0.8, 1.0])
# plot_per_iter('mean', TI_rolling_div)
# plot_per_iter('mean', TI_rolling_div_narrow)

# plt.savefig('div_expect_lambda_iter.pdf', format='pdf', bbox_inches = "tight")

plt.savefig('cusp_expect_lambda_iter.pdf', format='pdf', bbox_inches = "tight")
# plot_per_iter('mean', PP_rolling)
plot_per_iter('se', TI_rolling)
# plot_per_iter('se', PP_rolling)

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
# integrals_TI = get_integrals(TI_rolling_div_narrow, TI=True)
integrals_TI += LogLaplaceCovariance()
# integrals_PP = get_integrals(PP_rolling, TI=False)
ti_15_runs_integrals = [get_integrals(ti, lambdaVals = [0, 0.2, 0.5, 0.8, 1]) for ti in ti_15_runs_rolling]
ti_15_runs_integrals = [ti + LogLaplaceCovariance() for ti in ti_15_runs_integrals]

zExact = np.log(scipy.integrate.quad(lambda x: f(x), 0,10)[0])
zExact = np.ones(len(integrals_TI)) * zExact
print(zExact[0], LogLaplaceCovariance())


rolling_std = pd.Series(np.exp(integrals_TI)).rolling(len(integrals_TI),min_periods=1).std()

ar = np.arange(len(integrals_TI))
plt.plot(ar, np.exp(zExact), label='z', color='red', ls = '--')
plt.fill_between(ar, np.exp(zExact) - np.exp(zExact) * 0.01,
                  np.exp(zExact) + np.exp(zExact) * 0.01,
                  alpha = 0.2, color='green', linewidth=0.1, label=r'$z \pm 1\%$')
plt.fill_between(ar, np.exp(zExact) - np.exp(zExact) * 0.001,
                  np.exp(zExact) + np.exp(zExact) * 0.001,
                  alpha = 0.4, color='orange', linewidth=0.1, label=r'$z \pm 0.1\%$')
plt.plot(ar, np.exp(integrals_TI), label=r'$z_{ \mathrm{TI} }$')
# for i in range(15):
#     plt.plot(ar, np.exp(ti_15_runs_integrals[i]), color='blue', alpha=0.3)
logZref = np.ones(len(integrals_TI)) * LogLaplaceCovariance()
plt.plot(ar, np.exp(logZref), label=r'$z_{ \mathrm{ref} }$', color='purple',ls = '--')
# plt.fill_between(ar[1:], np.exp(integrals_TI[1:]) - rolling_std[1:],
#                   np.exp(integrals_TI[1:]) + rolling_std[1:],
#                   alpha = 0.3, color='b', linewidth=0.1, label='Ref TI $\pm$ sd')
plt.xlabel('Iteration')
plt.ylabel('Evidence')
plt.ylim([np.exp(zExact)[0]-0.023344311227239, np.exp(logZref)[0]+0.005])
# plt.xlim([0,20000])
# plt.legend(loc = 'lower right')
plt.legend(bbox_to_anchor=(0.70, 0.5))
# plt.savefig('div_narrow_evidence_iter.pdf', format='pdf', bbox_inches = "tight")

# plt.savefig('div_evidence_iter.pdf', format='pdf', bbox_inches = "tight")
plt.savefig('cusp_evidence_iter.pdf', format='pdf', bbox_inches = "tight")

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
        expectsPerLambda.append(lambda_dict[lam]['vals'].mean()+ zref) 
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    xnew = np.linspace(0, 1)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    lineLab = r'$\mathrm{log}(z_{\mathrm{ref}}) + \mathbb{E}_{\theta \sim q(\lambda;\theta)}$'
    lineLab += r'$[\mathrm{log} \frac{q_{1}(\theta)}{q_{0}(\theta)}]$'
    plt.plot(xnew,ynew, label = lineLab)
    plt.scatter(lambdaVals, expectsPerLambda)
    # expList = [expectsPerLambda[i] for i in [0,2,5,8,10]] 
    # plt.scatter([0.0, 0.2, 0.5, 0.8, 1.0], expList)
    plt.hlines(exact, -0.05, 1.05, ls = '--', color = 'red', label = r'$\mathrm{log}(z)$')
    plt.hlines(zref, -0.05, 1.05, ls = '--', color = 'purple', label = r'$\mathrm{log}(z_{\mathrm{ref}})$')
    plt.xlim([-0.05,1.05])
    plt.xlabel('$\lambda$')
    plt.ylabel('Log-evidence')
    plt.legend(loc = 'lower right', prop={'size': 13})


plot_logZextraAndExact(TI_dict, zExact)
plt.savefig('cusp_lambda.pdf', format='pdf', bbox_inches = "tight")


def plot_distribution_of_means(howMany = len(integrals_TI)):
    integrals = np.exp(integrals_TI)
    integrals = integrals[0:howMany]
    integrals = integrals[integrals > 1.52]
    integrals = integrals[integrals < 1.53]
    plt.hist(integrals, bins = 30)
    plt.xlabel('Integral rolling mean, iter = ' + str(howMany))
    plt.show()
    # sns.distplot(integrals, kde=False, norm_hist=True)
    

plot_distribution_of_means(1000)
plot_distribution_of_means(10000)

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

##############################################################################
# get a plot of TI-exact vs fraction of variance
fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
fraction2 = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]

sig = [np.sqrt(frac) * sigma for frac in fraction]
sig2 = [np.sqrt(frac) * sigma for frac in fraction2]

def get_logZextra(lambda_dict):
    lambdaVals = list(lambda_dict.keys())
    expectsPerLambda = []
    for lam in lambdaVals:
        expectsPerLambda.append(lambda_dict[lam]['vals'].mean())
    tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
    logzExtra = scipy.interpolate.splint(0, 1, tck, full_output=0)
    return logzExtra

def get_TI_dict_sigma(sig):
    TI_data = {'C': 1/(sig**2), 'peak': mu, 'f0': np.log(f(mu))}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, TI_data)
    return lambda_dict

sig_lambda_dict3 = {}
sig_results3 = {}
for s in sig+sig2:
    print(s)
    sig_lambda_dict3.update({s: get_TI_dict_sigma(sig=s)})
    sig_results3.update({s: get_logZextra(sig_lambda_dict3[s])})
    
sig_TI_exact_diff3 = {}

def f_ref_sig(x, sig):
    return f(mu) * np.exp(-1/2 * ((x-mu)/sig)**2)
for s in sig + sig2:
    TI_extra = sig_results3[s]
    TI_full = TI_extra + np.log(scipy.integrate.quad(lambda x: f_ref_sig(x, sig=s), 0,10)[0])
    TI_exact_diff = np.abs(TI_full - zExact[0])
    sig_TI_exact_diff3.update({s: TI_exact_diff})
    # TI_exact_diff = (TI_full/zExact[0]) * 100
    # sig_TI_exact_diff2.update({s: TI_exact_diff})
   
s, diff = zip(*(sig_TI_exact_diff3.items())) # unpack a list of pairs into two tuples
plt.scatter(fraction+fraction2, diff)
plt.xlabel('fraction of variance')
plt.ylabel('|TI-exact|')
plt.show()

def plot_expect_at_lambda_0(sigma_dict):
    for i in range(len(sig)):
        if (fraction[i] == 0.5 or fraction[i] == 0.6):
            s = list(sigma_dict.keys())[i]
            vals = sigma_dict[s][0.0]['vals']
            iteration = np.arange(len(vals))
            plt.plot(iteration, vals, label = fraction[i])
    plt.xlabel('iteration')
    plt.ylabel('expect for $\lambda$=0')
    plt.legend()
    plt.show()
        
plot_expect_at_lambda_0(sig_lambda_dict)
plot_expect_at_lambda_0(sig_lambda_dict2)


def f1(x):
    tmp = 0.5*(x - 4)**2 + 0.25*(x - 4)**6 + 0.5*(x - 4)**10
    return np.exp(-1 * tmp)

def f2(x):
    tmp = 0.5 * np.sqrt(np.abs(x-4)) + 0.5 * (x-4)**4
    return np.exp(-1 * tmp)

def f3(x):
  tmp =  np.sinh(x-4) + (x-4)**4
  return np.exp(-1 * tmp)
            
x = np.linspace(0,8,200)
plt.plot(x, f3(x), color = 'k', alpha=1)    
for i in range(len(fraction)):
    s = sig[i]
    plt.plot(x,f_ref_sig(x, s), alpha=0.5, color='b')
for i in range(len(fraction2)):
    s = sig2[i]
    plt.plot(x,f_ref_sig(x, s), alpha=0.5, color='g')   
plt.show()     
