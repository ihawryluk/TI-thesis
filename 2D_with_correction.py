# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:09:29 2020

@author: iwona
"""

import numpy as np
import pystan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sympy
sns.set()

SEED = 1234
ITER = 2000 * 10

def f(x):
    tmp =  (x[0]-0.5)**2 + (x[0]-0.5)**4
    tmp +=  (x[1]-0.5)**2 + (x[1]-0.5)**4
    tmp += 0.5*x[0]*x[1]*x[1]
    return np.exp(-0.25 * tmp)

max_arg = scipy.optimize.fmin(lambda x: (-1) * f(x), [1,1])
mode = max_arg

x, y = sympy.symbols('x y')
hexpr2d = 0.25 * ((x-0.5)**2 + (x-0.5)**4 + (y-0.5)**2 + (y-0.5)**4 + 0.5*x*y*y)
hessian = np.matrix(sympy.hessian(hexpr2d,[x,y]).evalf(subs={x:mode[0],y:mode[1]}), dtype='float')

def f_lim(x):
    z = f(x)
    z = np.asarray(z)
    z[x[0] < 0] = 0
    # z[x[1] < 0] = 0
    return z


f_stan = """
functions {
   real f_lpdf(vector x)
  {
  return -0.25 * (pow(x[1]-0.5,2) + pow(x[1]-0.5,4) + pow(x[2]-0.5,2) + pow(x[2]-0.5,4)+ 0.5*x[1]*x[2]*x[2]);
  }
    }
"""

custom_pdf_code = f_stan + """
data {}
transformed data {}
parameters {
    real x1;
    //real<lower=0> x1;
    real x2;

}
transformed parameters {
    vector[2] y;
    y[1] = x1;
    y[2] = x2;
}
model {
       y ~ f();
}
generated quantities {}
"""

m = pystan.StanModel(model_code=custom_pdf_code)
fit = m.sampling(iter=ITER*10, seed=SEED, control = {'adapt_delta': 0.8}, n_jobs=-1)
fit.plot()
print(fit)


df = fit.to_dataframe()
samples = df[['y[1]', 'y[2]']].values
cov = np.cov(np.array(samples), rowvar = False)
covinv = np.linalg.inv(cov)
mu = samples.mean(axis = 0)
    

def Logf(mu):
    return np.log(f(mu))
    
def LogLaplaceCovariance_noCorrection(mu, cov):
     result = 1/2 * len(mu) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(cov))
     result += Logf(mu) 
     return result

def LogLaplaceHessian_noCorrection(mode, hessian):
     result = 1/2 * len(mode) * np.log(2*np.pi) 
     result -= 1/2 * np.log(np.linalg.det(hessian))
     result += Logf(mode) 
     return result
 

def LogLaplaceCovariance_Correction(mu, cov):
     cov = np.diag(np.diag(cov))
     result = 1/2 * len(mu) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(cov))
     result += Logf(mu)
     # and now correction for the constraint
     correction = 0
     for i in range(2):
        sigma = np.sqrt(cov[i,i])
        mu_i = mu[i]
        correction += np.log((1 + scipy.special.erf(mu_i/(np.sqrt(2)*sigma))) / 2)
        # correction *= (1 + scipy.special.erf(mu_i/(np.sqrt(2)*sigma))) / 2
        print(result, correction)
     # return result + np.log(correction)
     return result + correction

def LogLaplacHessian_Correction(mode, hessian):
     hess = np.diag(np.diag(np.linalg.inv(hessian)))
     # hess = np.diag(np.diag(hessian))
     result = 1/2 * len(mode) * np.log(2*np.pi) 
     result += 1/2 * np.log(np.linalg.det(hess))
     result += Logf(mode)
     # and now correction for the constraint
     correction = 0
     for i in range(1): # second param is not constrained!!!
         sigma = np.sqrt(hess[i,i])
         mu_i = mode[i]
         correction += np.log((1 + scipy.special.erf(mu_i/(np.sqrt(2)*sigma))) / 2)
         # correction *= (1 + scipy.special.erf(mu_i/(np.sqrt(2)*sigma))) / 2
         print(result, correction)
     # return result + np.log(correction) 
     return result + correction
 

# zExact = scipy.integrate.dblquad(lambda x,y: f_lim([x,y]), -20,20,-20,20)[0]
# zLaplace_noCorr = LogLaplaceCovariance_noCorrection(mu, cov)
# zLaplace_Corr = LogLaplaceCovariance_Correction(mu, cov)

zExact = np.log(scipy.integrate.dblquad(lambda x,y: f([x,y]), 0,20,-20,20)[0])
ZLaplaceHessian_noCorr = LogLaplaceHessian_noCorrection(mode, 2*hessian)
ZLaplaceHessian_Corr = LogLaplacHessian_Correction(mode, 2*hessian)

    
hessianTI = """
functions {
  real lf(vector x)
  {
  return -0.25 * (pow(x[1]-0.5,2) + pow(x[1]-0.5,4) + pow(x[2]-0.5,2) + pow(x[2]-0.5,4)+ 0.5*x[1]*x[2]*x[2]);
  }
  real lfref(vector x, real f0, vector peak, matrix C)
  {
   return f0 - 0.5 * (x-peak)' * C * (x-peak);
   }
  real path_lpdf(vector x, real f0, vector peak, matrix C, real lambda)
  {
  return lambda * lf(x) + (1-lambda) * lfref(x, f0, peak, C);
  }
    }
data {
      matrix[2,2] C;
      vector[2] peak;
      real f0;
      real lambda;
      }
transformed data {}
parameters {
    real<lower=0> x1;
    //real x1;
    real x2;
}
transformed parameters {
    vector[2] theta;
    theta[1] = x1;
    theta[2] = x2;
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

model = pystan.StanModel(model_code=hessianTI)


def get_expect_for_lambda(lam, data, n_iter=ITER):
    fit = model.sampling(data=data, iter=n_iter, n_jobs=-1, 
                                control = {'adapt_delta': 0.9}, seed=SEED)
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

def get_TI_dict(diag = False):
    # data = {'C': covinv, 'peak': mu, 'f0': Logf(mu)}
    if diag:
        data = {'C': np.diag(np.diag(2*hessian)), 'peak': mode, 'f0': Logf(mode)}
    else:
        data = {'C': 2*hessian, 'peak': mode, 'f0': Logf(mode)}
    lambdaVals = np.arange(0,1.1,0.1)
    lambda_dict = MCMC_for_all_lambdas(lambdaVals, data)
    return lambda_dict

TI_dict_full = get_TI_dict(diag=False)
TI_dict_diag = get_TI_dict(diag=True)

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

TIadd_full_cov = get_logZextra(TI_dict_full)
TIadd_diag_cov = get_logZextra(TI_dict_diag)


print('Log-evidence results')
print('Exact', zExact)
print('Laplace no correction', ZLaplaceHessian_noCorr)
print('Laplace with correction', ZLaplaceHessian_Corr)
print('TI no correction', ZLaplaceHessian_noCorr + TIadd_full_cov)
print('TI with correction', ZLaplaceHessian_Corr + TIadd_diag_cov)

print('Evidence results')
print('Exact', np.exp(zExact))
print('Laplace no correction', np.exp(ZLaplaceHessian_noCorr))
print('Laplace with correction', np.exp(ZLaplaceHessian_Corr))
print('TI no correction', np.exp(ZLaplaceHessian_noCorr + TIadd_full_cov))
print('TI with correction', np.exp(ZLaplaceHessian_Corr + TIadd_diag_cov))


##################################################################
#### plots #######################################################

def f(x):
    tmp =  (x[0]-0.5)**2 + (x[0]-0.5)**4
    tmp +=  (x[1]-0.5)**2 + (x[1]-0.5)**4
    tmp += 0.5*x[0]*x[1]*x[1]
    return np.exp(-0.25 * tmp)

max_arg = scipy.optimize.fmin(lambda x: (-1) * f(x), [1,1])
mode = max_arg

x, y = sympy.symbols('x y')
hexpr2d = 0.25 * ((x-0.5)**2 + (x-0.5)**4 + (y-0.5)**2 + (y-0.5)**4 + 0.5*x*y*y)
hessian = np.matrix(sympy.hessian(hexpr2d,[x,y]).evalf(subs={x:mode[0],y:mode[1]}), dtype='float')

def f1(x):
    tmp =  (x+0.5)**2 + (x+0.5)**4
    return np.exp(-0.5 * tmp)

x = np.linspace(-3,5,100)

plt.plot(x,f[x,0])

plt.plot(x, f1(x))

def f_lim(x):
    z = f(x)
    z = np.asarray(z)
    # z[x[0] < 0] = 0
    # z[x[1] < 0] = 0
    return z

def f_ref_full(x):
    # peak = mu
    # C = np.linalg.inv(cov)
    peak = mode
    C = 2*hessian
    a1 = (x[0] - peak[0])**2 * C[0,0]
    a2 = (x[1] - peak[1])**2 * C[1,1]
    a3 = 2 * (x[0] - peak[0]) * (x[1] - peak[1]) * C[1,0]
    tmp = a1 + a2 + a3
    # return f(mode) * np.exp(-0.5 * tmp)
    return np.exp(np.log(f(mode)) - 0.5 * tmp)


def f_ref_diag(x):
    peak = mode
    C = 2*hessian
    a1 = (x[0] - peak[0])**2 * C[0,0]
    a2 = (x[1] - peak[1])**2 * C[1,1]
    #a3 = 2 * (x[0] - peak[0]) * (x[1] - peak[1]) * C[1,0]
    tmp = a1 + a2 #+ a3
    # return f(mode) * np.exp(-0.5 * tmp)
    return np.exp(np.log(f(mode)) - 0.5 * tmp)

def make0(x):
    return 0 * x

lower = -2
upper = 3
# Plot the functions - contour plots
xlist = np.linspace(lower-0.5,upper+0.5, 1000)
ylist = np.linspace(lower,upper, 1000)
X, Y = np.meshgrid(xlist, ylist)
fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,5))

x0list = np.linspace(lower-0.5,0, 1000)
y0list = np.linspace(lower,upper, 1000)
X0, Y0 = np.meshgrid(x0list, y0list)

Z = f_lim([X,Y])
cp1 = ax1.contourf(X, Y, Z)
Z0 = make0(X)
ax1.contourf(X0, Y0, Z0, alpha=0.5)
fig.colorbar(cp1, ax=ax1) # Add a colorbar to a plot
ax1.set_title('$q(\Theta)$')
ax1.set_xlabel('$\Theta_1$')
ax1.set_ylabel('$\Theta_2$')
ax1.plot([0, 0], [lower,upper], '-r')

Z = f_ref_full([X,Y])
cp2 = ax2.contourf(X, Y, Z)
fig.colorbar(cp2, ax=ax2)
ax2.contourf(X0, Y0, Z0, alpha=0.5)
ax2.set_title('$q_{ref}(\Theta)$, full covariance')
ax2.set_xlabel('$\Theta_1$')
ax2.set_ylabel('$\Theta_2$')
ax2.plot([0, 0], [lower,upper], '-r')

Z = f_ref_diag([X,Y])
cp3 = ax3.contourf(X, Y, Z)
fig.colorbar(cp3, ax=ax3)
ax3.contourf(X0, Y0, Z0, alpha=0.5)
ax3.set_title('$q_{ref}(\Theta)$, diagonal covariance')
ax3.set_xlabel('$\Theta_1$')
ax3.set_ylabel('$\Theta_2$')
ax3.plot([0, 0], [lower,upper], '-r')

plt.tight_layout()

plt.savefig('C:/Users/iwona/Desktop/TI_plots/hessianConstrained.pdf' ,format = 'pdf', bbox_inches = "tight")



