# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:53:02 2020

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

GI_fixed =  [5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]
GI_fixed_path = 'fixedGI/'
GI_est_path = ''

lambdaVals = np.arange(0,1.1,0.1)
lambdaVals = [round(l,1) for l in lambdaVals]

def create_TI_dict(GI):
    TI_dict = {}
    if not isinstance(GI, str):
        for lam in lambdaVals:
            vals = pd.read_csv(GI_fixed_path + str(GI) + '_' + str(lam) + '.csv')['diff'].values
            TI_dict.update({lam: {'vals': vals}})
    else:
        for lam in lambdaVals:
            vals =  pd.read_csv(GI_est_path + GI + str(lam) + '.csv')['diff'].values
            TI_dict.update({lam: {'vals': vals}})

    return TI_dict


# all_TI = {5.0 : create_TI_dict(5), 6.0 : create_TI_dict(6), 6.5 : create_TI_dict(6.5), 
#           7.0 : create_TI_dict(7), 8.0 : create_TI_dict(8)}
all_TI = {5.0 : create_TI_dict(5), 6.0 : create_TI_dict(6), 6.5 : create_TI_dict(6.5), 
          7.0 : create_TI_dict(7), 8.0 : create_TI_dict(8), 9.0 : create_TI_dict(9),
          10.0 : create_TI_dict(10), 20.0 : create_TI_dict(20),
          40.0 : create_TI_dict(40)}

all_TI.update({'AR2': create_TI_dict('GI_est_')})    
all_TI.update({'AR3': create_TI_dict('GI_est_AR3_')})
all_TI.update({'AR4': create_TI_dict('GI_est_AR4_')})        
# all_TI.update({'AR3_1.5_cov': create_TI_dict('GI_est_AR3_15_cov_')})    
# all_TI.update({'AR3_1.25_cov': create_TI_dict('GI_est_AR3_125_cov_')})
all_TI.update({'W_1': create_TI_dict('GI_est_W_1_')})    
all_TI.update({'W_2': create_TI_dict('GI_est_W_2_')})   
all_TI.update({'W_3': create_TI_dict('GI_est_W_3_')})     
all_TI.update({'W_4': create_TI_dict('GI_est_W_4_')})    
all_TI.update({'W_7': create_TI_dict('GI_est_W_7_')})
# all_TI.update({'W_7_diag': create_TI_dict('Diag_cov_GI_est_W_7_')})    
    
SKdata = pd.read_csv('C:/Users/iwona/Desktop/TI/bellman_harris/SouthKoreaData.csv')
cases = SKdata.cases.values


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
for k in all_TI.keys():
    all_rolling.update({k: get_all_vals_rolling(all_TI[k])})


def plot_per_iter_lambdaZero(meas, all_rolling_dict):
    for k in all_rolling_dict.keys():
        rolling_dict = all_rolling_dict[k]
        if meas == 'mean':
            df = rolling_dict['means']
            ylab = 'rolling mean'
        elif meas == 'std':
            df = rolling_dict['std']
            ylab = 'standard deviation'
        else:
            df = rolling_dict['se']
            ylab = 'standard error'

        plt.plot(df.index, df[0.0].values, label='GI = ' + str(k))
    plt.xlabel('iteration')
    plt.ylabel(ylab)
    plt.title('Expectation per iteration for $\lambda=0.0$')
    plt.legend()
    plt.show()
    
plot_per_iter_lambdaZero('mean', all_rolling)

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

laplace = pd.read_csv('Laplace.csv')
integrals_TI = {}
for k in all_TI.keys():
    print(k)
    integrals_TI_tmp = get_integrals(all_rolling[k])
    integrals_TI_tmp += laplace['LogLaplaceCovariance'][laplace['GI'] == str(k)].values[0]
    integrals_TI.update({k: integrals_TI_tmp})
    

for k in all_TI.keys():
    ar = np.arange(len(integrals_TI[k]))
    plt.plot(ar, integrals_TI[k], label='GI = ' + str(k))
plt.legend()
plt.xlabel('iteration')
plt.ylabel('evidence value')
plt.show()


def get_logZextra(all_TI_dict):
    """Calculates zExtra as a an exponent of the integral of expectations over all lamdbas
    And plots it too
    """
    for k in all_TI_dict:
        lambda_dict = all_TI_dict[k]
        lambdaVals = list(lambda_dict.keys())
        expectsPerLambda = []
        for lam in lambdaVals:
            expectsPerLambda.append(lambda_dict[lam]['vals'].mean())
        tck = scipy.interpolate.splrep(lambdaVals, expectsPerLambda, s=0)
        xnew = np.linspace(0, 1)
        ynew = scipy.interpolate.splev(xnew, tck, der=0)
        plt.plot(xnew,ynew, label = str(k))
        plt.scatter(lambdaVals, expectsPerLambda)
    plt.xlim([0,1])
    plt.xlabel('lambda')
    plt.ylabel('expectation')
    plt.legend()
    plt.show()

get_logZextra(all_TI)

results = pd.DataFrame(columns = ['GI', 'LogLaplace', 'LogTIadd', 'LogEvidence', 'sd', 'CI95'])
for k in all_TI.keys():
    d = {'GI': k,
         'LogLaplace': laplace['LogLaplaceCovariance'][laplace['GI'] == str(k)].values[0],
         # 'LogTIadd': all_rolling[k]['means'].values[-1][-1],
         # 'LogTIadd': get_integrals(all_rolling[k])[-1],
         'LogEvidence': integrals_TI[k][-1],
         'sd': integrals_TI[k].std(),
         'CI95': np.quantile(integrals_TI[k], [0.025, 0.975], axis = 0)}
    d.update({'LogTIadd': d['LogEvidence'] - d['LogLaplace']})
    results = results.append(d, ignore_index=True)

def BF(log1, log2):
    return (log1-log2)

BF_df = pd.DataFrame(columns = results.GI, index = results.GI)
for g1 in results.GI:
    for g2 in results.GI:
        log1 = results['LogEvidence'][results['GI'] == g1].values
        log2 = results['LogEvidence'][results['GI'] == g2].values
        BF_df.loc[g1,g2] = BF(log1, log2)[0]
        
# results.to_csv('BG_logEvidenceAllModels.csv', index=False)

# plot a heatmap
# BF_df_filtered = BF_df[BF_df.columns.drop(list(BF_df.filter(regex='cov')))]
# BF_df_filtered = BF_df_filtered[BF_df_filtered.columns.drop(list(BF_df_filtered.filter(regex='diag')))]
BF_df_filtered = BF_df[BF_df.columns.drop(list(BF_df.filter(regex='40')))]

# BF_df_filtered.drop(["AR3_1.25_cov", "W_7_diag", "AR3_1.5_cov"], inplace = True)
BF_df_filtered.drop([40.0], inplace = True) 

rename_dict = {'W_1': '$W=1$', 'W_2': '$W=2$', 'W_3': '$W=3$', 'W_4': '$W=4$', 'W_7': '$W=7$',
               'AR2': '$AR(2)$', 'AR3': '$AR(3)$', 'AR4': '$AR(4)$',
               5.0: '$GI=5$', 6.0: '$GI=6$', 6.5: '$GI=6.5$', 7.0: '$GI=7$', 
               8.0: '$GI=8$', 9.0: '$GI=9$', 10.0: '$GI=10$', 20.0: '$GI=20$'}
BF_df_filtered = BF_df_filtered.rename(columns=rename_dict, index=rename_dict)


BF_df_filtered = BF_df_filtered.astype(float)
# mask = np.zeros_like(BF_df_filtered, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(BF_df_filtered, annot=False,  center=0)
plt.xlabel('')
plt.ylabel('')
# plt.savefig('BH_BF_heatmap.pdf', bbox_inches = "tight")

def heatmap(start_str, save=False):
    start_str = '$' + start_str
    filter_col = [col for col in BF_df_filtered if col.startswith(start_str)]
    idx = list(BF_df_filtered.index) 
    filter_idx = [i for i in idx if i.startswith(start_str)]
    df = BF_df_filtered.loc[filter_idx, filter_col].astype(int)
    sns.heatmap(df, annot=True, fmt="d",  center=0)
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    if save:
        plt.savefig(save, bbox_inches = "tight")
    else:
        plt.show()

heatmap('GI', save='BH_BF_heatmap_GI.pdf')
heatmap('W', save='BH_BF_heatmap_W.pdf')
heatmap('AR', save='BH_BF_heatmap_AR.pdf')

def BF_interpret(val):
    add = ''
    if val < 0:
        add = ' against'
    val = abs(val)
    if val < 1:
        return 'Non-decisive'
    if val < 3.2:
        return 'Barely' + add
    if val < 10:
        return 'Substantial' + add
    if val < 100:
        return 'Strong' + add
    return 'Decisive' + add

BF_df_filtered_cat = BF_df_filtered.copy()
BF_df_filtered_cat = BF_df_filtered_cat.applymap(lambda x: BF_interpret(x))

# sns.heatmap(BF_df_filtered_cat, annot=False)
# plt.xlabel('')
# plt.ylabel('')