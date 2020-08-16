# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:44:59 2020

@author: iwona
"""


import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

PATH = 'C:/Users/iwona/Desktop/TI/bellman_harris/models_fits/'

namesAR = ['AR2', 'AR3', 'AR4']
namesW = ['W1', 'W2', 'W3', 'W4', 'W7']

namesAR = ['AR3', 'AR2']
namesW = ['W2', 'W7']
namesGI = ['GI8', 'GI20']


model_AR = {}
model_W = {}
model_GI = {}

for m in namesAR:
    model_AR.update({m: pd.read_csv(PATH + m + '.csv')})

for m in namesW:
    model_W.update({m: pd.read_csv(PATH + m + '.csv')})
    
for m in namesGI:
    model_GI.update({m: pd.read_csv(PATH + m + '.csv')})
    
params_names = {'Rt': '$R_t$', 'weekly_rho': '$\rho_{weekly}$', 'weekly_sd': '$\sigma$',
                'GI': '$GI$', 'R0': '$R_0$', 'prediction': 'Predictions',
                'phi': '$\phi$'}

model_names = {'AR3': '$AR(3)$', 'AR2': '$AR(2)$',
               'W2': '$W=2$', 'W7': '$W=7$',
               'GI8': '$GI=8$', 'GI20': '$GI=20$'}

def plot_posterior(param_str, models):
    for m in models.keys():
        df = models[m]
        vals = df[[col for col in df if col.startswith(param_str)]].values
        if m in model_names.keys():
            sns.distplot(vals, hist = False, kde=True, label = model_names[m])
        else:
            sns.distplot(vals, hist = False, kde=True, label = m)
        # sns.kdeplot(vals, shade=True)
        plt.legend()
    plt.xlabel(params_names[param_str])
    # plt.show()
    
def plot_posterior_means_per_day(param_str, models, show_ylab = False):
    for m in models.keys():
        df = models[m]
        vals = df[[col for col in df if col.startswith(param_str)]]
        vals_means = list(vals.mean(axis=0))
        x = np.arange(len(vals_means)) + 1
        if m in model_names.keys():
            plt.plot(x, vals_means, label = model_names[m], ls='--')
        else:
            plt.plot(x, vals_means, label = m)
    plt.legend()
    if show_ylab:
        plt.ylabel(params_names[param_str])
    plt.xlabel('Days')
    # plt.show()
    
    
plot_posterior('R0', model_AR)
plot_posterior('R0', model_W)
plot_posterior('R0', model_GI)

plot_posterior('phi', model_AR)
plot_posterior('phi', model_W)
plot_posterior('phi', model_GI)

plot_posterior('GI', model_AR)
plot_posterior('GI', model_W)

plot_posterior('weekly_rho', model_AR)
plot_posterior('weekly_rho', model_W)
plot_posterior('weekly_rho', model_GI)

plot_posterior('weekly_sd', model_AR)
plot_posterior('weekly_sd', model_W)
plot_posterior('weekly_sd', model_GI)

plot_posterior_means_per_day('prediction', model_AR)
plot_posterior_means_per_day('prediction', model_W)
plot_posterior_means_per_day('prediction', model_GI)


plot_posterior_means_per_day('Rt', model_AR)
plot_posterior_means_per_day('Rt', model_W)
plot_posterior_means_per_day('Rt', model_GI)

plot_posterior_means_per_day('SI_rev', model_AR)
plot_posterior_means_per_day('SI_rev', model_W)

plot_posterior_means_per_day('weekly_effect', model_AR)
plot_posterior_means_per_day('weekly_effect', model_W)
plot_posterior_means_per_day('weekly_effect', model_GI)

plot_posterior_means_per_day('prediction', model_AR)
plot_posterior_means_per_day('prediction', model_W)
plot_posterior_means_per_day('prediction', model_GI)

def plot_dRtdt(models):
    for m in models.keys():
        df = models[m]
        vals = df[[col for col in df if col.startswith('Rt')]]
        vals_means = list(vals.mean(axis=0))
        x = np.arange(len(vals_means)-1) + 1
        dtRt = []
        for i in range(1,len(vals_means)):
            dtRt.append(vals_means[i]-vals_means[i-1])
   
        if m in model_names.keys():
                plt.plot(x, dtRt, label = model_names[m])
        else:
            plt.plot(x, dtRt, label = m)
        plt.legend()
        # if show_ylab:
        #     plt.ylabel(params_names['Rt'])
        plt.xlabel('Days')

# plots to save
def plot_posterior_both_models(param_str, xlim, ylim):
    plt.subplot(1,3,1)
    plot_posterior(param_str, model_AR)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplot(1,3,2)
    plot_posterior(param_str, model_W)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])
    plt.subplot(1,3,3)
    plot_posterior(param_str, model_GI)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])
    plt.tight_layout()
    # plt.show()
    
SKdata = pd.read_csv('C:/Users/iwona/Desktop/TI/bellman_harris/SouthKoreaData.csv')
cases = SKdata.cases.values
days = np.arange(len(cases))
def plot_posterior_means_per_day_both_models(param_str, xlim, ylim):
    plt.subplot(1,3,1)
    plt.bar(days, cases, edgecolor='black', linewidth=0.2, width=1.0, alpha = 0.5)
    plot_posterior_means_per_day(param_str, model_AR, True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplot(1,3,2)
    plt.bar(days, cases, edgecolor='black', linewidth=0.2, width=1.0, alpha = 0.5)
    plot_posterior_means_per_day(param_str, model_W)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])
    plt.subplot(1,3,3)
    plt.bar(days, cases, edgecolor='black', linewidth=0.2, width=1.0, alpha = 0.5)
    plot_posterior_means_per_day(param_str, model_GI)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])
    plt.tight_layout()
    # plt.show()
    
plt.figure(figsize = [10,3])
plot_posterior_both_models('phi', [0,20], [0, 0.65])
plt.savefig(PATH + 'BH_phi.pdf')

plt.figure(figsize = [10,3])
plot_posterior_both_models('weekly_sd', [0,1.5], [0,9])
plt.savefig(PATH + 'BH_sd.pdf')

plt.figure(figsize = [10,3])
plot_posterior_means_per_day_both_models('Rt', [0,201], [0,33])
plt.savefig(PATH + 'BH_Rt.pdf')

plt.figure(figsize = [10,3])
plot_posterior_means_per_day_both_models('prediction', [0,201], [0,1000])
plt.savefig(PATH + 'BH_cases.pdf')
