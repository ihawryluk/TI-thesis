# Referenced thermodynamic integration and applications in Bayesian model selection

code used for the MRes thesis by Iwona Hawryluk

Supervised by Dr Thomas A. Mellan and Dr Samir Bhatt

Imperial College London, Department of Infectious Disease Epidemiology

MRes in Biomedical Research, Epidemiology, Evolution and Control of  Infectious Disease stream

August 2020

## Abstract:

Model selection is a fundamental part of building Bayesian statistical models, widely used in the field of epidemiology. The model selection task requires computing the ratio of the normalising constants (or model evidence), known as Bayes factors. Normalising constants often come in the form of intractable, high-dimensional integrals, therefore special probabilistic techniques need to be applied to correctly estimate the Bayes factors.
One such methods is thermodynamic integration (TI), which can be used to estimate the ratio of two models' evidence by integrating over the continuous geometric *paths* between the two un-normalised densities. 
In this paper we introduce a modified TI algorithm, called referenced TI, which allows to compute a single model's evidence in an efficient way by using a Gaussian reference density, whose normalising constant is known. 
We show that referenced TI is an asymptotically exact way of calculating the normalising constant of a single model, which converges to the correct result much faster than for example power posteriors methods.
We illustrate the implementation of the algorithm on an informative 1-dimensional example and apply it to a linear regression problem and a model of the COVID-19 epidemic in South Korea. 

## Repository description:

The code in this repository is organised as follows:

- 1-dimensional example used in section 3.1 of the thesis - *1D_with_plots.py*
- 2-dimensional example with constrained parameters, section 3.2 - *2D_with_correction.py*
- linear regression radiata pine example, section 4.1 - *linear_regression* folder
  - *radiata_friel_data.csv* - radiata pine data as used in Friel and Wyse (2011)
  - *radiata_pine_example.py* - code to generate outputs of a Laplace approximation, model switch TI, referenced TI and power posterior with 11 and 100 temperature placements
  - *radiata_repeated_runs.py* - outputs of 15 runs of the referenced TI and power posteriors method evaluated over 15 runs of the algorithms
- COVID-19 for South Korea with a Bellman-Harris process, section 4.2 - *bellman_harris* folder
  - SouthKoreaData.csv - case data for South Korea
  - *bellman_harris_AR2.py* (_AR3, _AR4) - autoregressive model with 2-3-4 days lag
  - *bellman_harris_W.py* - model with a fixed sliding window length (user has to change the parameter W value in the code)
  - *fixed_GI/bellman_harris_fixed_GI.py* - model with a fixed GI parameter
  - *load_bellman_harris_outputs.py* - loads and processes the outputs of all variants of the model; calculates log-evidence and Bayes Factors based on the Laplace approximation and the referenced TI outputs
  - *posterior_plots.py* - plots of the posterior distributions and generated quantities of different models

