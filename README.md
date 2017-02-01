demixed Principal Component Analysis (dPCA)
===========================================

dPCA is a linear dimensionality reduction technique that automatically discovers and highlights the essential features of complex population activities. The population activity is decomposed into a few demixed components that capture most of the variance in the data and that highlight the dynamic tuning of the population to various task parameters, such as stimuli, decisions, rewards, etc.

> D Kobak<sup>+</sup>, W Brendel<sup>+</sup>, C Constantinidis, CE Feierstein,
A Kepecs, ZF Mainen, X-L Qi, R Romo, N Uchida, CK Machens<br>
> **Demixed principal component analysis of neural population data**<br>
> eLife 2016, https://elifesciences.org/content/5/e10989<br>
> (arXiv link: http://arxiv.org/abs/1410.6031)

This repository provides easy to use Python and MATLAB implementations of dPCA as well as example code.

## Use dPCA

Simple example code for surrogate data can be found in [**dpca_demo.ipynb**](http://nbviewer.ipython.org/github/wielandbrendel/dPCA/blob/master/python/dPCA_demo.ipynb) and **dpca_demo.m**.

### Python package

The Python package is tested against Python 2.7 and Python 3.4. To install, first make sure that numpy, cython, scipy, sklearn, itertools and numexpr are avaible. Then copy the files from the Python subfolder to a location in the Python search path.

Alternatively, from the terminal you can install the package by running:

```
$  cd /path/to/dPCA/python
$  python setup.py install
```

API of dPCA is similar to sklearn. To use dPCA, you should first import dPCA,  
`from dpca import dPCA`  
then initialize it,    
`dpca = dPCA(labels, n_components, regularizer)`    
then call the fitting function on your data to get the latent components Z,    
`Z = dpca.fit_transform(X)`.

The required initialization parameters are:
- *X* - A multidimensional array containing the trial-averaged data. E.g. X[n,t,s,d] could correspond to the mean response of the *n*-th neuron at time *t* in trials with stimulus *s* and decision *d*. The observable (e.g. neuron index) needs to come first.
- *labels* - Optional; list of characters with which to describe the parameter axes, e.g. 'tsd' to denote time, stimulus and decision axis. All marginalizations (e.g. time-stimulus) are refered to by subsets of those characters (e.g. 'ts').
- *n_components* - Dictionary or integer; if integer use the same number of components in each marginalization, otherwise every (key,value) pair refers to the number of components (value) in a marginalization (key).

More detailed documentation, and additional options, can be found in **dpca.py**.

### MATLAB package

Add the Matlab subfolder to the Matlab search path.

Example code in `dpca_demo.m` generates surrogate data and provides a walkthrough for running PCA and dPCA analysis and  plotting the results.

### Support

Email wieland.brendel@bethgelab.org (Python) or dmitry.kobak@neuro.fchampalimaud.org (Matlab) with any questions.

### Contributors

A big thanks for 3rd party contributions goes to **cboulay**.
