demixed Principal Component Analysis (dPCA)
===========================================

dPCA is a linear dimensionality reduction technique that automatically discovers and highlights the essential features of complex population activities. The population activity is decomposed into a few demixed components that capture most of the variance in the data and that highlight the dynamic tuning of the population to various task parameters, such as stimuli, decisions, rewards, etc.

> D Kobak*, W Brendel*, C Constantinidis, C Feierstein,
A Kepecs, Z Mainen, R Romo, X-L Qi, N Uchida, C Machens<br>
> *Demixed principal component analysis of population activity in higher cortical areas reveals independent representation of task parameters*<br>
> arXiv preprint arXiv:????.???? (2014)<br>
> http://arxiv.org/abs/????.????

This repository provides easy to use Python and MATLAB implementations of dPCA as well as example code.

## Use dPCA

Simple example code for surrogate data can be found in **dpca_demo.py** and **dpca_demo.m**, and is reproduced at the end of this README.

### Python package

To install, first make sure that numpy, scipy, sklearn, itertools and numexpr are avaible. Then do `pip install git+????` or download, unpack and run `python setup.py install`.

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

### Special situations

Email wieland.brendel@bethgelab.org (Python) or dmitry.kobak@neuro.fchampalimaud.org (Matlab) with questions if you don't find your answer here.

#### What if not all parameter combinations are measured?

## Example code

The following code runs dPCA on surrogate in Python and MATLAB respectively. Identical code is in **dpca_demo.py** and **dpca_demo.m**.

### Python example code

```python
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from dpca import dPCA

```

### MATLAB example code

```MATLAB

```
