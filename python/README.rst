demixed Principal Component Analysis (dPCA)
===========================================

dPCA is a linear dimensionality reduction technique that automatically discovers and highlights the essential features of complex population activities. The population activity is decomposed into a few demixed components that capture most of the variance in the data and that highlight the dynamic tuning of the population to various task parameters, such as stimuli, decisions, rewards, etc.

.. code-block::

    @article{kobak2016dpca,
       title={Demixed principal component analysis of neural population data},
       volume={5},
       ISSN={2050-084X},
       url={http://dx.doi.org/10.7554/eLife.10989},
       DOI={10.7554/elife.10989},
       journal={eLife},
       publisher={eLife Sciences Publications, Ltd},
       author={Kobak, Dmitry and Brendel, Wieland and Constantinidis, Christos and Feierstein, Claudia E and Kepecs, Adam and Mainen, Zachary F and Qi, Xue-Lian and Romo, Ranulfo and Uchida, Naoshige and Machens, Christian K},
       year={2016},
       month={Apr}
    }

🚀 Quickstart
-------------

.. code-block:: bash

   pip install dpca


🎉 Use dPCA
-----------
Simple example code for surrogate data can be found in [**dpca_demo.ipynb**](http://nbviewer.ipython.org/github/wielandbrendel/dPCA/blob/master/python/dPCA_demo.ipynb) and **dpca_demo.m**.

API of dPCA is similar to sklearn. To use dPCA, you should first import dPCA and initialize it before callling the fitting function,

.. code-block:: python

   from dPCA import dPCA
   dpca = dPCA.dPCA(labels, n_components, regularizer)
   Z = dpca.fit_transform(X)
   
The required initialization parameters are:

* **X:** A multidimensional array containing the trial-averaged data. E.g. X[n,t,s,d] could correspond to the mean response of the *n*-th neuron at time *t* in trials with stimulus *s* and decision *d*. The observable (e.g. neuron index) needs to come first.
* **labels:** Optional; list of characters with which to describe the parameter axes, e.g. 'tsd' to denote time, stimulus and decision axis. All marginalizations (e.g. time-stimulus) are refered to by subsets of those characters (e.g. 'ts').
* **n_components:** Dictionary or integer; if integer use the same number of components in each marginalization, otherwise every (key,value) pair refers to the number of components (value) in a marginalization (key).

More detailed documentation, and additional options, can be found in **dpca.py**.
