cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def shuffle2D(np.ndarray[double, ndim=2] X):
    cdef np.ndarray[long, ndim=1] idx = np.where(~np.isnan(X[:,0]))[0]
    cdef unsigned int K = X.shape[1]
    cdef unsigned int T = len(idx)
    cdef unsigned int i,j,n,m,k
    cdef np.ndarray[long, ndim=1] randints = np.around(np.random.rand(T)*np.arange(T)).astype(int)
    
    for i in xrange(T-1, 0, -1):
        j = randints[i]
        n,m = idx[i], idx[j]
        for k in xrange(K):
            X[n,k], X[m,k] = X[m,k], X[n,k]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def classification(np.ndarray[double, ndim=2] class_means, np.ndarray[double, ndim=2] test):
    cdef int Q = class_means.shape[0]
    cdef int T = class_means.shape[1]
    cdef np.ndarray[double, ndim=1] performance = np.zeros(T)
    cdef float distance = 0.
    cdef int argmin

    # classify every data point in test according to nearest class mean
    for t in xrange(T):
        for p in xrange(Q):
            argmin = 0
            distance = abs(class_means[argmin,t] - test[p,t])
            
            # find closest class mean
            for q in xrange(1,Q):
                if abs(class_means[q,t] - test[p,t]) < distance:
                    distance = abs(class_means[q,t] - test[p,t])
                    argmin = q

            # add 1 if class is correct
            if argmin == p:
                performance[t] += 1.

        performance[t] /= Q

    return performance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def denoise_mask(int[::1] mask, int n_consecutive):
    cdef int subseq = 0
    cdef int N = mask.shape[0]
    cdef int k, n
        
    for n in xrange(N):
        if mask[n] == 1:
            subseq += 1
        else:
            if subseq < n_consecutive:
                for k in xrange(n-subseq,n):
                    mask[k] = 0
                    
    return mask