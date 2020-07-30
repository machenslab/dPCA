from numba import jit
import numpy as np

@jit(nopython=True)
def shuffle2D(X):
    idx = np.where(~np.isnan(X[:,0]))[0]
    K = X.shape[1]
    T = len(idx)
    randints = np.random.rand(T) * np.arange(T)
    
    for i in range(T-1, 0, -1):
        j = round(randints[i])
        n,m = idx[i], idx[j]
        for k in range(K):
            X[n,k], X[m,k] = X[m,k], X[n,k]

@jit(nopython=True)
def classification(class_means, test):
    Q = class_means.shape[0]
    T = class_means.shape[1]
    performance = np.zeros(T)
    distance = 0.

    # classify every data point in test according to nearest class mean
    for t in range(T):
        for p in range(Q):
            argmin = 0
            distance = abs(class_means[argmin,t] - test[p,t])
            
            # find closest class mean
            for q in range(1,Q):
                if abs(class_means[q,t] - test[p,t]) < distance:
                    distance = abs(class_means[q,t] - test[p,t])
                    argmin = q

            # add 1 if class is correct
            if argmin == p:
                performance[t] += 1.

        performance[t] /= Q

    return performance

@jit(nopython=True)
def denoise_mask(mask, n_consecutive):
    subseq = 0
    N = mask.shape[0]
        
    for n in range(N):
        if mask[n] == 1:
            subseq += 1
        else:
            if subseq < n_consecutive:
                for k in range(n-subseq,n):
                    mask[k] = 0
                    
    return mask