import numpy as np
from numpy import linalg
import scipy

#normalization
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

#lanczos iteration
def iterate(H, x, m: int):
    n = len(H)

    #ensure n>=m
    if m>n:
        m = n

    T = np.zeros((m,m))
    V = np.zeros((n,m))
    x =  normalize(x) #first krylov vector
    xt = np.transpose(x) #first krylov vector transpose
    V[:,0] = xt #first krylov subspace basis vector

    """first iteration"""
    w = np.dot(H, V[:,0]) # new candidate vector
    alpha = np.dot(w.conj(), V[:,0]) #a0 for tridiagonal matrix
    T[0,0] = alpha
    w = w - alpha*V[:,0] #gram-schmidt
    print('-----------------------------')
    print('j=0')
    print('w', w)

    """subsequent iterations"""
    for j in range(1,m):
        print('-----------------------------')
        print('j=',j)
        print('ok')
        beta = np.sqrt(np.dot(w,w))
        T[j-1,j] = beta
        T[j,j-1] = beta

        V[:,j] = w / beta #add if b!=0 else here?

        #gram schmidt
        for i in range(j-1):
            V[:,j] = V[:,j] - np.dot(V[:,j].conj(), V[:,i])*V[:,i]

        #normalize
        V[:,j] = normalize(V[:,j])

        #define new w
        w = np.dot(H, V[:,j])

        #define new alpha
        alpha = np.dot(w.conj(), V[:,j])
        T[j,j] = alpha

        #define new w for next iteration
        w = w - alpha*V[:,j] - beta*V[:,j-1] 
    
    return T, V



def tri_eig_decompose(T):
    #1D array of diagonal elements 
    diag = np.diagonal(T)
    
    #1D array of off diagonal elements
    n = len(T)
    off_diag = np.array([])
    for i in range(0,n-1):
        off_diag = np.append(off_diag, T[i,i+1])
        
    scipy.linalg.eigh_tridiagonal(diag, off_diag)
    return w, v
    
    