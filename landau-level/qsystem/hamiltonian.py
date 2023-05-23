import numpy as np
from constants import *
from particles import *
from lanczos import *
from scipy.sparse.linalg import eigsh


import time


#hamiltonian builder
class hamiltonian:
    def __init__(self, N, spacing, potential, extent, dim = 2, E_min = 0): 
        #args:
        #N: number of particles
        #spacing: number of intervals/gridpoints on meshgrid
        #V: potential term
        #extent: spacial extent (come back for units)
        #dim: dimensions
        self.N = N
        self.spacing = spacing
        self.potential = potential
        self.extent = extent
        self.observable_count = 0
        self.dim = dim
        self.E_min = E_min
        #print('Variables defined...')

        #dx finite difference value
        self.dx = extent/spacing

        #ensure N is an integer
        if type(N) != int:
            raise Exception('Particle number N must be of int type.')
        
        #ensure dimensions is correct
        if dim not in range(1,3,1):
            raise Exception('Dimension must be either 1, 2 or 3')
        
        if N == 1:
            self.particle = single_particle()
        elif N == 2:
            self.particle = two_particle()
        else:
            print('Particle systems larger than N = 2 not supported.')
        #print('Particle system defined...')

        self.particle.matrix_operators(self)
        
        #construct Hamiltonian H = T+V (if py and px are not second order, they appear in the potential term V)
        self.T = self.particle.kinetic_term(self)
        #print('T matrix initialized...')

        self.V = self.potential_term()
        #print('V matrix initialized...')

        print('Hamiltonian constructed...')
        print('--------------------------')

    def potential_term(self):
        V = self.potential(self.particle)

        return V

    def denseT(self):
        return self.T.todense()

    def denseV(self):
        return self.V.todense()

    def matrix(self):
        return self.T + self.V

    def solve(self, max_state):
        #args:
        #iteration (m): number of iterations to perform
        H = self.T + self.V
        print("Hamiltonian constructed...")

        t0 = time.time()

        
        #x = np.transpose(np.ones(self.spacing**2))
        #T, V = iterate(H, x, iteration)

        #eigVal, eigVec = tri_eig_decompose(T)
        
        eigVal, eigVec = eigsh(H, k = max_state, which = 'LM', sigma = 0)
        print("Diagonalization took ", time.time() - t0," seconds")

        #normalizing
        return eigVal/eV, eigVec/np.sqrt(self.dx**self.dim)
