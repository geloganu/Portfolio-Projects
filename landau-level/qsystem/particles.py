import numpy as np
from hamiltonian import *
from constants import *
from scipy.sparse import diags
from scipy.sparse import kron
from scipy.sparse import eye
import scipy
#particle builder for hamiltonian builder

#single particle case N=1
class single_particle:
    def __init__(self, m = me, spin = None):
        '''
        m: mass of electron set to %me by default
        spin: set to 'None' if spin polarized
        '''

        self.m = m
        self.spin = spin

    def matrix_operators(self, H):
        H.ndim = 2

        x = diags([np.linspace(-H.extent/2, H.extent/2, H.spacing)], [0])
        y = diags([np.linspace(-H.extent/2, H.extent/2, H.spacing)], [0])
        I = eye(H.spacing)

        self.x = kron(I,x)
        self.y = kron(y,I)

        diff_x = diags([-1., 0., 1.], [-1, 0, 1] , shape=(H.spacing, H.spacing))*1/(2*H.dx)
        diff_y = diags([-1., 0., 1.], [-1, 0, 1] , shape=(H.spacing, H.spacing))*1/(2*H.dx)

        self.px = kron(I, - hbar *1j * diff_y)
        self.py = kron(- hbar *1j * diff_x, I)
        
        self.I = kron(I,I)

    def kinetic_term(self, H):

        I = eye(H.spacing)
        T_ =  diags([-2., 1., 1.], [0,-1, 1] , shape=(H.spacing, H.spacing))* -0.5 /(self.m*H.dx**2)
        if H.dim ==1:
            T = T_

        elif H.dim==2:
            T =  kron(T_,I) + kron(I,T_)
        return T
       

class two_particle:
    def __init__(self, m = me, spin = None,  N = 2):
        """
        args:
        N: number of particles (for N != 1) for now N = 2
        m: mass of electron set to %me by default
        spin: set to 'None' if spin polarized
        """
        self.N = N 
        self.m = m
        self.spin = spin
    
    def matrix_operators(self, H):
        H.ndim = 4

        #cartesian coord space
        x1 = diags([np.linspace(-H.extent/2, H.extent/2, H.spacing)], [0])
        y1 = diags([np.linspace(-H.extent/2, H.extent/2, H.spacing)], [0])
        x2 = diags([np.linspace(-H.extent/2, H.extent/2, H.spacing)], [0])
        y2 = diags([np.linspace(-H.extent/2, H.extent/2, H.spacing)], [0])

        #identity matrices
        I = eye(H.spacing)
        Identity = kron(I,I)

        #cartesian coord space in hilbert space
        self.x1 = kron(kron(x1,I),Identity)
        self.y1 = kron(kron(I,y1),Identity)
        self.x2 = kron(Identity,kron(x2,I))
        self.y2 = kron(Identity,kron(I,y2))

        #momentum operator matrix template
        diff_matrix = - hbar *1j * diags([-1., 0., 1.], [-1, 0, 1] , shape=(H.spacing, H.spacing))*1/(2*H.dx)

        self.px1 = kron(kron(diff_matrix,I),Identity)
        self.py1 = kron(kron(I,diff_matrix),Identity)
        self.px2 = kron(Identity,kron(diff_matrix,I))
        self.py2 = kron(Identity,kron(I,diff_matrix))

        #seperation coordinate space
        xx1, yy1, xx2 , yy2  = np.mgrid[ -H.extent/2: H.extent/2:H.spacing*1j, -H.extent/2: H.extent/2:H.spacing*1j, -H.extent/2: H.extent/2:H.spacing*1j, -H.extent/2: H.extent/2:H.spacing*1j]

        r1 = np.sqrt(xx1**2+yy1**2)
        r2 = np.sqrt(xx2**2+yy2**2)

        TOL = 0.0001
        rsep = np.abs(r1 - r2)
        rsep_inv = np.where(rsep < TOL, 1/TOL, 1/rsep)

        self.r1 = diags([r1.reshape(H.spacing ** H.ndim)],[0])
        self.r2 = diags([r2.reshape(H.spacing ** H.ndim)],[0])
        self.rsep = diags([rsep.reshape(H.spacing ** H.ndim)], [0])   
        self.rsep_inv = diags([rsep_inv.reshape(H.spacing ** H.ndim)], [0])   

        self.I = kron(Identity,Identity)
        


    def kinetic_term(self, H):

        I = eye(H.spacing)
        Identity = kron(I,I)

        T_ =  diags([-2., 1., 1.], [0,-1, 1] , shape=(H.spacing, H.spacing))* -0.5 /(self.m*H.dx**2)

        T = kron(kron(T_,I),Identity) + kron(kron(I,T_),Identity) + kron(Identity, kron(T_,I)) + kron(Identity, kron(I,T_))
        
        return T

