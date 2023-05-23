import numpy as np
from constants import *
from hamiltonian import *
from particles import *

def landau(particle):
    #constants
    #e = electron charge
    #B = field strength
    #m = me
    
    #constants
    B = 150*T
    m = me
    
    #cyclotron frequency omega = eB/m
    omega = e*B/m
    
    #angular momentum term
    Lz = -particle.px @ particle.y + particle.py @ particle.x
    
    #x^2 + y^2
    coord_term = np.dot(particle.x,particle.x)+np.dot(particle.y,particle.y)
    
    return -omega*Lz/2 + m/2*(omega/2)**2*coord_term

def two_particle_landau(particle):
    #constants
    #e = electron charge
    #B = field strength
    #m = me
    
    #constants
    B = 150*T
    m = me
    omega = e*B/2*m

    #x^2 + y^2
    coord_term = 1/2 * m * omega**2 * (particle.r1**2 + particle.r2**2)
    print(coord_term.shape)

    #angular momentum term
    Lz1 = -1j * hbar * (-particle.px1 @ particle.y1 + particle.py1 @ particle.x1)
    Lz2 = -1j * hbar * (-particle.px2 @ particle.y2 + particle.py2 @ particle.x2)
    
    angm_term = - omega * (Lz1+Lz2)
    print(angm_term.shape)
    
    #coulomb interaction term
    k = e
    coulomb_term = k * particle.rsep_inv


    return coord_term + angm_term + coulomb_term

H = hamiltonian(N = 2, spacing = 50, potential = two_particle_landau, extent = 20*Å, dim = 2)
eigVal, eigVec = H.solve(max_state=30)
print(eigVal)
