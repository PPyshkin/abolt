#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:50:33 2024

@author: velix
"""
from numpy import *

from scipy.sparse import csr_matrix, coo_array 

from scipy.sparse.linalg import spsolve

from scipy.linalg import solve

import numpy

from numpy.polynomial.legendre import *

from sys import exit, argv
import os
import time
import argparse
import csv


def savetrajectory(fname, XX, YY, namex = "X", namey = "Y"):
    with open(fname, mode='w') as test_file:
        test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)                                                                   
        test_writer.writerow([namex, namey])      
        for i in range(len(XX)):                
            test_writer.writerow([XX[i], YY[i]])  



def boltsolver(li, le, R):

    Ny = 200
    NPhi = 240
    
    EF = 1
    dy = 1 / (Ny - 1)
    dphi = 2*pi / NPhi
    

    
    alpha = 100000000
    
    Y = linspace(0, 1, Ny)
    PHI = linspace(0, 2*pi - dphi, NPhi)
    
    
    l = le * li / (le + li)
    
    def idxs(k, n):
        return NPhi * k + n % NPhi
    
    
    Chi = arange(Ny*NPhi, dtype=float)
    Chi.fill(0)
    
    E = arange(Ny*NPhi, dtype=float)
    E.fill(0)
    
    
    N = 0
    
    for m in range(1, int(NPhi/2)):
        
        p_m = exp(- alpha**2 * sin(PHI[m])**2)
        
        N += dphi * sin(PHI[m]) * (1 - p_m)
    
    coorow = []
    coocol = []
    cooData = []
    
    print ("Starting filling the matrix...", flush = True)
    print ("li = ",li, " lee = ", le, " R = ", R, flush = True)
    
    for k in range(1, Ny-1):
        for n in range(0, NPhi):
            coorow.append( idxs(k  , n) )
            coocol.append( idxs(k+1, n) )
            cooData.append(sin(PHI[n%NPhi]) / (2*dy))
    
            coorow.append( idxs(k  , n) )
            coocol.append( idxs(k-1, n) )
            cooData.append(- sin(PHI[n%NPhi]) / (2*dy))
            
            coorow.append( idxs(k  , n) )
            coocol.append( idxs(k, n + 1) )
            cooData.append( 1  / (R * dphi))
            
            coorow.append( idxs(k  , n) )
            coocol.append( idxs(k, n ) )
            cooData.append( - 1  / (R * dphi))
    
            coorow.append( idxs(k  , n) )
            coocol.append( idxs(k, n ) )
            cooData.append( 1  / (l))
            
            # ========= relaxation ===================
            
    
            
            for m in range(NPhi):
                coorow.append( idxs(k  , n) )
                coocol.append( idxs(k, m ) )
                cooData.append( - dphi  / ( l))
                
            for m in range(NPhi):
                coorow.append( idxs(k  , n) )
                coocol.append( idxs(k, m ) )
                cooData.append( - dphi * cos(PHI[m%NPhi]) * cos(PHI[n%NPhi])  / (pi * le) )
    
    
    # bottom boundary 
    
    for n in range(1,int(NPhi/2)):
        coorow.append( idxs(0, n) )
        coocol.append( idxs(0, n) )
        cooData.append(1)    
    
        p_n = exp(- alpha**2 * sin(PHI[n])**2)

        coorow.append( idxs(0, n) )
        coocol.append( idxs(0, NPhi-n) )
        cooData.append( - p_n )    
        
              
        for m in range(1, int(NPhi/2)):  
            p_m = exp(- alpha**2 * sin(PHI[m])**2)
            coorow.append( idxs(0, n) )
            coocol.append( idxs(0, NPhi-m) )
            cooData.append( - (1 - p_n) * (1 - p_m) * dphi * sin(PHI[m]) / N   )    
        
    # top boundary
    
    for n in range(int(NPhi/2)+1, NPhi):
        coorow.append( idxs(Ny-1, n) )
        coocol.append( idxs(Ny-1, n) )
        cooData.append(1)    
    
        p_n = exp(- alpha**2 * sin(PHI[n])**2)
    
        coorow.append( idxs(Ny-1, n) )
        coocol.append( idxs(Ny-1, NPhi-n) )
        cooData.append( - p_n * 1)    
        
        for m in range(1, int(NPhi/2)):  
            p_m = exp(- alpha**2 * sin(PHI[m])**2)
            coorow.append( idxs(0, n) )
            coocol.append( idxs(0, m) )
            cooData.append( - (1 - p_n) * (1 - p_m) * dphi * sin(PHI[m]) / N   )        
        
        
    
    # bottom, additional eqs.
    
    for n in range(int(NPhi/2), NPhi+1):
        E[idxs(0, n)] = EF * cos(PHI[n%NPhi])
        
        coorow.append( idxs(0  , n) )
        coocol.append( idxs(1, n) )
        cooData.append(sin(PHI[n%NPhi]) / (dy))
    
        coorow.append( idxs(0  , n) )
        coocol.append( idxs(0, n) )
        cooData.append(- sin(PHI[n%NPhi]) / (dy))
        
        coorow.append( idxs(0  , n) )
        coocol.append( idxs(0, n + 1) )
        cooData.append( 1  / (R * dphi))
        
        coorow.append( idxs(0  , n) )
        coocol.append( idxs(0, n ) )
        cooData.append( - 1  / (R *dphi))
    
        coorow.append( idxs(0  , n) )
        coocol.append( idxs(0, n ) )
        cooData.append( 1  / (l))
        
        # ========= relaxation ===================
        
    
        for m in range(NPhi):
            coorow.append( idxs(0  , n) )
            coocol.append( idxs(0, m ) )
            cooData.append( - dphi  / (2 * pi * l))
            
        for m in range(NPhi):
            coorow.append( idxs(0  , n) )
            coocol.append( idxs(0, m ) )
            cooData.append( - dphi * cos(PHI[m%NPhi]) * cos(PHI[n%NPhi])  / (pi * le) )
    
    #top, add
    
    for n in range(0, int(NPhi/2)+1):
        E[idxs(Ny-1, n)] = EF * cos(PHI[n])
        coorow.append( idxs(Ny-1  , n) )
        coocol.append( idxs(Ny-1, n) )
        cooData.append(sin(PHI[n%NPhi]) / (dy))
    
        coorow.append( idxs(Ny-1  , n) )
        coocol.append( idxs(Ny-2, n) )
        cooData.append(- sin(PHI[n%NPhi]) / (dy))
        
        coorow.append( idxs(Ny-1  , n) )
        coocol.append( idxs(Ny-1, n + 1) )
        cooData.append( 1  / (R * dphi))
        
        coorow.append( idxs(Ny-1  , n) )
        coocol.append( idxs(Ny-1, n ) )
        cooData.append( - 1  / (R * dphi))
    
        coorow.append( idxs(Ny-1  , n) )
        coocol.append( idxs(Ny-1, n ) )
        cooData.append( 1  / (l))
        
        # ========= relaxation ===================
        
    
        
        for m in range(NPhi):
            coorow.append( idxs(Ny-1  , n) )
            coocol.append( idxs(Ny-1, m ) )
            cooData.append( - dphi  / (2 * pi * l))
            
        for m in range(NPhi):
            coorow.append( idxs(Ny-1  , n) )
            coocol.append( idxs(Ny-1, m ) )
            cooData.append( - dphi * cos(PHI[m%NPhi]) * cos(PHI[n%NPhi])  / (pi * le) )
    
    # driving field
    
    for k in range(1, Ny-1):
        for n in range(0, NPhi):
            E[idxs(k, n)] = EF * cos(PHI[n])
    
    
    main_coo_matrix = coo_array((cooData, (coorow, coocol)), shape=(Ny*NPhi, Ny*NPhi))
        
    A = csr_matrix(main_coo_matrix)
    
    print ("Matrix is ready, starting to solve...", flush = True)
    # ============== MAIN SOLVER ==================================
    
    chi = spsolve(A, E)
    
    # =============================================================
    
    
    CHI = chi.reshape(Ny, NPhi)
    
    def curr():
        
        j = arange(Ny*NPhi, dtype=float)
        j.fill(0)    
        for k in range(Ny):
            for m in range(NPhi):
                j[k] += CHI[k, m] * cos(PHI[m]) * dphi * dy / pi / EF
        return j

    def curr_y():
        
        j = arange(Ny*NPhi, dtype=float)
        j.fill(0)    
        for k in range(Ny):
            for m in range(NPhi):
                j[k] += CHI[k, m] * sin(PHI[m]) * dphi * dy / pi / EF
        return j
    
    def dn():
        
        n = arange(Ny*NPhi, dtype=float)
        n.fill(0)    
        for k in range(Ny):
            for m in range(NPhi):
                n[k] += CHI[k, m] * dphi * dy / 2 / pi 
        return n
    
    J = curr()
    JY = curr_y()
        
    I =  sum(J)
    
    IY =  sum(JY)
    
    RR = li / I        
    
    n = dn()
    
    charge = sum(n)
    
    print("I = ", I, "R / R_0 = ", RR, "Charge = ", charge, "IY = ", IY)
    
    return I, RR, charge, IY, JY


LEE = [100, 10, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

li = 3

LEFF = []

WR = [0.00000001, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3]

le = 100000000000

R = 1000000000000

"""
for le in LEE:    
    leff, Resist = boltsolver(li, le, R)
    LEFF.append(leff)
"""    

for wr in WR:    
    leff, Resist, q, IY, JY = boltsolver(li, le, -1/wr)
    LEFF.append(Resist)

savetrajectory("Raich1.csv", WR, LEFF)   