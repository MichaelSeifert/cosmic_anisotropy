#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:12:25 2024

@author: mseifer1
"""

from ABvacmetric0 import ABvacmetric0
import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions import helperfunctions
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings("ignore")

def main():

    # unpack parameters
    O_m, O_r, O_L, O_k, O_B, B0, h, n0vec = 0.5, 0.02, 0.3, 0.079375, 0.1, 0.025, 1, [2/3, 2/3, -1/3]
    
    # calculate ABvacmetric0 for these parameters
    solution, contractFlag, times = ABvacmetric0([O_m, O_r, O_L, O_k, O_B, B0])
            
    #the following section interpolates over angle values to reduce the number of calls to helperfunctions. Huge performace increase, especially for large datasets.
    #could potentially be even more efficient to solve a PDE instead of an ODE + interpolation to get helperfunctions.
    
    #set points to interpolate between
    
    
    # Logarithmically distributed from 10^(-minintval) to 1, with a certain num.
    # of points per decade
    minintval = 6
    intptsperdecade = 5
    c_theta_points = np.exp(np.linspace(-minintval*np.log(10.),0,minintval*intptsperdecade+1))
    
    print(np.exp(np.linspace(-np.log(10.),0,intptsperdecade+1)))

    # Linear distribution
    # c_theta_points=np.linspace(0.000001,1,100)
    
    # log dist. up to 0.1, linear from 0.1 to 1
    # minintval = 6
    # intptsperdecade = 6
    # c_theta_points = np.concatenate(
    #     (np.exp(np.linspace(-minintval*np.log(10.),-np.log(10.),(minintval-1)*intptsperdecade,endpoint=False)),
    #     np.linspace(0.1,1,20))
    #     )
    
    # print("Theta interpolation points:\n", c_theta_points)
    
    
    z_points = np.arange(0, 2.0, .01)
    # z_points = np.concatenate((np.arange(0, 1.8, 0.01), np.arange(1.801, 2, 0.001)))
    # print("z interpolation points:\n", z_points)
    
    #store results
    trows = []
    qrows = []
    psirows = []
    
    #calculate helperfuncs for each theta value, and each z values within that. Append to points lists
    for thetaval in c_theta_points:
        helperfuncs, redshifts = helperfunctions([solution, thetaval])
        trow = []
        qrow = []
        psirow = []
        for zval in z_points:
            teval, qoval, psioval = helperfuncs(zval)
            trow.append(teval)
            qrow.append(qoval)
            psirow.append(psioval)
        trows.append(np.array(trow))
        qrows.append(np.array(qrow))
        psirows.append(np.array(psirow))
        
    tevals = np.array(trows)
    qovals = np.array(qrows)
    psiovals = np.array(psirows)
            
    #create the interpolating functions
    degx=1
    degy=1
    interpolate_te = RectBivariateSpline(c_theta_points, z_points, tevals, 
                                         kx=degx, ky=degy)
    interpolate_qo = RectBivariateSpline(c_theta_points, z_points, qovals, 
                                         kx=degx, ky=degy)
    interpolate_psio = RectBivariateSpline(c_theta_points, z_points, psiovals, 
                                           kx=degx, ky=degy)

    
    
    # sol, redshifts = helperfunctions([equation, 0.5])
    
    fig, ax = plt.subplots(3, 1, sharex = True)
    
    cthvals=np.linspace(0.1,0.5,9)
    for cth in reversed(cthvals):
        helpersol, redshifts = helperfunctions([solution, cth])
        # print(redshifts)
        redshifts = np.linspace(0.01,2.,1000)
        t, q, psi = helpersol(redshifts)        
        ax[0].plot(redshifts, interpolate_te(cth,redshifts)[0] - t, label=round(cth,2))
        ax[1].plot(redshifts, interpolate_qo(cth,redshifts)[0] - q)
        ax[2].plot(redshifts, interpolate_psio(cth,redshifts)[0] - psi)

    # ax[0].set_yscale('asinh')    
    # ax[1].set_yscale('asinh')    
    # ax[2].set_yscale('asinh')    
    ax[0].set_ylabel('tau_e')
    ax[1].set_ylabel('q_o')
    ax[2].set_ylabel('psi_o')
    ax[2].set_xlabel('z')
    fig.legend(loc='outside upper right')
    plt.show()

        
        
    
if(__name__ == "__main__"):
    main()