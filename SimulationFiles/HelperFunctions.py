import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ABvacmetric0 import ABvacmetric0

def vectorfield(z, w, p):
    
    """
    Defines the differential equations for ABvacmetric0 derived from Eq 9b and 9c

    Arguments:
        w :  vector of the state variables:
                  w = [t, q, psi]
        z :  red shift
        p :  vector of the parameters
                  p = [ABsolution, theta_0 ]
                      ABsolution is the ODE solution object from ABvacmetric0
    """
     
    # define stuff
    t, q, psi = w
    ABsolution, theta_0 = p
    A, B, C, D = ABsolution(t)
    #equation 32 in the mathematics document. It was easier to just define once as Q and not retype every time.
    Q = (((C + 2*D)*np.exp(-2*A - 4*B)*math.pow(theta_0,2))+((C-D)*np.exp(-2*A + 2*B)*(1 - math.pow(theta_0,2))))
    
    # create f = [t', q', psi',]
    f = [-(1+z)/Q, (np.exp(-2*A + 2*B))/Q, (np.exp(-4*A - 2*B))/(Q * np.power(1 + z, 2))]
    
    return f
    

def helperfunctions(p):
    """
    Solves the ODE relating t, q, psi. 

    Arguments:
        p :  vector of the parameters
                  p = [ABsolution, theta_0]
    """
    
    
    continuousOutput = True; # whether or not to output a continuous solution
    vectorizedInput = True; # defines whether or not the solver is handling a system or an individual ODE
    zRange = [0, 2] # defines the range of z values to solve over
    initVal = [0, 0, 0] # defines initial values for t, q, psi
    
    # calculate the solution. Returns a Bunch object with many fields.
    solution = solve_ivp(vectorfield, zRange, initVal, dense_output = continuousOutput, vectorized = vectorizedInput, args = (p,))
    
    sol = solution.sol # the continuous solution of the differential equation as instance of OdeSolution
    redshifts = solution.t # the redshifts that were considered
    
    return [sol, redshifts]

def main():    
    
    #[O_m, O_r, O_L, O_k, O_B, B0]
    p = [0.5, 0.02, 0.3, 0.079375, 0.1, 0.025]
    solution = ABvacmetric0(p)
    
    equation = solution[0]
    
    # sol, redshifts = helperfunctions([equation, 0.5])
    
    fig, ax = plt.subplots(3, 1, sharex = True)
    
    cthvals=[0.01, 0.1, 0.2, 0.5, 1]
    for cth in reversed(cthvals):
        sol, redshifts = helperfunctions([equation, cth])
        redshifts = np.linspace(0.01,2.,100)
        t, q, psi = sol(redshifts)
        ax[0].plot(redshifts, t, label=cth)
        ax[1].plot(redshifts, q)
        ax[2].plot(redshifts, psi)
        
    ax[0].set_ylabel('tau_e')
    ax[1].set_ylabel('q_o')
    ax[2].set_ylabel('psi_o')
    ax[2].set_xlabel('z')
    ax[2].set_yscale('linear')    
    fig.legend(loc='outside upper right')
    plt.show()
    
    print(sol(.5))
    
if(__name__ == "__main__"):
    main()

    