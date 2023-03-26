import ABvacmetric0
import numpy as np
import math
import helperfunctions

c = 299724.58 # speed of light in km/s
 
def distMod(z, nvec, paramList):
    
    """
    Calculates the predicted distance modulus 

    Arguments:
        z :  float redshift
        nvec :  numpy array of floats, direction of sky position
        paramList :  vector of the parameters
                  paramList = [O_m, O_r, O_L, O_k, O_B, B0, h, n0vec]
    """
    
    # unpack parameters
    O_m, O_r, O_L, O_k, O_B, B0, h, n0vec = paramList
    
    # calculate ABvacmetric0 for these parameters
    solution, contractFlag, times = ABvacmetric0([O_m, O_r, O_L, O_k, O_B, B0])
    
    # find cos(theta) between nvec and n0vec
    c_theta = abs(np.dot(nvec, n0vec))
    
    # find helper solution values for the value of z and c_theta we have
    helperfuncs, redshifts = helperfunctions([solution, c_theta])
    teval, q0val, psi0val = helperfuncs(z)
    
    # extract values for A and B at teval
    A, B, C, D = solution(teval)
    
    # return the distance modulus
    # Note: the sinc function has an extra pi in the input to transform from the normalized sinc function to the unnormalized sinc function, sinc(x) = sin(x) / x
    return -2.5 * math.log(((np.exp((2 * A) + (10 * B))) * math.pow((h * math.pow(10, -3)) / c, 2)) / ((math.pow((math.pow(c_theta, 2) + ((np.exp(6 * B)) * (1 - math.pow(c_theta, 2))))), 2.5) * q0val * psi0val * np.sinc((q0val / np.pi) * (1 - math.sqrt(-3 * O_k * math.pow(c_theta, 2))))), 10)
    