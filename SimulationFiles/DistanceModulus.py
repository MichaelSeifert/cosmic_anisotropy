from ABvacmetric0 import ABvacmetric0
import numpy as np
import matplotlib as plt
from HelperFunctions import helperfunctions
from LoadingDataFromCSV import DataSet
from scipy.interpolate import RectBivariateSpline

c = 299724580 # speed of light in m/s
 
def distMod(z_list, nvec_list, paramList):
    
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
    
    # find abs(cos(theta)) between nvec and n0vec
    c_theta_list = abs(np.dot(nvec_list, n0vec))
    
    
    #the following section interpolates over angle values to reduce the number of calls to helperfunctions. Huge performace increase, especially for large datasets.
    #could potentially be even more efficient to solve a PDE instead of an ODE + interpolation to get helperfunctions.
    
    #set points to interpolate between
    
    # Original choice
    # c_theta_points = np.array([.000001, .00001, .0001, .001, .01, .1, 1])
    
    # Logarithmically distributed from 10^(-minintval) to 1, with a certain num.
    # of points per decade
    minintval = 6
    intptsperdecade = 5
    c_theta_points = np.exp(np.linspace(-minintval*np.log(10.),0,minintval*intptsperdecade+1))

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
    # z_points = np.arange(0, 2.0, .01)
    z_points = np.concatenate((np.arange(0, 1.8, 0.01), np.arange(1.801, 2, 0.001)))
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
    degx=3
    degy=3
    interpolate_te = RectBivariateSpline(c_theta_points, z_points, tevals, 
                                         kx=degx, ky=degy)
    interpolate_qo = RectBivariateSpline(c_theta_points, z_points, qovals, 
                                         kx=degx, ky=degy)
    interpolate_psio = RectBivariateSpline(c_theta_points, z_points, psiovals, 
                                           kx=degx, ky=degy)
            
    # stores distmods
    distmods = []
    
    # find distmod for each z and c_theta
    for i in range(len(z_list)):
        c_theta = c_theta_list[i]
        z = z_list[i]
        teval = interpolate_te(c_theta, z)[0][0]
        q0val = interpolate_qo(c_theta, z)[0][0]
        psi0val = interpolate_psio(c_theta, z)[0][0]
        A, B, C, D = solution(teval)
        # Note: the sinc function has an extra pi in the input to transform from the normalized sinc function to the unnormalized sinc function, sinc(x) = sin(x) / x
        distmod = -2.5 * np.log10(((np.exp((2 * A) + (10 * B))) * np.power(h / c, 2) / ((np.power((np.power(c_theta, 2) + ((np.exp(6 * B)) * (1 - np.power(c_theta, 2)))), 2.5) * q0val * psi0val * np.real(np.sinc((q0val / np.pi) * np.emath.sqrt(-3 * O_k * (1 - np.power(c_theta, 2)))))))))
        distmods.append(distmod)
        
    # return the distance modulus array
    return np.array(distmods)
    
def main():
    
    dataset = DataSet("SimulatedData.csv")
    calculatedDistMods = distMod(dataset.zdata, dataset.nangledata, [0.5, 0.02, 0.3, 0.079375, 0.1, 0.025, 1, [2/3, 2/3, -1/3]])
    print(calculatedDistMods)
    
    '''
    Code below plots calculated DMs from "true" DMs imported from no-noise 
    data set.  Used for debugging purposes.
    '''
    trueDataSet = DataSet("Simulated_data_no_noise.csv")
    plt.pyplot.scatter(trueDataSet.distmoddata,calculatedDistMods)
    plt.pyplot.xlabel("True dist. mod.")
    plt.pyplot.ylabel("Calc. dist. mod.")
    
if(__name__ == "__main__"):
    main()