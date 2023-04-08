from ABvacmetric0 import ABvacmetric0
import numpy as np
from helperfunctions import helperfunctions
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
    
    # find cos(theta) between nvec and n0vec
    c_theta_list = abs(np.dot(nvec_list, n0vec))
    
    c_theta_points = np.array([.000001, .00001, .0001, .001, .01, .1, 1])
    z_points = np.arange(0, 2.0, .01)
    
    trows = []
    qrows = []
    psirows = []
    
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
    
    interpolate_te = RectBivariateSpline(c_theta_points, z_points, tevals)
    interpolate_qo = RectBivariateSpline(c_theta_points, z_points, qovals)
    interpolate_psio = RectBivariateSpline(c_theta_points, z_points, psiovals)
            
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
    print(distMod([dataset.zdata[0]], [dataset.nangledata[0]], [0.28, 0.01, 0.69, 0.01, 0.01, 0, 0.7, [-0.561726, -0.73281, -0.383997]]))
    
if(__name__ == "__main__"):
    main()