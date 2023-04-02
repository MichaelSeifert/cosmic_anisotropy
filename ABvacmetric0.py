import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def vectorfield(t, w, p):
    
    """
    Defines the differential equations for ABvacmetric0 derived from Eq 9b and 9c

    Arguments:
        w :  vector of the state variables:
                  w = [A,B,C,D]
        t :  proper time
        p :  vector of the parameters
                  p = [O_m, O_r, O_L, O_k, O_B, B0]
    """
    
    # define the current equations for A B C and D, and load in parameters
    A, B, C, D = w
    O_m, O_r, O_L, O_k, O_B, B0 = p
    
    # create f = [A', B', C', D']
    f = [C,
         D,
         -(O_r / (2 * np.exp(4 * A))) + ((3 / 2) * O_L) + (O_k / (2 * np.exp(2 * (A - B)))) - (O_B / (2 * np.exp(4 * (A - B)))) - ((3 / 2) * math.pow(C, 2)) - ((3 / 2) * math.pow(D, 2)),
         -(3 * C * D) - (O_k / (np.exp(2 * (A - B)))) - (2 * O_B / (np.exp(4 * (A - B))))]
         
    
    return f

def hitZMax(t, w, p):
    
    """
    Checks if we hit zmax in any direction. Acts as helper function to solve_ivp call in ABvacmetric0 to act as an event trigger.
    """
    
    # set the largest value of redshift to integrate to
    zmax = 2
    
    # get the A and B functions
    A = w[0]
    B = w[1]
    
    # we want to stop when either direction hits zmax, so that our domain is valid everywhere
    return min(np.exp(-(A + 2 * B)) - 1 - (zmax + .1), np.exp(-(A - B)) - 1 - (zmax + .1))

def contracting(t, w, p):
    
    """
    Checks for contracting universes. Acts as helper function to solve_ivp call in ABvacmetric0 to act as an event trigger.
    """
    
    # get C and D aka A' and B'
    C = w[2]
    D = w[3]
    
    # stop integration if we hit this condition equal to 0
    return C + 2*D
    
    
def ABvacmetric0(p):
    
    """
    Solves the ODE relating A, B, C, and D, replicating the function of ABvacmetric0. Note: this solves the differential equation when called.

    Arguments:
        p :  vector of the parameters
                  p = [O_m, O_r, O_L, O_k, O_B, B0]
    """
    
    # load B0 from input params
    B0 = p[5]
    
    #Designate events to check either as terminal or directional via monkey patch.
    contracting.terminal = True;
    hitZMax.terminal = True;
    
    # solve_ivp parameters
    continuousOutput = True; # whether or not to output a continuous solution
    vectorizedInput = True; # defines whether or not the solver is handling a system or an individual ODE
    tRange = [0, np.NINF] # defines the range of t values to solve over
    initVal = [0, 0, 1, B0] # defines initial values for A, B, C, and D
    eventsToCheck = [contracting, hitZMax] # defines the functions to check for termination of integration. Must be of form event([A, B, C, D], t). Event triggers on event() = 0.
    
    # calculate the solution. Returns a Bunch object with many fields. See documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    solution = solve_ivp(vectorfield, tRange, initVal, dense_output = continuousOutput, vectorized = vectorizedInput, events = eventsToCheck, args = (p,))
    
    sol = solution.sol # the continuous solution of the differential equation as instance of OdeSolution
    t_events = solution.t_events # tracks the occurences of event triggers during integration
    times = solution.t # the time points that were considered
    
    # if there was an event that triggered from contracting, return the solution and the 'contract flag' set to True
    if(t_events[0].size > 0):
        return [sol, True, times]
    
    # if we got here, there was no contraction, and we can return a False 'contract flag'
    return [sol, False, times]

# this test main function calls ABvacmetric0 for a given set of parameters and graphs the resulting A and B functions
def main():
    
    fig, axs = plt.subplots(2, sharex=True)
    
    p = [.28, .01, .69, .01, .01, 0]
    
    O_m, O_r, O_L, O_k, O_B, B0 = p
    
    solution = ABvacmetric0(p)
    
    equation = solution[0]
    contract = solution[1]
    times = solution[2]
    
    tmin = equation.t_min
    tmax = equation.t_max
    
    print(tmax, tmin)
    
    tpoints = range(int(tmin * 1000), int(tmax * 1000))
    actualtpoints = []
    
    for i in range(len(tpoints)):
        actualtpoints.append(tpoints[i] / 1000)
        
    ypointsA = []
    ypointsB = []
    ypointsC = []

    for i in actualtpoints:
        point = equation(i)
        A = point[0]
        B = point[1]
        C = point[2]
        D = point[3]
        ypointsA.append(A)
        ypointsB.append(B)
        ypointsC.append((O_m/(np.exp(3*A))) + O_r/np.exp(4*A) + O_L + O_k/(np.exp(2*(A-B))) + O_B/(np.exp(4*(A-B))) - math.pow(C, 2) + math.pow(D, 2))
        
    axs[0].plot(actualtpoints, ypointsA, label = "A")
    axs[0].plot(actualtpoints, ypointsB, label = "B")
    axs[1].plot(actualtpoints, ypointsC, label = "Eq 9a")
    axs[1].scatter(x = times, y = np.zeros(len(times)))
    plt.legend()
    plt.show()
    
if(__name__ == "__main__"):
    main()
    
    
    
    
    