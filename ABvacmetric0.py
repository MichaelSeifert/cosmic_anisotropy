from scipy.integrate import solve_ivp
import numpy as np

def vectorfield(t, w, p):
    
    """
    Defines the differential equations for ABvacmetric0 derived from Eq 9b and 9c

    Arguments:
        w :  vector of the state variables:
                  w = [A,B,C,D]
        t :  proper time
        p :  vector of the parameters:
                  p = [O_m, O_r, O_L, O_k, O_B, B0]
    """
    
    # define the current equations for A B C and D, and load in parameters
    A, B, C, D = w
    O_m, O_r, O_L, O_k, O_B, B0 = p
    
    # create f = [A', B', C', D']
    f = [C,
         D,
         ]
         # waiting to check math before finishing implementing the equations
    
    return f

def hitZMax(t, w):
    
    """
    Checks if we hit zmax in all directions. Acts as helper function to solve_ivp call in ABvacmetric0 to act as an event trigger.
    """
    
    # set the largest value of redshift to integrate to
    zmax = 2
    
    # get the A and B functions
    A = w[0]
    B = w[1]
    
    # need some condition in the form of a continuous function to check zmax. It will trigger on the condition hitting 0.

def contracting(t, w):
    
    """
    Checks for contracting universes. Acts as helper function to solve_ivp call in ABvacmetric0 to act as an event trigger.
    """
    
    # get C and D aka A' and B'
    C = w[2]
    D = w[3]
    
    # stop integration if we hit this condition equal to 0
    return C(t) + 2*D(t)
    
    
def ABvacmetric0(p):
    
    """
    Solves the ODE relating A, B, C, and D, replicating the function of ABvacmetric0. Note: this solves the differential equation when called.

    Arguments:
        p :  vector of the parameters:
                  p = [O_m, O_r, O_L, O_k, O_B, B0]
    """
    
    # load B0 from input params
    B0 = p[5]
    
    #Designate events to check either as terminal or directional via monkey patch.
    contracting.terminate = True;
    hitZMax.terminate = True;
    
    # solve_ivp parameters
    continuousOutput = True; # whether or not to output a continuous solution
    vectorizedInput = True; # defines whether or not the solver is handling a system or an individual ODE
    tRange = [np.NINF, 0] # defines the range of t values to solve over
    initVal = [0, 0, 1, B0] # defines initial values for A, B, C, and D
    eventsToCheck = [contracting, hitZMax] # defines the functions to check for termination of integration. Must be of form event([A, B, C, D], t). Event triggers on event() = 0.
    
    # calculate the solution. Returns a Bunch object with many fields. See documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    solution = solve_ivp(vectorfield, tRange, initVal, dense_output = continuousOutput, vectorized = vectorizedInput, events = eventsToCheck, args = (p))
    
    sol = solution.sol # the continuous solution of the differential equation as instance of OdeSolution
    t_events = solution.t_events # tracks the occurences of event triggers during integration
    
    # if there was an event that triggered from contracting, return the solution and the 'contract flag' set to True
    if(t_events[0].length > 0):
        return [sol, True]
    
    # if we got ehre, there was no contraction, and we can return a False 'contract flag'
    return [sol, False]
    
    
    
    
    