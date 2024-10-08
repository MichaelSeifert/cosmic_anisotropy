import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ABvacmetric0 import ABvacmetric0
import pde 

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
    tau, q, psi = w
    ABsolution, theta_0 = p
    A, B, C, D = ABsolution(tau)
    #equation 32 in the mathematics document. It was easier to just define once as Q and not retype every time.
    Q = (((C + 2*D)*np.exp(-2*A - 4*B)*math.pow(theta_0,2))+((C-D)*np.exp(-2*A + 2*B)*(1 - math.pow(theta_0,2))))
    
    # create f = [t', q', psi',]
    f = [-(1+z)/Q, (np.exp(-2*A + 2*B))/Q, (np.exp(-4*A - 2*B))/(Q * np.power(1 + z, 2))]
    
    return f

def qfunction(cth, ABsolution, z):
    
    A, B, C, D = ABsolution(z)
    Q = (((C + 2*D)*np.exp(-2*A - 4*B)*math.pow(cth,2))+
         ((C-D)*np.exp(-2*A + 2*B)*(1 - math.pow(cth,2))))
    
    return Q

class helperfunctionPDE(pde.PDEBase):

    def __init__(self, cosparams=None): #this will fail if parameters aren't defined
        super().__init__()
        
        
        self.cosparams = cosparams

    def evolution_rate(self, state, z=0): 
        # Need to set explicit_time_dependence flag -- where?
        # NEED TO GET ctheta in here somehow
    
        """
        Defines the differential equations for ABvacmetric0 derived from Eq 9b and 9c
    
        Arguments:
            w :  vector of the state variables:
                      w = [t, q, psi]
            z :  red shift
        """
         
        # extract coordinates from grid
        # see https://github.com/zwicker-group/py-pde/discussions/46
        assert state.grid.dim == 1  # implementation only works for 1d problems
        ctheta = state.grid.cell_coords[:, 0]  # this extracts the x-coordinates of all cells
        
        # define stuff
        tau, q, psi = state
        # ABsolution, theta_0 = p
        ABsolution, flag, times = ABvacmetric0(self.cosparams) # This is currently borken
        A, B, C, D = ABsolution(tau.data)
        #equation 32 in the mathematics document. It was easier to just define once as Q and not retype every time.
        Q = (((C + 2*D)*np.exp(-2*A - 4*B)*np.power(ctheta,2))+((C-D)*np.exp(-2*A + 2*B)*(1 - np.power(ctheta,2))))
        
        # create f = [t', q', psi']
        # derivs = np.array([-(1+z)/Q, (np.exp(-2*A + 2*B))/Q, (np.exp(-4*A - 2*B))/(Q * np.power(1 + z, 2))])
        tau_z =pde.ScalarField(state.grid, -(1+z)/Q)        
        q_z = pde.ScalarField(state.grid, (np.exp(-2*A + 2*B))/Q)
        psi_z = pde.ScalarField(state.grid, (np.exp(-4*A - 2*B))/(Q * np.power(1 + z, 2)))
        
        return pde.FieldCollection([tau_z, q_z, psi_z])


def helperfunctions(ABsolution, targetzvals=[]):
    """
    Solves the ODE relating t, q, psi. 

    Arguments:
        ABsolution: ODE solution object from ABvacmetric0
        targetzvals : sorted array of desired z values for calculation of 
        helper functions.  Default:  None.  NOT YET SUPPORTED
    """

    if len(targetzvals) == 0:
        continuousOutput = True  
        maxz = 2
        minz = 0
        # If no specific z values requested, continuous solution returned
    else:
        continuousOutput = False
        maxz = targetzvals[-1]
        minz = 0
        # If specific z values requested, no continuous solution needed


    grid = pde.CartesianGrid([[0,1]], 20, periodic=False)
    tau = pde.ScalarField(grid)
    q = pde.ScalarField(grid)
    psi = pde.ScalarField(grid)
    # The line below returns a "not enough values to unpack" error
    A, B, C, D = ABsolution
    
    eq = PDE(
        {
            "tau": "-(1+t)/qfunction(x,ABsolution)",
            "q": "(np.exp(-2*A(t) + 2*B(t)))/qfunction(x,ABsolution)",
            "psi": "np.exp(-4*A(t) - 2*B(t))/(qfunction(x,ABsolution) * np.power(1 + t, 2))"
            }
        )
    
    print('So far so good!')

    


# =============================================================================
# def oldhelperfunctions(p, targetzvals=[]):
#     """
#     Solves the ODE relating t, q, psi. 
# 
#     Arguments:
#         p :  vector of the parameters
#                   p = [ABsolution, theta_0]
#         targetzvals : sorted array of desired z values for calculation of 
#         helper functions.  Default:  None
#     """
#     
#     if len(targetzvals) == 0:
#         continuousOutput = True  
#         maxz = 2
#         minz = 0
#         # If no specific z values requested, continuous solution returned
#     else:
#         continuousOutput = False
#         maxz = targetzvals[-1]
#         minz = 0
#         # If specific z values requested, no continuous solution needed
#         
#     vectorizedInput = True; 
#     # defines whether or not the solver is handling a system or an individual ODE
#     
#     zRange = [minz, maxz] # defines the range of z values to solve over
#     
#     initVal = [0, 0, 0] # defines initial values for t, q, psi
#     
#     # calculate the solution. Returns a Bunch object with many fields.
#     solution = solve_ivp(vectorfield, zRange, initVal, 
#                          # method = 'RK45',
#                          dense_output = continuousOutput, 
#                          vectorized = vectorizedInput, 
#                          t_eval = targetzvals,
#                          args = (p,),
#                          atol = 1e-9
#                          )
#     
#     sol = solution.sol  # the continuous solution of the differential equation 
#                         # as instance of OdeSolution
#     funcvals = solution.y # Explicitly calculated function values
#     redshifts = solution.t # the redshifts that were considered.  If 
#                            # targetzvals was provided, this should be the same.
#     
#     return [sol, redshifts, funcvals]
# 
# =============================================================================

def main():    
    
    p = [0.5,0.02,0.3,0.079375,0.1,0.025]
    grid = pde.CartesianGrid([[0,1]], 20, periodic=False)
    state = pde.FieldCollection.from_scalar_expressions(grid, ["0","0","0"])
    
    eq = helperfunctionPDE(cosparams=p)  # define the pde
    result = eq.solve(state, t_range=10, dt=0.01)
    result.plot()
    
if(__name__ == "__main__"):
    main()

    