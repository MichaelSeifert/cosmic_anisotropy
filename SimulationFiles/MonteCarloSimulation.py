from ChiSquared import chiSquared
from LoadingDataFromCSV import DataSet
import numpy as np
import pandas as pd

def MCMC(dataset, totalPoints, initPoint, run = 0, cont = False):
    
    """
    Runs the Monte Carlo for a set number of points

    Arguments:
        dataset :  Dataset object containing the simulated data
        totalPoints :  Target number of points to run the Monte Carlo until reaching
        initPoint :  The initial point to start the simulation at. If continuing a previous run, the last point of that run will be used
        run :  Used to distinguish Monte Carlo runs from each other
        cont :  False if starting a new Monte Carlo run, True if continuing one
    """
    
    #maximum step size values
    stepSize = .01
    angleStepSize = .05
    
    '''
    "Temperature" .  T = 1 is what we 
    want to get to eventually.  Higher T means that the MCMC chain will sample
    a larger portion of the parameter space.
    '''
    temperature = 1
    
    #If continuing, load in relevant data, else prepare for a new run
    if(cont):
        file = pd.read_csv("MonteCarloRun" + str(run) + ".csv")
        df=pd.DataFrame(file.iloc[-1:,:].values)
        pointsComplete = int(df.iloc[0][0])
        initPoint = [df.iloc[0][1], df.iloc[0][2], df.iloc[0][3], df.iloc[0][4], df.iloc[0][5], df.iloc[0][6], df.iloc[0][7], [df.iloc[0][8], df.iloc[0][9], df.iloc[0][10]]]

    else:
        pointsComplete = 0
        
    #set initial settings
    initChi2 = chiSquared(initPoint, dataset)
    currentPoint = initPoint
    currentChi2 = initChi2
    
    #prepare data lists
    indexes = [pointsComplete]
    oms = [initPoint[0]]
    ors = [initPoint[1]]
    oLs = [initPoint[2]]
    oks = [initPoint[3]]
    oBs = [initPoint[4]]
    bos = [initPoint[5]]
    hs = [initPoint[6]]
    nvec1s = [initPoint[7][0]]
    nvec2s = [initPoint[7][1]]
    nvec3s = [initPoint[7][2]]
    chi2s = [initChi2]
    stepSizes = [stepSize]
    angleStepSizes = [angleStepSize]
    
    #prepare to write out to file
    pointsDict = {
        "Index": indexes,
        "O_m": oms,
        "O_r": ors,
        "O_L": oLs,
        "O_k": oks,
        "O_B": oBs,
        "b0": bos,
        "h": hs,
        "nvec1": nvec1s,
        "nvec2": nvec2s,
        "nvec3": nvec3s,
        "Chi2": chi2s,
        "StepSize": stepSizes,
        "AngleStepSize": angleStepSizes        
        }
    
    dataframe = pd.DataFrame(pointsDict)
    
    #if not continuing a run, print the initial point as point 0
    if(not cont):
        dataframe.to_csv("MonteCarloRun" + str(run) + ".csv", header = True, index = False)
    
    #run points in batches of 100
    for i in range(int(totalPoints / 100)):
        
        #for calculating acceptance ratio
        paramAttempted = 0
        paramAccepted = 0 
        dirAttempted = 0
        dirAccepted = 0
        dirStep = False
        
        #reset the data lists
        if(pointsComplete != 1):
            indexes = []
            oms = []
            ors = []
            oLs = []
            oks = []
            oBs = []
            bos = []
            hs = []
            nvec1s = []
            nvec2s = []
            nvec3s = []
            chi2s = []
            stepSizes = []
            angleStepSizes = []
        
        #run 100 points
        for j in range(100):
            
            #prepare a candidate point
            candPoint = currentPoint.copy()
            
            if(np.random.randint(2) == 0): # Step in parameters
            
                dirStep = False
                paramAttempted = paramAttempted + 1
            
                #perform a step in parameters
                candPoint[0] = candPoint[0] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
                candPoint[1] = candPoint[1] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
                candPoint[2] = candPoint[2] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
                candPoint[4] = candPoint[4] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
                candPoint[5] = candPoint[5] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
                candPoint[6] = candPoint[6] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
            
                #make sure values are physical
                if(candPoint[0] < 0):
                    candPoint[0] = 0
                if(candPoint[1] < 0):
                    candPoint[1] = 0   
                if(candPoint[4] < 0):
                    candPoint[4] = 0
                
                # Set O_k to fit the other parameters
                candPoint[3] = 1 - candPoint[0] - candPoint[1] - candPoint[2] - candPoint[4] - np.power(candPoint[5], 2)
                
            else: # Step in direction 
            
                # Set flag
                dirStep = True
                dirAttempted = dirAttempted + 1
            
                #perform a step in direction, and renormalize
                currentDir = candPoint[7].copy()
                n0vec = [0, 0, 0]
                n0vec[0] = (currentDir[0] + ((2 * angleStepSize * np.random.rand(1)[0]) - (1 * angleStepSize)))
                n0vec[1] = (currentDir[1] + ((2 * angleStepSize * np.random.rand(1)[0]) - (1 * angleStepSize)))
                n0vec[2] = (currentDir[2] + ((2 * angleStepSize * np.random.rand(1)[0]) - (1 * angleStepSize)))
                tempnorm = np.linalg.norm(np.array(n0vec))
                n0vec[0] = (n0vec[0] / tempnorm)
                n0vec[1] = (n0vec[1] / tempnorm)
                n0vec[2] = (n0vec[2] / tempnorm)
                candPoint[7] = n0vec.copy()

                '''                
                # choose new candidate direction randomly over unit sphere
                n0vec = np.random.normal(0.0,1.0,3)
                tempnorm = np.linalg.norm(np.array(n0vec))
                n0vec[0] = (n0vec[0] / tempnorm)
                n0vec[1] = (n0vec[1] / tempnorm)
                n0vec[2] = (n0vec[2] / tempnorm)
                candPoint[7] = n0vec.copy()
                '''                
 
           
            #recalculate chi squared for the step
            candChi2 = chiSquared(candPoint, dataset)
            
            #perform the step if chi sqaured decreases or probabilistically
            if(candChi2 <= currentChi2 or np.random.rand(1)[0] < np.exp((currentChi2 - candChi2)/temperature)):
                #print(f"step: {100*i + j}")
                currentPoint = candPoint
                currentChi2 = candChi2
                if(dirStep):
                    dirAccepted = dirAccepted + 1
                else:
                    paramAccepted = paramAccepted + 1
                
            #save the point
            pointsComplete += 1
            indexes.append(pointsComplete)
            oms.append(currentPoint[0])
            ors.append(currentPoint[1])
            oLs.append(currentPoint[2])
            oks.append(currentPoint[3])
            oBs.append(currentPoint[4])
            bos.append(currentPoint[5])
            hs.append(currentPoint[6])
            nvec1s.append(currentPoint[7][0])
            nvec2s.append(currentPoint[7][1])
            nvec3s.append(currentPoint[7][2])
            chi2s.append(currentChi2)
            stepSizes.append(stepSize)
            angleStepSizes.append(angleStepSize)
            
        #write all the points to file
        pointsDict = {
            "Index": indexes,
            "O_m": oms,
            "O_r": ors,
            "O_L": oLs,
            "O_k": oks,
            "O_B": oBs,
            "b0": bos,
            "h": hs,
            "nvec1": nvec1s,
            "nvec2": nvec2s,
            "nvec3": nvec3s,
            "Chi2": chi2s,
            "StepSize": stepSizes,
            "AngleStepSize": angleStepSizes        
        }
    
        dataframe = pd.DataFrame(pointsDict)
        dataframe.to_csv("MonteCarloRun" + str(run) + ".csv", header = False, index = False, mode = 'a')
        
        print("Step:", (i+1)*100)
        print("Chi-squared:", currentChi2)

        # Adjust parameter step size if needed
        print("Param. acceptance:", paramAccepted, "/", paramAttempted, "=", paramAccepted/paramAttempted)
        if(paramAccepted/paramAttempted < .20):
            stepSize = stepSize / 2
        elif(paramAccepted/paramAttempted > .25):
            stepSize = stepSize * 2
        print("New param. step size:", stepSize)
            
        # Adjust parameter step size if needed
        print("Ang. acceptance:", dirAccepted, "/", dirAttempted, "=", dirAccepted/dirAttempted )
        if(dirAccepted/dirAttempted < .20):
            angleStepSize = angleStepSize / 2
        elif(dirAccepted/dirAttempted > .25):
            angleStepSize = angleStepSize * 2
        print("New ang. step size:", angleStepSize)
            
        print()
                
        
        

'''            
        #we never did get acceptance ratio based step size changes working
        # optimal acceptance is .234 I guess?
        if(numberStepsIn100 / 100.0 < .20): # steps too big
            stepSize = stepSize / 2
            angleStepSize = angleStepSize / 2
        elif(numberStepsIn100 / 100.0 > .25): # steps too small
            stepSize = stepSize * 2
            angleStepSize = angleStepSize * 2
        print(numberStepsIn100 / 100.0)
'''        
            
    
    
def main():
    
    '''
    dir1, dir2, dir3 = np.random.rand(3)
    currentDir = [dir1, dir2, dir3]
    n0vec = [dir1 / np.linalg.norm(np.array(currentDir)), dir2 / np.linalg.norm(np.array(currentDir)), dir3 / np.linalg.norm(np.array(currentDir))]
    
    O_m = np.random.normal(.3, .01, 1)[0]
    while O_m < 0:
        O_m = np.random.normal(.3, .01, 1)[0]
        
    O_r = np.random.normal(0, .01, 1)[0]
    while O_r < 0:
        O_r = np.random.normal(0, .01, 1)[0]
        
    O_B = np.random.normal(0, .01, 1)[0]
    while O_B < 0:
        O_B = np.random.normal(0, .01, 1)[0]
        
    O_L = np.random.normal(.7, .01, 1)[0] 
    
    b0 = np.random.normal(0, .01, 1)[0]
    while b0 < -3 or b0 > 3:
        b0 = np.random.normal(0, .01, 1)[0]
        
    #h between .6 and .8
    h = .2 * np.random.rand(1)[0] + 0.6
    '''
    
    # True parameter values from simulated data
    n0vec = [2/3, 2/3, -1/3]
    O_m = 0.5
    O_r = 0.02
    O_B = 0.1
    O_L = 0.3
    b0 = 0.025
    h = 1
    
    O_k = 1 - O_m - O_r - O_L - O_B - np.power(b0, 2)
        
    startPoint = [O_m, O_r, O_L, O_k, O_B, b0, h, n0vec]
    
    print(startPoint)
    
    dataset = DataSet("SimulatedData.csv")
    
    MCMC(dataset, 1e5, startPoint, run=0, cont=False)
    
    print()
    
if(__name__ == "__main__"):
    main()
