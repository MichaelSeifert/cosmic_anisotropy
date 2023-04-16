from ChiSquared import chiSquared
from LoadingDataFromCSV import DataSet
from GeneratePoints import RandomPoints
import numpy as np
import pandas as pd

def MCMC(dataset, totalPoints, initPoint, run = 0):
    
    initChi2 = chiSquared(initPoint, dataset)
    
    stepSize = .000001
    
    currentPoint = initPoint
    currentChi2 = initChi2
    
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
    
    for i in range(int(totalPoints / 100)):
        
        numberStepsIn100 = 0;
        
        for i in range(100):
            candPoint = currentPoint
            
            candPoint[0] = candPoint[0] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
            candPoint[1] = candPoint[1] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
            candPoint[2] = candPoint[2] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
            candPoint[4] = candPoint[4] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
            candPoint[5] = candPoint[5] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
            candPoint[6] = candPoint[6] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize))
        
            if(candPoint[0] < 0):
                candPoint[0] = 0
                
            if(candPoint[1] < 0):
                candPoint[1] = 0
                
            if(candPoint[4] < 0):
                candPoint[4] = 0
                
            for j in range(7):
                if(candPoint[j] > 1):
                    candPoint[j] = 1
                
            # Set O_k to fit the other parameters
            candPoint[3] = 1 - candPoint[0] - candPoint[1] - candPoint[2] - candPoint[4] - np.power(candPoint[5], 2)
            
            currentDir = candPoint[7]
            n0vec = [0, 0, 0]
            n0vec[0] = (currentDir[0] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize)))
            n0vec[1] = (currentDir[1] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize)))
            n0vec[2] = (currentDir[2] + ((2 * stepSize * np.random.rand(1)[0]) - (1 * stepSize)))
            n0vec[0] = (currentDir[0] / np.linalg.norm(np.array(currentDir)))
            n0vec[1] = (currentDir[1] / np.linalg.norm(np.array(currentDir)))
            n0vec[2] = (currentDir[2] / np.linalg.norm(np.array(currentDir)))
            
            candChi2 = chiSquared(candPoint, dataset)
            
            if(candChi2 <= currentChi2 or np.random.rand(1)[0] < np.exp(currentChi2 - candChi2)):
                currentPoint = candPoint
                currentChi2 = candChi2
                numberStepsIn100 = numberStepsIn100 + 1
                
            oms.append(currentPoint[0])
            ors.append(currentPoint[0])
            oLs.append(currentPoint[0])
            oks.append(currentPoint[0])
            oBs.append(currentPoint[0])
            bos.append(currentPoint[0])
            hs.append(currentPoint[0])
            nvec1s.append(currentPoint[0])
            nvec2s.append(currentPoint[0])
            nvec3s.append(currentPoint[0])
            chi2s.append(currentChi2)
            
        # optimal acceptance is .234 I guess?
        print(numberStepsIn100 / 100.0)
        
        if(numberStepsIn100 / 100.0 < .23): # steps too big
            stepSize = stepSize / 2
        else: # steps too small
            stepSize = stepSize * 2
            
    pointsDict = {
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
        "Chi2": chi2s
        }
    
    dataframe = pd.DataFrame(pointsDict)
    dataframe.to_csv("MonteCarloRun" + str(run) + ".csv")
            
    
    
def main():
    
    randomPoints = RandomPoints()
    startPoint = randomPoints.getPoint()
    print(startPoint)
    
    dataset = DataSet("SimulatedData.csv")
    
    MCMC(dataset, 1000, startPoint)
    
if(__name__ == "__main__"):
    main()
