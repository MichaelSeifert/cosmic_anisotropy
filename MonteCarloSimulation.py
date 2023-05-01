from ChiSquared import chiSquared
from LoadingDataFromCSV import DataSet
import numpy as np
import pandas as pd

def MCMC(dataset, totalPoints, initPoint, run = 0, cont = False):
    
    stepSize = .00001
    angleStepSize = .001
    
    if(cont):
        file = pd.read_csv("MonteCarloRun" + str(run) + ".csv")
        df=pd.DataFrame(file.iloc[-1:,:].values)
        pointsComplete = int(df.iloc[0][0])
        initPoint = [df.iloc[0][1], df.iloc[0][2], df.iloc[0][3], df.iloc[0][4], df.iloc[0][5], df.iloc[0][6], df.iloc[0][7], [df.iloc[0][8], df.iloc[0][9], df.iloc[0][10]]]
        print(initPoint)
    else:
        pointsComplete = 1
        
    initChi2 = chiSquared(initPoint, dataset)
    currentPoint = initPoint
    currentChi2 = initChi2
    
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
        "Chi2": chi2s
        }
    
    dataframe = pd.DataFrame(pointsDict)
    if(not cont):
        dataframe.to_csv("MonteCarloRun" + str(run) + ".csv", header = True, index = False)
    
    for i in range(int(totalPoints / 100)):
        
        numberStepsIn100 = 0;
        
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
        
        for j in range(100):
            candPoint = currentPoint.copy()
            
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
                
            # Set O_k to fit the other parameters
            candPoint[3] = 1 - candPoint[0] - candPoint[1] - candPoint[2] - candPoint[4] - np.power(candPoint[5], 2)
            
            currentDir = candPoint[7].copy()
            n0vec = [0, 0, 0]
            n0vec[0] = (currentDir[0] + ((2 * angleStepSize * np.random.rand(1)[0]) - (1 * angleStepSize)))
            n0vec[1] = (currentDir[1] + ((2 * angleStepSize * np.random.rand(1)[0]) - (1 * angleStepSize)))
            n0vec[2] = (currentDir[2] + ((2 * angleStepSize * np.random.rand(1)[0]) - (1 * angleStepSize)))
            n0vec[0] = (n0vec[0] / np.linalg.norm(np.array(currentDir)))
            n0vec[1] = (n0vec[1] / np.linalg.norm(np.array(currentDir)))
            n0vec[2] = (n0vec[2] / np.linalg.norm(np.array(currentDir)))
            candPoint[7] = n0vec.copy()
            
            candChi2 = chiSquared(candPoint, dataset)
            
            if(candChi2 <= currentChi2 or np.random.rand(1)[0] < np.exp(currentChi2 - candChi2)):
                #print(f"step: {100*i + j}")
                currentPoint = candPoint
                currentChi2 = candChi2
                numberStepsIn100 = numberStepsIn100 + 1
                
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
                "Chi2": chi2s
            }
    
        dataframe = pd.DataFrame(pointsDict)
        dataframe.to_csv("MonteCarloRun" + str(run) + ".csv", header = False, index = False, mode = 'a')
            
        # optimal acceptance is .234 I guess?
        """if(numberStepsIn100 / 100.0 < .23): # steps too big
            stepSize = stepSize / 2
        elif(stepSize < .0001): # steps too small, but not getting too big either
            stepSize = stepSize * 2"""
        #print(numberStepsIn100 / 100.0)
        
            
    
    
def main():
    
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
    
    O_k = 1 - O_m - O_r - O_L - O_B - np.power(b0, 2)
        
    startPoint = [O_m, O_r, O_L, O_k, O_B, b0, h, n0vec]
    
    print(startPoint)
    
    dataset = DataSet("SimulatedData.csv")
    
    MCMC(dataset, 100, startPoint, run=0, cont=True)
    
if(__name__ == "__main__"):
    main()
