# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:10:34 2023

@author: wyatt
"""

from MonteCarloSimulation import MCMC
from LoadingDataFromCSV import DataSet
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run one MCMC simulation')
parser.add_argument("numSteps",
                    help='the number of potential steps this MCMC should take', type = int)
parser.add_argument("run",
                    help='which run number this is (for file output naming)', type = int)
parser.add_argument("dataset", 
                    help = "file name of the simulated dataset")
parser.add_argument("continue", 
                    help="continuing an existing run?", type = bool)
args = parser.parse_args()

def runOneMCMC():
    
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
    
    dataset = DataSet(args.dataset)
    
    MCMC(dataset, args.numSteps, startPoint, args.run)
    
def main():
    runOneMCMC()
    
if(__name__ == "__main__"):
    main()