#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:34:40 2024

@author: mseifer1
"""

from ChiSquared import chiSquared
from DistanceModulus import distMod
from LoadingDataFromCSV import DataSet
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    dataset = DataSet("Simulated_Data_10k.csv")
    calculatedDistMods = distMod(dataset.zdata, dataset.nangledata, 
                                 [0.5, 0.02, 0.3, 0.079375, 0.1, 0.025, 1, [2/3, 2/3, -1/3]])
    print(calculatedDistMods)
    
    '''
    Code below plots calculated DMs from "true" DMs imported from no-noise 
    data set.  Used for debugging purposes.
    '''
    trueDataSet = DataSet("Simulated_data_no_noise_10k.csv")
    plt.scatter(trueDataSet.distmoddata,
                abs(calculatedDistMods - trueDataSet.distmoddata),
                s=3)
    plt.xlabel("True dist. mod.")
    plt.ylabel("Error in dist. mod.")
    plt.yscale('log')
    
    print(chiSquared([0.5, 0.02, 0.3, 0.079375, 0.1, 0.025, 1, [2/3, 2/3, -1/3]], 
                     dataset))
    
if(__name__ == "__main__"):
    main()