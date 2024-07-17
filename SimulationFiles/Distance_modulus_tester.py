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
import warnings
from timeit import default_timer as timer

warnings.filterwarnings("ignore")

def main():
    
    dataset = DataSet("Simulated_Data_no_noise_10k.csv")
    
    start = timer()
    calculatedDistMods = distMod(dataset.zdata, dataset.nangledata, 
                                 [0.5, 0.02, 0.3, 0.079375, 0.1, 0.025, 1, [2/3, 2/3, -1/3]])
    end = timer()
    print("Elapsed time: ", end - start, "s")

    # print(calculatedDistMods)
    print("Sources with delta = NaN: ", len(np.argwhere(np.isnan(calculatedDistMods))))
    print("Sources with delta = inf: ", len(np.argwhere(np.isinf(calculatedDistMods))))
    
    
    '''
    Code below plots calculated DMs from "true" DMs imported from no-noise 
    data set.  Used for debugging purposes.
    '''
    trueDataSet = DataSet("Simulated_data_no_noise_10k.csv")
    errors = calculatedDistMods - trueDataSet.distmoddata

    # Create scatter plot of error in dist. mod    
    # plt.scatter(trueDataSet.distmoddata,
    #             abs(errors),
    #             s=3)
    # plt.xlabel("True dist. mod.")
    # plt.ylabel("Error in dist. mod.")
    # plt.yscale('log')
    
    # Retrieve information for nbad worst sources
    nbad = 50
    badsourceinds = np.argpartition(abs(errors),-nbad)[-nbad:]
    badsourceinds = badsourceinds[np.argsort(errors[badsourceinds])]
    print("Worst errors in dist. mod.:\n", errors[badsourceinds])
    
    badsourcelocs = dataset.nangledata[badsourceinds]
    
    # print(badsourcelocs)
    badxvals = badsourcelocs[:,0]
    badyvals = badsourcelocs[:,1]
    badzvals = badsourcelocs[:,2]
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(xvals,yvals,zvals)
    
    badcosinevals=[]
    for i in range(nbad):
        badcosinevals.append(np.dot(badsourcelocs[i],[2/3, 2/3, -1/3]))
    allcosinevals=[]
    for i in range(10000):
        allcosinevals.append(np.dot(dataset.nangledata[i], [2/3, 2/3, -1/3]))

    # print(badcosinevals)  
    # plt.hist(badcosinevals, bins=20)  

    # Plot all source cosines just as a sanity check
    # xvals = dataset.nangledata[:,0]
    # yvals = dataset.nangledata[:,1]
    # zvals = dataset.nangledata[:,2]
    
    allcosinevals=[]
    for i in range(10000):
        allcosinevals.append(np.dot(dataset.nangledata[i], [2/3, 2/3, -1/3]))
 
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle('All cos(theta) values vs. \'bad\' values')
    axs[0].hist(allcosinevals, bins="auto")
    axs[1].hist(badcosinevals, bins="auto")
                              
    # plt.hist(allcosinevals, bins='auto')                       

    # Plot z values of worst errors
        
    badzvals=dataset.zdata[badsourceinds]
    allzvals=dataset.zdata
    
    print("z values of worst errors:\n", badzvals)
    
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle('All z values vs. \'bad\' z values')
    axs[0].hist(allzvals, bins="auto")
    axs[1].hist(badzvals, bins="auto")
    
    print("Chi-squared: ", 
          chiSquared([0.5, 0.02, 0.3, 0.079375, 0.1, 0.025, 1, [2/3, 2/3, -1/3]], 
                      dataset))
    
if(__name__ == "__main__"):
    main()