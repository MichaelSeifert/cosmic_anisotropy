# Written by Wyatt Carbonell on 3/10/23. Replicates the function of "Random data from simulation" in the Mathematica implementation. Requires modified data csv as included in same folder as this code.

import numpy as np
import pandas as pd

class DataSet:  # a DataSet is created to easily store the data fields in a readily accessible manner.

    file = ""  # file storing data

    zdata = np.array(float) # data column for redshift data [float]
    distmoddata = np.array(float) # data column for distance modulus [float]
    errordata = np.array(float) # data column for error [float]
    nangledata = np.array(np.array(float)) # data column for RA & Dec as unit vector [[float, float, float]]

    #initializer
    def __init__(self, file):
        
        self.file = file # store what data file is being used just in case

        csv = pd.read_csv(file, skipinitialspace=True)  # reads the csv into a pandas dataframe

        self.zdata = np.array(csv["Redshift"]) # read in redshift data
        self.distmoddata = np.array(csv["Distance Modulus"]) #  in distance modulus data
        self.errordata = np.array(csv["Error"]) # read in error data
        
        #read in each of the nangle components
        nangle1 = csv["N Angle 1"] 
        nangle2 = csv["N Angle 2"]
        nangle3 = csv["N Angle 3"]
        
        #combine the nangle components into vectors
        nanglepoints = []
        for i in range(len(nangle1)):
            nanglepoints.append(np.array([nangle1[i], nangle2[i], nangle3[i]]))
            
        self.nangledata = np.array(nanglepoints) # store the nangle data
        
# test main to print the read data
def main():
    
    simulatedData = DataSet("SimulatedData.csv")
    
    print(simulatedData.nangledata)
    
main()