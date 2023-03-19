from DistanceModulus import distMod
from LoadingDataFromCSV import DataSet
import math

def chiSquared(params, dataset):
    
    """
    Calculates chi squared based on available data

    Arguments:
        params :  vector of the parameters
                  params = [O_m, O_r, O_L, O_k, O_B, B0, h, n0vec]
        dataset :  dataset object as loaded in from csv
    """

    # accumulator for the running sum
    chisq = 0
    
    # go through the data and sum up chi squared
    for i in range(dataset.zdata.size):
        chisq += (math.pow(dataset.distmoddata[i] - distMod(dataset.zdata[i], dataset.nangledata[i], params), 2) / math.pow(dataset.errordata[i], 2))
        
    return chisq
