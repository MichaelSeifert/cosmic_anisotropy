from DistanceModulus import distMod
from LoadingDataFromCSV import DataSet
import numpy as np
import matplotlib.pyplot as plt

def chiSquared(params, dataset):
    
    """
    Calculates chi squared based on available data

    Arguments:
        params :  vector of the parameters
                  params = [O_m, O_r, O_L, O_k, O_B, B0, h, n0vec]
        dataset :  dataset object as loaded in from csv
    """
    # calculate chi squared
    chi2 = np.power(dataset.distmoddata - distMod(dataset.zdata, dataset.nangledata, params), 2) / np.power(dataset.errordata, 2)
    
    #creates a graph of chi squared versus sky position. We were checking for any direction bias
    """
    plt.scatter(np.dot(dataset.nangledata, params[7]), chi2)
    plt.yscale("log")
    plt.xscale("log")
    """
    
    return np.sum(chi2)

def main():
    results = []
    for i in range(100):
        result = chiSquared([0.28, 0.01, 0.69, 0.01, 0.01, 0, 0.7, [-0.623956, -0.444153, 0.642967]], DataSet("SimulatedData.csv"))
        results.append(result)
    print(results)

if(__name__ == '__main__'):
    main()
