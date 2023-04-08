import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helperfunctions import helperfunctions
from ABvacmetric0 import ABvacmetric0

class DataSet:  # a DataSet is created to easily store the data fields in a readily accessible manner.

    file = ""  # file storing data

    #initializer
    def __init__(self, file):
        
        self.file = file # store what data file is being used just in case

        csv = pd.read_csv(file, skipinitialspace=True)  # reads the csv into a pandas dataframe

        self.zdata = np.array(csv["redshifts"])
        self.tdata = np.array(csv["teval"])
        self.qdata = np.array(csv["qoval"])
        self.psidata = np.array(csv["psioval"])
        
def main():
    dataset = DataSet("mathematicaData.csv")
    
    fig, axs = plt.subplots(3, sharex=True)
    
    p = [0.28, 0.01, 0.69, 0.01, 0.01, 0]
    solution = ABvacmetric0(p)
    
    equation = solution[0]
    
    sol, redshifts = helperfunctions([equation, .25])
    
    tedata = []
    qodata = []
    psiodata = []
    
    for z in dataset.zdata:
        te, qo, psio = sol(z)
        tedata.append(te)
        qodata.append(qo)
        psiodata.append(psio)
        
    axs[0].plot(dataset.zdata, dataset.tdata, label = "Mathematica", color = "Blue")
    axs[0].plot(dataset.zdata, tedata, label = "Python", color = "Red")
    axs[1].plot(dataset.zdata, dataset.qdata, label = "Mathematica", color = "Blue")
    axs[1].plot(dataset.zdata, qodata, label = "Python", color = "Red")
    axs[2].plot(dataset.zdata, dataset.psidata, label = "Mathematica", color = "Blue")
    axs[2].plot(dataset.zdata, psiodata, label = "Python", color = "Red")
    
    plt.legend()
    plt.show()
    
if(__name__ == "__main__"):
    main()