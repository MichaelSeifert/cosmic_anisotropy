import random
import numpy as np

class RandomPoints:
    
    currentIndex = 0
    
    def __init__(self):
        self.generatePoints()
        
    def generatePoints(self):
        self.O_m = np.random.rand(10000)
        self.O_r = np.random.rand(10000)
        self.O_B = np.random.rand(10000)
        self.b0 = (2 * np.random.rand(10000)) - 1
        self.O_L = (2 * np.random.rand(10000)) - 1
        self.O_k = 1 - self.O_m - self.O_r - self.O_L - self.O_B - np.power(self.b0, 2)
        self.dir1 = np.random.rand(10000)
        self.dir2 = np.random.rand(10000)
        self.dir3 = np.random.rand(10000)
        self.h = np.random.rand(10000) + 0.5
        
    #gets the next point, generating more if needed, and making sure nvec is normalized.
    def getPoint(self):
        
        currentDir = [self.dir1[self.currentIndex], self.dir2[self.currentIndex], self.dir3[self.currentIndex]]
        n0vec = [0, 0, 0]
        n0vec[0] = (currentDir[0] / np.linalg.norm(np.array(currentDir)))
        n0vec[1] = (currentDir[1] / np.linalg.norm(np.array(currentDir)))
        n0vec[2] = (currentDir[2] / np.linalg.norm(np.array(currentDir)))
        point = [self.O_m[self.currentIndex], self.O_r[self.currentIndex], self.O_L[self.currentIndex], self.O_k[self.currentIndex], self.O_B[self.currentIndex], self.b0[self.currentIndex], self.h[self.currentIndex], n0vec]
        
        self.currentIndex = self.currentIndex + 1
        
        if(self.currentIndex == 10000):
            self.currentIndex = 0
            self.generatePoints()
        
        return point
    

def main():
    points = RandomPoints()
    
if(__name__ == "__main__"):
    main()