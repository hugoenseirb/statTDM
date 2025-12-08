import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("data1.npy") 
data2 = np.load("data2.npy") 

X = data1[:,0]
Y = data1[:,1]
plt.scatter(X,Y)
plt.show()