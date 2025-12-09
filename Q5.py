import numpy as np
data3 = np.load("data3.npy") 

from OLS import OLS

#Q5
print("====== DATA 3 ======")
print("Risque Empirique pour Datya3 polynomiale", OLS(10, data3))