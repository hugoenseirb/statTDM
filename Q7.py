# REPONSES AUX Q3 Q4 Q7

from OLS import OLS
from Ridge import Ridge

import numpy as np
import matplotlib.pyplot as plt

data2 = np.load("data2.npy") 
data3 = np.load("data3.npy") 

#OLS
q_vec = [2,3,4,5,6,7,8,9,10] # nb de nombre de fonctions φᵢ(x) et aussi : degré du polyome phi.
RE_vec = []

for qi in q_vec:
    R_OLS = OLS(qi, data2)
    print("[OLS] q =", qi, "R =", R_OLS)
    RE_vec.append(R_OLS)
    
plt.figure()
plt.plot(q_vec, RE_vec, marker="o", color="blue")
plt.xlabel("Degré du polynôme q")
plt.ylabel("Risque empirique R(q)")
plt.title("[OLS] Évolution du risque empirique en fonction de q")
plt.grid(True)
plt.show()

#RIDGE
lambda_vec = [0, 0.1, 1, 5, 10, 50]
R_ridge_vec = []

for lambd in lambda_vec:
    R_ridge = Ridge(lambd, data3)
    print("[Ridge] lambda = ", lambd, "R(lambda) = ", R_ridge)
    R_ridge_vec.append(R_ridge)
    
plt.figure()
plt.plot(lambda_vec, R_ridge_vec, marker="o", color="blue")
plt.xlabel("Valeurs de lambda")
plt.ylabel("Risque empirique R(q)")
plt.title("[RIDGE] Évolution du risque empirique en fonction de lambda")
plt.grid(True)
plt.show()