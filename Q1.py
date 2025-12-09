import numpy as np
import matplotlib.pyplot as plt

# load la classe
data1 = np.load("data1.npy") 

# affichage du nuage
X = data1[0,:] 
Y = data1[1,:]

plt.scatter(X,Y)

# regression linéaire OLS

n = X.shape[0]                 # on récup le nb de point ( 100 ici )
colonne_de_un = np.ones((n,1)) # colonne de n 1 (chaque point aura son 1 sur sa ligne)
X_colonne = X.reshape(-1,1)    # on passe X de ligne à colonne

X_avec_un = np.hstack((colonne_de_un, X_colonne)) # on assemble la colonne de 1 avec la colonne de X

beta_vect = np.linalg.inv(X_avec_un.T @ X_avec_un) @ (X_avec_un.T @ Y) # calcul des coefficients OLS : bêta = (X^T X)^(-1) X^T Y
beta0 = beta_vect[0]
beta1 = beta_vect[1]
print("beta0 =", beta0)
print("beta1 =", beta1)

Y_prediction = X_avec_un @ beta_vect            # calcule la prediction yi = b0 + b1xi pour tous les points i

R = np.mean((Y - Y_prediction)**2)              # risque empirique (écart entre Y fournis et Y prédits)
print("Risque empirique : R(beta0, beta) =", R) # on affiche le risque empirique pr modèle OLS

idx = np.argsort(X)       # selon ChatGPT pour trier X afin d'avoir un bel affichage
droite = beta0 + beta1*X  # expression de la droite (droite = vecteur de yi)
plt.plot(X[idx], droite[idx], color="red") 
plt.show()