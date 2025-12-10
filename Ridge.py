import numpy as np
import matplotlib.pyplot as plt

def Ridge(lambd, data):
    X = data[0,:]
    Y = data[1,:]
    plt.figure()
    plt.scatter(X,Y)
    
    n = X.shape[0]                 # on récup le nb de point ( 100 ici )
    colonne_de_un = np.ones((n,1)) # colonne de n 1 (chaque point aura son 1 sur sa ligne)
    X_colonne = X.reshape(-1,1)    # on passe X de ligne à colonne
    
    X_avec_un = np.hstack((colonne_de_un, X_colonne)) # on assemble la colonne de 1 avec la colonne de X
    
    I = np.eye(2) # identité 2x2 (car beta = [beta0, beta1])
    
    beta_vect = np.linalg.inv(X_avec_un.T @ X_avec_un + lambd * I) @ (X_avec_un.T @ Y) # obtenu par dérivation du pb d'opti à la Q6
    
    beta0 = beta_vect[0]
    beta1 = beta_vect[1]
    Y_prediction = X_avec_un @ beta_vect # on calcul (vecteur beta)*(matrice X)
   
    # risque empirique
    R_ridge = np.mean((Y - Y_prediction)**2) 
    
    idx = np.argsort(X)                 # selon ChatGPT pour trier X afin d'avoir un bel affichage
    plt.plot(X[idx],Y_prediction[idx])  # on superpose le modele au nuage
    plt.title("[Ridge] lambda = " +str(lambd))
    plt.show()
   
    return R_ridge , beta_vect