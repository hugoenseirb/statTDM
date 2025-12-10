import numpy as np
import matplotlib.pyplot as plt

def OLS(q, data):
    
    X = data[0,:]
    Y = data[1,:]
    
    plt.figure()
    plt.scatter(X,Y)
    
    n = X.shape[0]                 # on récup le nb de point ( 100 ici )
    colonne_de_un = np.ones((n,1)) # colonne de n 1 (chaque point aura son 1 sur sa ligne)
    X_colonne = X.reshape(-1,1)    # on passe X de ligne à colonne

    X_phi = np.hstack([X_colonne**k for k in range(1, q + 1)])

    X_phi_avec_un = np.hstack((colonne_de_un, X_phi)) # on assemble la colonne de 1 avec la colonne de X

    beta_vect = np.linalg.inv(X_phi_avec_un.T @ X_phi_avec_un) @ (X_phi_avec_un.T @ Y) # calcul des coefficients : bêta = (X^T X)^(-1) X^T Y

    Y_prediction = X_phi_avec_un @ beta_vect # on calcul (vecteur beta)*(matrice X)

    # risque empirique
    R_OLS = np.mean((Y - Y_prediction)**2)

    idx = np.argsort(X)       # selon ChatGPT pour trier X afin d'avoir un bel affichage
    plt.plot(X[idx], Y_prediction[idx], color="red")  # affichage de la droite sur le nuage
    plt.title("[OLS] q = " + str(q))
    plt.show()
    
    return R_OLS , beta_vect