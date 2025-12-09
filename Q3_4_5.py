
import numpy as np
import matplotlib.pyplot as plt

def affiche_RE_et_graph(q, data):
    # load la classe
    data2 = np.load(data) 
    X = data2[0,:]
    Y = data2[1,:]
    
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
    R = np.mean((Y - Y_prediction)**2)
    print("Risque empirique pour q =", q, ":", R)

    idx = np.argsort(X)       # selon ChatGPT pour trier X afin d'avoir un bel affichage
    plt.plot(X[idx], Y_prediction[idx], color="red")  # affichage de la droite sur le nuage
    plt.title("q = " + str(q))
    plt.show()
    return R
    

q_vec = [2,3,4,5,6,7,8,9,10] # nb de nombre de fonctions φᵢ(x) et aussi : degré du polyome phi.
# on affiche le plot et le risque empirique pour chaque degré de phi.
RE_vec = []
for qi in q_vec:
    print("degré du pol phi : q = ",qi)
    RE = affiche_RE_et_graph(qi, "data2.npy")
    print("Risque empirique : R = ", RE)
    RE_vec.append(RE)
    
plt.figure()
plt.plot(q_vec, RE_vec, marker="o", color="blue")
plt.xlabel("Degré du polynôme q")
plt.ylabel("Risque empirique R(q)")
plt.title("Évolution du risque empirique en fonction de q")
plt.grid(True)
plt.show()

#Q5
print("====== DATA 3 ======")
RE = affiche_RE_et_graph(qi, "data3.npy")