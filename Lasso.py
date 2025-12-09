import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso as SkLasso

def Lasso(lambd, data):
    X = data[0,:]
    Y = data[1,:]

    plt.figure()
    plt.scatter(X, Y)

    n = X.shape[0]
    X_colonne = X.reshape(-1, 1) # X en vecteur colonne

    # def du model via le module sklearn
    model = SkLasso(alpha=lambd, fit_intercept=True)
    model.fit(X_colonne, Y)

    # prédictions
    Y_prediction = model.predict(X_colonne)

    # risque empirique
    R_lasso = np.mean((Y - Y_prediction) ** 2)

    # affichage du modèle sur le nuage
    idx = np.argsort(X)
    plt.plot(X[idx], Y_prediction[idx], color="red")
    plt.title(f"[LASSO] lambda =" +str(lambd))
    plt.show()

    return R_lasso