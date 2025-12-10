###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554, parser='auto')
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 
###############################################################################
# DISPLAY A SAMPLE
###############################################################################
m=16
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  #lt.title("Label = %i" % lb[i])


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###############################################################################
# NORMALISATION DES DONNÉES (pr converger + vite)
###############################################################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

###############################################################################
# RÉGRESSION LOGISTIQUE
###############################################################################
log_reg = LogisticRegression(penalty="l2", multi_class="multinomial", solver="lbfgs", max_iter=1000)

log_reg.fit(X_train_scaled, y_train)

# Prédictions sur le jeu de test
y_pred = log_reg.predict(X_test_scaled)

# matrice de confusion (permet de voir avec quel nombre notre nombre à été confondu)
mat_conf = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=mat_conf, display_labels=np.arange(10))  # chiffres 0 à 9

plt.figure(figsize=(8,8))
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Matrice de confusion, Régression logistique (MNIST)")
plt.show()

##################################################################
# Q2 – AFFICHAGE DES COEFFICIENTS BETA COMME IMAGES
#################################################################

betas = log_reg.coef_

plt.figure(figsize=(10, 10))

for k in range(10):
    ax = plt.subplot(5, 2, k + 1)
    plt.imshow(betas[k].reshape(28, 28), cmap="RdBu")
    plt.title(f"Coefficients beta pour la classe {k}")
    plt.colorbar()
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Visualisation des coefficients beta, Régression logistique")
plt.tight_layout()
plt.show()

###############################################################################
# RÉGRESSION LOGISTIQUE LASSO + VISUALISATION DES COEFF
##############################################################################
from sklearn.linear_model import LogisticRegression

# On réduit la taille du jeu d'apprentissage pour que LASSO converge vite
n_sub = 10000   # par ex. 10 000 images au lieu de 52 500
X_train_sub = X_train_scaled[:n_sub]
y_train_sub = y_train[:n_sub]

lasso_log_reg = LogisticRegression(
    penalty="l1",
    solver="saga", # nécessaire pour l1
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    n_jobs=-1, # utilise tous les cœurs dispo
    verbose=1  # affiche la progression dans le terminal
)

print("Entraînement du modèle LASSO sur", n_sub, "images...")
lasso_log_reg.fit(X_train_sub, y_train_sub)
print("Entraînement terminé.")

# Coefficients β (10 classes × 784 pixels)
betas_lasso = lasso_log_reg.coef_

plt.figure(figsize=(10, 10))
for k in range(10):
    ax = plt.subplot(5, 2, k + 1)
    plt.imshow(betas_lasso[k].reshape(28, 28), cmap="RdBu")
    plt.title(f"[LASSO] Coefficients beta pour classe {k}")
    plt.colorbar()
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Visualisation des coefficients beta, regr logistique", fontsize=16)
plt.tight_layout()
plt.show()