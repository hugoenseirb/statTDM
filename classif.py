###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd

from sklearn.linear_model import LogisticRegression
###############################################################################

def OLS(X, y):

    # estimation des coefficients
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    # prédictions et risque empirique
    y_pred = X @ beta
    R = np.mean((y - y_pred) ** 2)

    return R, beta


def Ridge(X, y, lamb):
  
    p1 = X.shape[1]
    I = np.eye(p1)

    beta = np.linalg.inv(X.T @ X + lamb * I) @ (X.T @ y)

    y_pred = X @ beta
    R = np.mean((y - y_pred) ** 2)

    return R, beta

def Logistic(X, y):
    # Régression logistique
    logreg = LogisticRegression(penalty="l2", C=1.0) # selon Chat GPT la norme l2 permet d'utiliser la norme euclidienne.
    logreg.fit(X[:,1:], y)
    b0_log = logreg.intercept_[0]
    b1_log, b2_log = logreg.coef_[0]
    print("LogisticRegression : intercept =", b0_log, "coef =", (b1_log, b2_log))
    return b0_log, b1_log, b2_log

################################################################################
# PARAMETERS
################################################################################
# Dimension and sample size
p=2
n=600
# Proportion of sample from classes 0, 1, and outliers
p0 = 3/6
p1 = 2/6
pout = 1/6
# Examples of means/covariances of classes 0, 1 and outliers
mu0 = np.array([-2,-2])
mu1 = np.array([2,2])
muout = np.array([-8,-8])
Sigma_ex1 = np.eye(p)
Sigma_ex2 = np.array([[5, 0.1],
                      [1, 0.5]])
Sigma_ex3 = np.array([[0.5, 1],
                      [1, 5]])
Sigma0 = Sigma_ex1
Sigma1 = Sigma_ex1
Sigmaout = Sigma_ex1
# Regularization coefficient
lamb = 0
################################################################################

################################################################################
# DATA/LABELS GENERATION
################################################################################
# Sample sizes
n0 = int(np.floor(n*p0))
n1 = int(np.floor(n*p1))
nout = int(np.floor(n*pout))*10
n = n0 + n1 + nout

if n0+n1+nout < n:
   n0 += int(n - (n0+n1+nout))
# Data and labels
mu0_mat = mu0.reshape((p,1))@np.ones((1,n0))
mu1_mat = mu1.reshape((p,1))@np.ones((1,n1))
x0 = np.zeros((p,n0+nout))
x0[:,0:n0] = mu0_mat + la.sqrtm(Sigma0)@rnd.randn(p,n0)
x1 = mu1_mat + la.sqrtm(Sigma1)@rnd.randn(p,n1)
if nout > 0:
  muout_mat = muout.reshape((p,1))@np.ones((1,nout))
  x0[:,n0:n0+nout] = muout_mat + la.sqrtm(Sigmaout)@rnd.randn(p,nout)
y = np.concatenate((-np.ones(n0+nout),np.ones(n1)))
X = np.ones((n,p+1))
for i in np.arange(n):
     X[0:n0+nout,1:p+1] = x0.T
     X[n0+nout:n,1:p+1] = x1.T
################################################################################
# Apprentissage des classifieurs OLS / Ridge
R_ols, beta_ols = OLS(X,y) 
R_ridge, beta_ridge = Ridge(X,y,lamb) 

print("OLS : R =", R_ols, "beta =", beta_ols)
print("Ridge : R =", R_ridge, "beta =", beta_ridge)

b0_log, b1_log, b2_log = Logistic(X, y)

################################################################################
# PLOTS
################################################################################
fig, ax = plt.subplots()
ax.plot(x0[0,:], x0[1,:], 'xb')
ax.plot(x1[0,:], x1[1,:], 'xr')

# Bornes minimales et maximales en x
xmin_classe0 = x0[0, :].min()
xmin_classe1 = x1[0, :].min()
xmax_classe0 = x0[0, :].max()
xmax_classe1 = x1[0, :].max()

# Borne globale (min des deux classes, max des deux classes)
xmin = min(xmin_classe0, xmin_classe1)
xmax = max(xmax_classe0, xmax_classe1)

# Discrétisation de l’axe x pour tracer les droites
xx = np.linspace(xmin, xmax, 100)

# droite OLS
b0, b1, b2 = beta_ols
ax.plot(xx, -(b0 + b1*xx)/b2, 'g-')

# droite ridge
b0r, b1r, b2r = beta_ridge
ax.plot(xx, -(b0r + b1r*xx)/b2r, 'm--')

# droite logistique
yy_log = -(b0_log + b1_log*xx) / b2_log
ax.plot(xx, yy_log, 'k-.')

plt.show()
