# Partial Least Square SVD
# More information: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD
# Simply perform a svd on the crosscovariance matrix: X’Y There are no iterative deflation here
import numpy as np
from sklearn.cross_decomposition import PLSSVD
plsca = PLSSVD(n_components=2, scale=True, copy=True)
plsca.fit(X, Y)

X_c, Y_c = plsca.transform(X, Y)
X_c.shape, Y_c.shape





