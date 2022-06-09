# PLSR2 and PLSR1
# PLSRegression implements the PLS 2 blocks regression known as PLS2 or PLS1 in case of one dimensional response. 
# This class inherits from _PLS with mode=”A”, deflation_mode=”regression”, norm_y_weights=False and algorithm=”nipals”.
# Matrices:
# T: x_scores_
# U: y_scores_
# W: x_weights_
# C: y_weights_
# P: x_loadings_
# Q: y_loadings_
# more information: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

import numpy as np
import sklearn
from sklearn.cross_decomposition import PLSRegression
plsr2 = PLSRegression(n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True)
plsr2.fit(X, Y)
Y_pred = plsr2.predict(X)

