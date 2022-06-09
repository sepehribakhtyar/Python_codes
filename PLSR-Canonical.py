# PLSCanonical implements the 2 blocks canonical PLS of the original Wold algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].
# More information: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html#sklearn.cross_decomposition.PLSCanonical
from sklearn.cross_decomposition import PLSCanonical
plsca = PLSCanonical(n_components=2)
plsca.fit(X, Y)
X_c, Y_c = plsca.transform(X, Y)




