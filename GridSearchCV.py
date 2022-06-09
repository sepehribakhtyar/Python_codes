# GridSearchCV
# Exhaustive search over specified parameter values for an estimator.
# sklearn.model_selection.GridSearchCV
# sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False
# The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.
###############################################################################
# More information: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
###############################################################################
# Ref:D. Paper, Hands-on Scikit-Learn for Machine Learning Applications: Data Science Fundamentals with Python, Apress, 2020.
##################################################################################################################################
# Example for SVC:
from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X, y)
###################################################################










