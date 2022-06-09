# Randomized_Search_CV
# Randomized search on hyper parameters.
# sklearn.model_selection.RandomizedSearchCV
# The parameters of the estimator used to apply these methods are optimized by cross-validated search over parameter settings.
# In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.
# sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)
#######################################################################################################################
# More information: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
##########################################################################################################################
# Ref:D. Paper, Hands-on Scikit-Learn for Machine Learning Applications: Data Science Fundamentals with Python, Apress, 2020.
##################################################################################################################################
# Example for LogisticRegression:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,random_state=0)
distributions = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'])
clf = RandomizedSearchCV(logistic, distributions, random_state=0)
search.best_params_
####################################################################################################









