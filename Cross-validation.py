# Cross-validation
# More information: https://scikit-learn.org/stable/modules/cross_validation.html
# This procedure perform k-fold cross validation.
##########################################################################################
# Example for SVM:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) 
###################################
# Cross validation on SVM
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores
######################################################
# By default, the score computed at each CV iteration is the score method of the estimator. It is possible to change this by using the scoring parameter:
from sklearn import metrics
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
scores
#####################################################
help(sklearn.model_selection.cross_val_score)
#####################################################
cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv='warn', n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score='raise-deprecating')
###########################################################
help(sklearn.model_selection.cross_validate)
# To run cross-validation on multiple metrics and also to return train scores, fit times and score times.
####################################################################################
help(sklearn.model_selection.cross_val_predic)
#  Get predictions from each split of cross-validation for diagnostic purposes. 
########################################################################
help(sklearn.metrics.make_scorer)
# Make a scorer from a performance metric or loss function.
##############################################################################3
  






