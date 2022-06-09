# This script performs random forest for regression.
# Ytrain and Ytest should be vectors with one dimension.
# More information: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
# More information: https://scikit-learn.org/stable/modules/ensemble.html#forest
###################################################################################################################################################################
# Run grid search for finding optimum trainable parameters
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'n_estimators':[20,50,100,150,200],'ccp_alpha': [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}]
RFR=RandomForestRegressor()
RFR_GridSearch=GridSearchCV(estimator=RFR, param_grid=tuned_parameters, cv=5,scoring='r2')
RFR_GridSearch.fit(Xtrain,Ytrain)
##################################################################################################################################################################
# the best trainable parameters values
RFR_GridSearch.best_params_
# The best score
RFR_GridSearch.best_score_
##############################################################################################################
# create RFR model by using the best value of trainable parameters
Best_RFR=RandomForestRegressor(n_estimators=, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=, max_samples=None)
Best_RFR.fit(Xtrain,Ytrain)
# Prediction
Ytrain_pred=Best_RFR.predict(Xtrain)
Ytest_pred=Best_RFR.predict(Xtest)
# Calculating R2
from sklearn.metrics import r2_score
R2_train=r2_score(Ytrain, Ytrain_pred)
R2_test=r2_score(Ytest, Ytest_pred)
#########################################################################
