# This script runs k-nearst neighbours for regression process
###################################################################
# Run grid search for finding optimum trainable parameters
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}]
kNN_regression= KNeighborsRegressor(weights='distance', algorithm='auto')
kNN_GridSearch=GridSearchCV(estimator=kNN_regression, param_grid=tuned_parameters, cv=5,scoring='r2')
kNN_GridSearch.fit(Xtrain,Ytrain)
###########################################################################################################################################################
# the best trainable parameters values
kNN_GridSearch.best_params_
# The best score
kNN_GridSearch.best_score_
#################################################################################################
# create e-SVR model by using the best value of trainable parameters
Best_kNN_regression=KNeighborsRegressor(n_neighbors=,weights='distance', algorithm='auto')
Best_kNN_regression.fit(Xtrain,Ytrain)
# Prediction
Ytrain_pred=Best_kNN_regression.predict(Xtrain)
Ytest_pred=Best_kNN_regression.predict(Xtest)
# Calculating R2
from sklearn.metrics import r2_score
R2_train=r2_score(Ytrain, Ytrain_pred)
R2_test=r2_score(Ytest, Ytest_pred)
##################################################################









