# Performs Epsilon-Support Vector Regression with grid search for determining the best values of trainable parameters in scikit-learn
# Ytrain and Ytest should be vectors with one dimension
################################################################################
# Run grid search for finding optimum trainable parameters
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel':['rbf'],'C':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100],'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0],'epsilon':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}]
esvr = SVR()
esvr_GridSearch=GridSearchCV(estimator=esvr, param_grid=tuned_parameters, cv=5,scoring='r2')
esvr_GridSearch.fit(Xtrain,Ytrain)
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
# the best trainable parameters values
esvr_GridSearch.best_params_
# The best score
esvr_GridSearch.best_score_
#################################################################################################
# create e-SVR model by using the best value of trainable parameters
Best_esvr=SVR(kernel=, degree=3, gamma=, C=, epsilon=)
Best_esvr.fit(Xtrain,Ytrain)
# Prediction
Ytrain_pred=Best_esvr.predict(Xtrain)
Ytest_pred=Best_esvr.predict(Xtest)
# Calculating R2
from sklearn.metrics import r2_score
R2_train=r2_score(Ytrain, Ytrain_pred)
R2_test=r2_score(Ytest, Ytest_pred)
##################################################################







