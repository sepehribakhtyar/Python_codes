# Decision trees can also be applied to regression problems, using the DecisionTreeRegressor class.
# More information: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
#####################################################################################################################################################################
# Run grid search for finding optimum trainable parameters
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'ccp_alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}]
DTR = DecisionTreeRegressor()
DTR_GridSearch=GridSearchCV(estimator=DTR, param_grid=tuned_parameters, cv=5,scoring='r2')
DTR_GridSearch.fit(Xtrain,Ytrain)
##############################################################################################################3
# the best trainable parameters values
DTR_GridSearch.best_params_
# The best score
DTR_GridSearch.best_score_
###########################################################################
# create DTR model by using the best value of trainable parameters
from sklearn.tree import DecisionTreeRegressor
Best_DTR = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)
Best_DTR.fit(Xtrain,Ytrain)
# Prediction
Ytrain_pred=Best_DTR.predict(Xtrain)
Ytest_pred=Best_DTR.predict(Xtest)
# Calculating R2
from sklearn.metrics import r2_score
R2_train=r2_score(Ytrain, Ytrain_pred)
R2_test=r2_score(Ytest, Ytest_pred)
#########################################################################

