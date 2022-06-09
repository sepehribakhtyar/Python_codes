# This code has been written by Dr. Bakhtyar Sepehri to investigate the validation of QSAR models by calculating Q2F1, Q2F2, Q2F3 and Concordance Correlation Coefficient (CCC).
# For more study, please see following paper:
# P. Gramatica, A. Sangion, A Historical Excursus on the Statistical Validation Parameters for QSAR Models: A Clarification Concerning Metrics and Terminology, J. Chem. Inf. Model. 56(2016)1127-1131.
############################################################################################################################################################################################################
# In this code:
# Ytrain is vector of dependent variable values for training set;
# Ytrain_pred is vector of predicted dependent variable values by created model for training set;
# Ytest is vector of dependent variable values for test set;
# Ytest_pred is vector of predicted dependent variable values by created model for test set;
# Ytrain, Ytrain_pred, Ytest and Ytest_pred vectors should have two dimensions (Use reshape function in numpy).
#############################################################################################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: Q2F1,Q2F2,Q2F3=General_validation(Ytrain, Ytrain_pred, Ytest, Ytest_pred)
#############################################################################################################################################################################################################
def General_validation(Ytrain, Ytrain_pred, Ytest, Ytest_pred):
     import numpy as np
     m=len(Ytrain)
     z=len(Ytest)
     Q2F1=1-(np.sum((Ytest_pred-Ytest)**2)/(np.sum((Ytest-(np.mean(Ytrain_pred)*np.ones((z,1))))**2)))
     Q2F2=1-(np.sum((Ytest_pred-Ytest)**2)/(np.sum((Ytest-(np.mean(Ytest_pred)*np.ones((z,1))))**2)))
     Q2F3=1-((np.sum((Ytest_pred-Ytest)**2)/z)/((np.sum((Ytrain-(np.mean(Ytrain_pred)*np.ones((m,1))))**2)/m)))
     return Q2F1,Q2F2,Q2F3













