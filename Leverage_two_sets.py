# This code has been written by Dr. Bakhtyar Sepehri to calculate leverage for QSAR models.
# For more study, please see following paper:
# B. Sepehri, R. Ghavami, Design of new CD38 inhibitors based on CoMFA modelling and molecular docking analysis of 4‑amino-8-quinoline carboxamides and 2,4-diamino-8-quinazoline carboxamides, SAR QSAR Environ. Res. 30 (2019) 21-38. https://doi.org/10.1080/1062936X.2018.1545695.
###############################################################################################################################################################################################################################################################################################
# In this code:
# Xtrain is descriptor matrix for train set.
# Xtest is descriptor matrix for test set.
# In both Xtrain and Xtest matrices, samples (molecules) are in rows and variables (descriptors) are in columns. 
################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: Leverage_critical,Leverage_train,Leverage_test=Leverage_two_sets (Xtrain, Xtest)
##################################################################################################################################
def Leverage_two_sets (Xtrain, Xtest):
    import numpy as np
    DIM_train=np.shape(Xtrain)
    Rows_train=DIM_train[0]
    Columns_train=DIM_train[1]
    Leverage_critical=(3*(Columns_train+1))/Rows_train
    X=np.vstack((Xtrain,Xtest))
    H=X@np.linalg.inv(np.transpose(X)@X)@np.transpose(X)
    Leverage=np.diagonal(H)
    zz=len(Leverage)
    Leverage_train=Leverage[0:Rows_train]
    Leverage_test=Leverage[Rows_train:zz]
    return Leverage_critical,Leverage_train,Leverage_test
    











