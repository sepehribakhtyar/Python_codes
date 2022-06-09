# PLS1 by NIPALS 
# Description: NIPALS algorithm for PLS1 regression (Ytrain is univariate) by Dr. B. Sepehri from university of Kurdistan.
# Details: The NIPALS algorithm is the originally proposed algorithm for PLS. Here, the y-data are only allowed to be univariate. This simplifies the algorithm.
# References: K. Varmuza and P. Filzmoser: Introduction to Multivariate Statistical Analysis in Chemometrics. CRC Press, Boca Raton, FL, 2009.
##########################################################################################################################################################################
# input arguments:
# Xtrain: Independent variables for train set (m*n matrix). Variables are in columns and objects (molecules) are in rows. 
# Ytrain: Dependent variable (a column vector) for train set (m*1 matrix)
# Xtest: Independent variables for test set (r*n matrix)
# Ytest: Dependent variable (a column vector) for train set (r*1matrix)
# NPC: number of PLS components
# it: number of iterations
# tol: tolerance for convergence
#####################################################################################################################################
# Calculated Value:
# P: matrix with loadings for X
# T: matrix with scores for X
# W: weights for X
# C: weights for Y
# b: final regression coefficients
# R2_train: Squared correlation coeffiecient for train set
# R2_test: Squared correlation coeffiecient for test set
# RMSE_train: Root mean squared errors for train set
# Ytrain_pred: Predicted property (Dependent variable) for train set
# Ytest_pred: Predicted property (Dependent variable) for test set
############################################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command:b, R2_train,RMSE_train, R2_test, RMSE_test, Ytrain_pred, Ytest_pred, T, P, W, C=PLSR1_NIPALS(Xtrain, Ytrain, Xtest, Ytest, NPC, it = 1000, tol = 1e-08)
############################################################################################################################################################
def PLSR1_NIPALS(Xtrain, Ytrain, Xtest, Ytest, NPC, it = 1000, tol = 1e-08):
    import numpy as np
    DIM_Xtrain=np.shape(Xtrain)
    Rows_Xtrain=DIM_Xtrain[0]
    Columns_Xtrain=DIM_Xtrain[1]
    T=np.zeros((Rows_Xtrain,NPC)) 
    P=np.zeros((Columns_Xtrain,NPC))
    W=np.zeros((Columns_Xtrain,NPC))
    C=np.zeros((NPC,1))
    for i in range(0,NPC):
        w=Xtrain.T@Ytrain
        w=w/np.linalg.norm(w)
        t=Xtrain@w
        c=(Ytrain.T@t)/(t.T@t)
        p=(Xtrain.T@t)/(t.T@t)
        Xtrain=Xtrain - (t @ p.T)
        Ytrain=Ytrain-(t*c)
        T[:,[i]]=t
        P[:,[i]]=p
        W[:,[i]]=w
        C[[i]]=c
    b=W@np.linalg.inv(P.T@W)@C
    DIM_test=np.shape(Xtest)
    Rows_test=DIM_test[0]
    Ones_test=np.ones((Rows_test,1))
    Ones_train=np.ones((Rows_Xtrain,1))
    Ytrain_pred=Xtrain@b
    Ytest_pred=Xtest@b
    Ytrain_bar=np.mean(Ytrain)*Ones_train
    w=Ytrain-Ytrain_bar
    SST_train=np.transpose(w)@w
    A=Ytrain_pred-Ytrain_bar
    SSR_train=np.transpose(A)@A
    R2_train=SSR_train/SST_train
    Res_train=Ytrain-Ytrain_pred
    RMSE_train=np.std(Res_train)
    Res_test=Ytest-Ytest_pred
    RMSE_test=np.std(Res_test)
    Ytest_bar=np.mean(Ytest)*Ones_test
    z=Ytest-Ytest_bar
    d=Ytest-Ytest_pred
    R2_test=1-((np.transpose(d)@d)/(np.transpose(z)@z))
    return (b, R2_train,RMSE_train, R2_test, RMSE_test, Ytrain_pred, Ytest_pred, T, P, W, C) 













