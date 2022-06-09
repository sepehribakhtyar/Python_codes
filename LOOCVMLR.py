# This program has been written by Dr. bakhtyar sepehri. 
# This function runs leave-one-out (LOO) cross validation test. 
#############################################################################
# Ytrain is dependent variable vector (a column vector).
# Xtrain is independent variables (desciptors) matrix for train set.
# m is the number of objects and n is the number of variables.
# In Xtrain matrix, objects are in rows and independent variables are in columns.
#########################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: R2_cv,RMSE_cv=LOOCVMLR(Xtrain,Ytrain)
#############################################################################################
def LOOCVMLR(Xtrain,Ytrain):
    import numpy as np
    DIM_train=np.shape(Xtrain)
    Rows_train=DIM_train[0]
    Ones_train=np.ones(((Rows_train),1))
    Ones=np.ones(((Rows_train-1),1))
    Ytrain_pred_cv=np.zeros((Rows_train,1))
    one=np.ones((1,1))
    for i in range(0,Rows_train):
        Y=Ytrain
        X=Xtrain
        RX=X[[i],:]
        X=np.delete(X,i,axis=0)
        Y=np.delete(Y,i,axis=0)
        X=np.hstack((Ones,X))
        b=np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@Y
        RX=np.hstack((one,RX))
        Ytrain_pred_cv[i]=RX@b
    Ytrain_bar=np.mean(Ytrain)*Ones_train
    w=Ytrain-Ytrain_bar
    SST_train=np.transpose(w)@w
    A=Ytrain_pred_cv-Ytrain_bar
    SSR_train=np.transpose(A)@A
    R2_cv=SSR_train/SST_train
    Res_train=Ytrain-Ytrain_pred_cv
    RMSE_cv=np.std(Res_train)
    return R2_cv,RMSE_cv

















