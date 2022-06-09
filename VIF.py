# This program has been written by Dr. Bakhtyar Sepehri.
# this function calculates Variance Inflation Factor (VIF) between descriptors that is a criteria of multi-collinearity between descriptors. If there is no corellation between a descriptor with others, VIF is 1 and maximum acceptable VIF is 10.
###########################################################################################################################################################################################################################################################
# Xtrain is independent variables matrix for train set.
# In Xtrain, variables are in columns and objects are in rowes.
###################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: VIF=VIF(Xtrain)
#####################################################################################################################################################
def VIF(Xtrain):
    import numpy as np
    DIM_train=np.shape(Xtrain)
    Rows_train=DIM_train[0]
    Columns_train=DIM_train[1]
    Ones_train=np.ones((Rows_train,1))
    RXmatpred=np.zeros([Rows_train,Columns_train])
    RXmatpredbar=np.zeros([Rows_train,Columns_train])
    R2=np.zeros(Columns_train)
    VIF=np.zeros(Columns_train)
    for i in range(0,Columns_train):
        X=Xtrain
        RX=X[:,[i]]
        X=np.delete(X,i,axis=1)
        X=np.hstack((Ones_train,X))
        b=np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@RX
        RXmatpred[:,[i]]=X@b
        RXmatpredbar[:,[i]]=(np.ones(Rows_train)*np.mean(RXmatpred[:,[i]])).reshape(Rows_train,1)
    for i in range(0,Columns_train):
        w=Xtrain[:,[i]]-(RXmatpredbar[:,[i]])
        SST=np.transpose(w)@w
        A=RXmatpred[:,[i]]-(RXmatpredbar[:,[i]])
        SSR=np.transpose(A)@A
        R2[i]=SSR/SST
        VIF[i]=1/(1-R2[i])
    return VIF







