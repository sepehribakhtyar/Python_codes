# This program has been written by Dr. bakhtyar sepehri. 
# This function runs leave-one-out (LOO) cross validation test on PLSR1 model. 
# Ytrain is dependent variable vector (a column vector).
# Xtrain is independent variables (desciptors) matrix for train set.
# objects(molecules) must be in rows.
# In Xtrain matrix, objects are in rows and independent variables are in columns.
######################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: R2_cv,RMSE_cv=PLSR1_NIPALS_LOOCV(Xtrain,Ytrain,NPC, it = 1000, tol = 1e-08)
#####################################################################    
def PLSR1_NIPALS_LOOCV(Xtrain,Ytrain, NPC, it = 1000, tol = 1e-08):
    import numpy as np
    DIM_Xtrain=np.shape(Xtrain)
    Rows_Xtrain=DIM_Xtrain[0]
    Columns_Xtrain=DIM_Xtrain[1]
    Ytrain_pred_cv=np.zeros((Rows_Xtrain,1))
    for i in range (0,Rows_Xtrain):
        Y=Ytrain
        X=Xtrain
        RX=X[[i],:]
        X=np.delete(X,i,axis=0)
        Y=np.delete(Y,i,axis=0)
        T=np.zeros(((Rows_Xtrain-1),NPC)) 
        P=np.zeros((Columns_Xtrain,NPC))
        W=np.zeros((Columns_Xtrain,NPC))
        C=np.zeros((NPC,1))
        for i in range(0,NPC):
            w=X.T@Y
            w=w/np.linalg.norm(w)
            t=X@w
            c=(Y.T@t)/(t.T@t)
            p=(X.T@t)/(t.T@t)
            X=X-(t @ p.T)
            Y=Y-(t*c)
            T[:,[i]]=t
            P[:,[i]]=p
            W[:,[i]]=w
            C[[i]]=c
        b=W@np.linalg.inv(P.T@W)@C
        Ytrain_pred_cv[i]=RX@b
    Ones_train=np.ones((Rows_Xtrain,1))
    Ytrain_bar=np.mean(Ytrain)*Ones_train
    w=Ytrain-Ytrain_bar
    SST_train_cv=np.transpose(w)@w
    A= Ytrain_pred_cv-Ytrain_bar
    SSR_train_cv=np.transpose(A)@A
    R2_cv=SSR_train_cv/SST_train_cv
    Res_train_cv= Ytrain_pred_cv-Ytrain
    RMSE_cv=np.std(Res_train_cv)
    return (R2_cv,RMSE_cv)




