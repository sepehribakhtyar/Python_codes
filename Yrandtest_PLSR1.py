# This program performs Y-randomization test on  PLSR1 model.
# This program has been written by  Dr. Bakhtyar Sepehri. 
# objects(molecules) must be in rows.
# repeat is the number of iteartion.
# Ytrain is dependent variable vector.
# Xtrain is independent variable matrix (desciptor matrix for train set.).
# In Xtrain,molecules are in rows and descriptors are in columns.
####################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: R2_max=Yrandtest_PLSR1(Ytrain,Xtrain,repeats,NPC, it = 1000, tol = 1e-08)
##############################################################################################################
def Yrandtest_PLSR1(Ytrain,Xtrain,repeats,NPC, it = 1000, tol = 1e-08):
    import numpy as np
    R2_train=np.zeros((repeats,1))
    RMSE_train=np.zeros((repeats,1))
    for i in range(0,repeats):
        np.random.shuffle(Ytrain)
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
        Ones_train=np.ones((Rows_Xtrain,1))
        Ytrain_pred=Xtrain@b
        Ytrain_bar=np.mean(Ytrain)*Ones_train
        w=Ytrain-Ytrain_bar
        SST_train=np.transpose(w)@w
        A=Ytrain_pred-Ytrain_bar
        SSR_train=np.transpose(A)@A
        R2_train[i]=SSR_train/SST_train
    R2_max=np.max(R2_train)
    return (R2_max)






