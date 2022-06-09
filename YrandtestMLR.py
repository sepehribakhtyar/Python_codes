# This program performs Y-randomization test on  Multiple Linear Regression (MLR) model.
# This program has been written by  Dr. Bakhtyar Sepehri. 
######################################################################################################
# repeat is the number of iteartion.
# Ytrain is dependent variable vector (a column vector).
# Xtrain is independent variable matrix for train set.
# In Xtrain,objets are in rows and independent variables are in columns.
#################################################################################3
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: R2_max=YrandtestMLR(Ytrain,Xtrain,repeats)
#############################################################################################################
def YrandtestMLR(Ytrain,Xtrain,repeats):
    import numpy as np
    m=len(Ytrain)
    R2train=np.zeros((repeats,1))
    Ones_train=np.ones((m,1))
    Xtrain_new=np.hstack((Ones_train,Xtrain))
    for i in range(0,repeats):
        np.random.shuffle(Ytrain)
        b=np.linalg.inv(np.transpose(Xtrain_new)@Xtrain_new)@np.transpose(Xtrain_new)@Ytrain
        Ytrain_pred=Xtrain_new@b
        Ytrain_bar=np.mean(Ytrain)*Ones_train
        w=Ytrain-Ytrain_bar
        SST_train=np.transpose(w)@w
        A=Ytrain_pred-Ytrain_bar
        SSR_train=np.transpose(A)@A
        R2train[i]=SSR_train/SST_train
    max_R2train=np.max(R2train)
    return max_R2train           

     
            
    
    
    























