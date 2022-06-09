# This script has been written by Dr. B. Sepehri for performing mean centeringand auto-scaling operation on matrices including Xtrain (m*n matrix), Ytrain (m*1 matrix), Xtest (z*n matrix) and Ytest (z*1 matrix).
# This program has been written by Dr. Bakhtyar Sepehri.
# Program has been written based on following refernce:
# K. Varmuza, P. Filzmoser, Introduction to multivariate Statistical analysis in chemometrics, Taylor & Francis Group, LLC, 2009.
# In Xtrain and Xtest matrices, objects are in rows and variables are in columns.
# In Ytrain and Ytest vectors, objects are in rows.
########################################################################################################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command:autoscaled_Xtrain,meancentered_Xtrain,autoscaled_Xtest,meancentered_Xtest,autoscaled_Ytrain,meancentered_Ytrain,autoscaled_Ytest,meancentered_Ytest=Pre_processing (Xtrain,Xtest,Ytrain,Ytest)
########################################################################################################################################################################################################################
def Pre_processing (Xtrain,Xtest,Ytrain,Ytest):
    import numpy as np
    # Xtrain Mean centering calculations
    DIM_Xtrain=np.shape(Xtrain)
    Rows_Xtrain=DIM_Xtrain[0]
    Columns_Xtrain=DIM_Xtrain[1]
    mXtrain=np.mean(Xtrain,axis=0).reshape(1,Columns_Xtrain)
    meanmatrix_Xtrain=np.repeat(mXtrain,Rows_Xtrain,axis=0)
    meancentered_Xtrain=Xtrain-meanmatrix_Xtrain 
    std_Xtrain=np.std(Xtrain,axis=0).reshape(1,Columns_Xtrain)
    autoscaled_Xtrain=np.zeros((Rows_Xtrain,Columns_Xtrain))
    for i in range (0,Columns_Xtrain):
        autoscaled_Xtrain[:,[i]]=meancentered_Xtrain[:,[i]]/std_Xtrain[0,[i]]               
    # Xtest Mean centering calculations
    DIM_Xtest=np.shape(Xtest)
    Rows_Xtest=DIM_Xtest[0]
    Columns_Xtest=DIM_Xtest[1]
    mXtest=np.mean(Xtest,axis=0).reshape(1,Columns_Xtest)
    meanmatrix_Xtest=np.repeat(mXtest,Rows_Xtest,axis=0)
    meancentered_Xtest=Xtest-meanmatrix_Xtest  
    std_Xtest=np.std(Xtest,axis=0).reshape(1,Columns_Xtest)
    autoscaled_Xtest=np.zeros((Rows_Xtest,Columns_Xtest))
    for i in range (0,Columns_Xtest):
        autoscaled_Xtest[:,[i]]=meancentered_Xtest[:,[i]]/std_Xtest[0,[i]]               
    # Ytrain Mean centering calculations
    DIM_Ytrain=np.shape(Ytrain)
    Rows_Ytrain=DIM_Ytrain[0]
    mYtrain=np.mean(Ytrain,axis=0).reshape(1,1) 
    meanmatrix_Ytrain=np.repeat(mYtrain,Rows_Ytrain,axis=0)
    meancentered_Ytrain=Ytrain-meanmatrix_Ytrain 
    autoscaled_Ytrain=meancentered_Ytrain/np.std(Ytrain)
    # Ytest Mean centering calculations
    DIM_Ytest=np.shape(Ytest)
    Rows_Ytest=DIM_Ytest[0]
    mYtest=np.mean(Ytest,axis=0).reshape(1,1) 
    meanmatrix_Ytest=np.repeat(mYtest,Rows_Ytest,axis=0)
    meancentered_Ytest=Ytest-meanmatrix_Ytest  
    autoscaled_Ytest=meancentered_Ytest/np.std(Ytest)
    return autoscaled_Xtrain,meancentered_Xtrain,autoscaled_Xtest,meancentered_Xtest,autoscaled_Ytrain,meancentered_Ytrain,autoscaled_Ytest,meancentered_Ytest


