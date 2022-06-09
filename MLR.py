# This program builds Multiple Linear Regression (MLR).
# This program has been written by Dr. Bakhtyar sepehri.
# Program has been written based on following refernces:
# Daryl S. Paulson, Handbook of regression and modeling : applications for the clinical and
# pharmaceutical industries,  Taylor & Francis Group, LLC, 2007.
# Applied Linear Regression, Wiley series in probability and statistics,2005.
#################################################################################################################################################
# In Xtrain and Xtest matrices, objects are in rows and independent variables are in columns.
# Xtrain is independent vriable(s) matrix for train set and Xtest is independent variabl(s) matrix for test set.
# Ytrain is dependent variable vector (a clumn vector) for train set and Ytest is dependent variable vector (a column vector) for test set.
########################################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: R2_train,RMSE_train,b,sb,t,F,R2_test,RMSE_test,Ytrain_pred,Ytest_pred=MLR(Xtrain,Xtest,Ytrain,Ytest)
####################################################################################################################################################
def MLR(Xtrain,Xtest,Ytrain,Ytest):
    import numpy as np
    DIM_train=np.shape(Xtrain)
    Rows_train=DIM_train[0]
    Columns_train=DIM_train[1]
    Ones_train=np.ones((Rows_train,1))
    Xtrain_new=np.hstack((Ones_train,Xtrain))
    b=np.linalg.inv(np.transpose(Xtrain_new)@Xtrain_new)@np.transpose(Xtrain_new)@Ytrain
    DIM_test=np.shape(Xtest)
    Rows_test=DIM_test[0]
    Ones_test=np.ones((Rows_test,1))
    Xtest_new=np.hstack((Ones_test,Xtest))
    Ytrain_pred=Xtrain_new@b
    Ytest_pred=Xtest_new@b
    Ytrain_bar=np.mean(Ytrain)*Ones_train
    w=Ytrain-Ytrain_bar
    SST_train=np.transpose(w)@w
    A=Ytrain_pred-Ytrain_bar
    SSR_train=np.transpose(A)@A
    R2_train=SSR_train/SST_train
    Res_train=Ytrain-Ytrain_pred
    RMSE_train=np.std(Res_train)
    F=((Rows_train-Columns_train-1)*R2_train)/(Columns_train*(1-R2_train))
    se=np.sqrt((np.sum((Ytrain-Ytrain_pred)**2))/(Rows_train-len(b)))
    sb=se*(np.sqrt(np.diag(np.linalg.inv(np.transpose(Xtrain_new) @ Xtrain_new))))
    sb=sb.reshape(len(sb),1)
    t=np.divide(b,sb)
    Res_test=Ytest-Ytest_pred
    RMSE_test=np.std(Res_test)
    Ytest_bar=np.mean(Ytest)*Ones_test
    z=Ytest-Ytest_bar
    d=Ytest-Ytest_pred
    R2_test=1-((np.transpose(d)@d)/(np.transpose(z)@z))
    return R2_train,RMSE_train,b,sb,t,F,R2_test,RMSE_test,Ytrain_pred,Ytest_pred









