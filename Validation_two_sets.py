# This code has been written by Dr. Bakhtyar Sepehri to investigate the validation of QSAR models.
# For more study, please see following paper:
# P. Gramatica, A. Sangion, A Historical Excursus on the Statistical Validation Parameters for QSAR Models: A Clarification Concerning Metrics and Terminology, J. Chem. Inf. Model. 56(2016)1127-1131.
##################################################################################################################################################################################################################
# In this code:
# Ytrain is vector of dependent variable values for training set;
# Ytrain_pred is vector of predicted dependent variable values by created model for training set;
# Ytest is vector of dependent variable values for test set;
# Ytest_pred is vector of predicted dependent variable values by created model for test set;
# Ytrain, Ytrain_pred, Ytest and Ytest_pred vectors should have two dimensions (Use reshape function in numpy).
# to run code,first define Ytrain, Ytrain_pred, Ytest, Ytest_pred vectors and copy and past following argument in workspace:
# CCC2_train,r2_train,RMSE_train,k_train,k_prim_train,r2_zero_train,r_prim2_zero_train,r2_m_train,r_prim2_m_train,r2_m_mean_train,delta_r2_m_train,r2_relative_1_train,r2_relative_2_train,delta_r2_zero_train, MAE_train, CCC2_test, r2_test,RMSE_test,k_test,k_prim_test,r2_zero_test,r_prim2_zero_test,r2_m_test,r_prim2_m_test,r2_m_mean_test,delta_r2_m_test,r2_relative_1_test,r2_relative_2_test, delta_r2_zero_test, MAE_test=Validation_two_sets(Ytrain, Ytrain_pred, Ytest, Ytest_pred)
# Created model is acceptable if:
# Q2>0.5 and r2>0.6
# r2_relative<0.1
# 0.85<k and k_prim<1.15
# delta_r2_zero<0.3
# r2_m_mean>0.5
# delta_r2_m<0.2
# Good luck
#####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: CCC2_train,r2_train,RMSE_train,k_train,k_prim_train,r2_zero_train,r_prim2_zero_train,r2_m_train,r_prim2_m_train,r2_m_mean_train,delta_r2_m_train,r2_relative_1_train,r2_relative_2_train,delta_r2_zero_train, MAE_train, CCC2_test, r2_test,RMSE_test,k_test,k_prim_test,r2_zero_test,r_prim2_zero_test,r2_m_test,r_prim2_m_test,r2_m_mean_test,delta_r2_m_test,r2_relative_1_test,r2_relative_2_test, delta_r2_zero_test, MAE_test=Validation_two_sets(Ytrain, Ytrain_pred, Ytest, Ytest_pred)
#####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
def Validation_two_sets(Ytrain, Ytrain_pred, Ytest, Ytest_pred):
    import numpy as np
    m=len(Ytrain)
    z=len(Ytest)
    mean_Ytrain_vector=np.mean(Ytrain)*np.ones((m,1))
    mean_Ytest_vector=np.mean(Ytest)*np.ones((z,1))
    mean_Ytrain_pred_vector=np.mean(Ytrain_pred)*np.ones((m,1))
    mean_Ytest_pred_vector=np.mean(Ytest_pred)*np.ones((z,1)) 
    # Calculating statistical parameters for training set
    r2_train=(np.sum((Ytrain-mean_Ytrain_vector)*(Ytrain_pred-mean_Ytrain_pred_vector)))**2/((np.sum((Ytrain-mean_Ytrain_vector)**2))*(np.sum((Ytrain_pred-mean_Ytrain_pred_vector)**2)))
    residuales_train=Ytrain-Ytrain_pred
    RMSE_train=np.std(residuales_train)
    k_train=(np.sum(Ytrain*Ytrain_pred))/(np.sum((Ytrain_pred)**2))
    k_prim_train=(np.sum(Ytrain*Ytrain_pred))/(np.sum((Ytrain)**2))
    r2_zero_train=1-(np.sum((Ytrain-(k_train*Ytrain_pred))**2))/(np.sum((Ytrain-mean_Ytrain_vector)**2))
    r_prim2_zero_train=1-(np.sum((Ytrain_pred-(k_prim_train*Ytrain))**2))/(np.sum((Ytrain_pred-mean_Ytrain_vector)**2))
    r2_m_train=r2_train*(1-np.sqrt(np.abs(r2_train-r2_zero_train)))
    r_prim2_m_train=r2_train*(1-np.sqrt(np.abs(r2_train-r_prim2_zero_train)))
    r2_m_mean_train=(r2_m_train+r_prim2_m_train)/2
    delta_r2_m_train=np.abs(r2_m_train-r_prim2_m_train)
    r2_relative_1_train=np.abs((r2_train-r2_zero_train))/r2_train
    r2_relative_2_train=np.abs((r2_train-r_prim2_zero_train))/r2_train
    delta_r2_zero_train=np.abs(r2_zero_train-r_prim2_zero_train)
    MAE_train=np.abs(np.sum(Ytrain_pred-Ytrain)/len(Ytrain))
    CCC2_train=((2*np.sum((Ytrain-mean_Ytrain_vector)*(Ytrain_pred-(mean_Ytrain_pred_vector))))/(np.sum((Ytrain-mean_Ytrain_vector)**2)+np.sum((Ytrain_pred-mean_Ytrain_pred_vector)**2)+m*((np.mean(Ytrain)-np.mean(Ytrain_pred))**2)))**2
    # Calculating statistical parameters for test set 
    r2_test=(np.sum((Ytest-mean_Ytest_vector)*(Ytest_pred-mean_Ytest_pred_vector)))**2/((np.sum((Ytest-mean_Ytest_vector)**2))*(np.sum((Ytest_pred-mean_Ytest_pred_vector)**2)))
    residuales_test=Ytest-Ytest_pred
    RMSE_test=np.std(residuales_test)
    k_test=(np.sum(Ytest*Ytest_pred))/(np.sum((Ytest_pred)**2))
    k_prim_test=(np.sum(Ytest*Ytest_pred))/(np.sum((Ytest)**2))
    r2_zero_test=1-(np.sum((Ytest-(k_test*Ytest_pred))**2)/np.sum((Ytest-mean_Ytest_vector)**2))
    r_prim2_zero_test=1-(np.sum((Ytest_pred-(k_prim_test*Ytest))**2))/(np.sum((Ytest_pred-mean_Ytest_vector)**2))
    r2_m_test=r2_test*(1-np.sqrt(np.abs(r2_test-r2_zero_test)))
    r_prim2_m_test=r2_test*(1-np.sqrt(np.abs(r2_test-r_prim2_zero_test)))
    r2_m_mean_test=(r2_m_test+r_prim2_m_test)/2
    delta_r2_m_test=np.abs(r2_m_test-r_prim2_m_test)
    r2_relative_1_test=np.abs((r2_test-r2_zero_test))/r2_test
    r2_relative_2_test=np.abs((r2_test-r_prim2_zero_test))/r2_test
    delta_r2_zero_test=np.abs(r2_zero_test-r_prim2_zero_test)
    MAE_test=np.abs((np.sum(Ytest_pred-Ytest)/len(Ytest)))
    CCC2_test=((2*np.sum((Ytest-mean_Ytest_vector)*(Ytest_pred-(mean_Ytest_pred_vector))))/(np.sum((Ytest-mean_Ytest_vector)**2)+np.sum((Ytest_pred-mean_Ytest_pred_vector)**2)+z*((np.mean(Ytest)-np.mean(Ytest_pred))**2)))**2
    return (CCC2_train,r2_train,RMSE_train,k_train,k_prim_train,r2_zero_train,r_prim2_zero_train,r2_m_train,r_prim2_m_train,r2_m_mean_train,delta_r2_m_train,r2_relative_1_train,r2_relative_2_train,delta_r2_zero_train, MAE_train, CCC2_test, r2_test,RMSE_test,k_test,k_prim_test,r2_zero_test,r_prim2_zero_test,r2_m_test,r_prim2_m_test,r2_m_mean_test,delta_r2_m_test,r2_relative_1_test,r2_relative_2_test, delta_r2_zero_test, MAE_test)








