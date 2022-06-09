# This function runs leave-many-out (LMO) cross validation test based on K-fold cross validation.
# This program has been written by Dr. Bakhtyar Sepehri.
######################################################################################################################################################################
# "repeats" is the number of iteartion.
# "NG" is the number of groups. 
# "Ytrain" is dependent variable (activity or property) vector.
# "Xtrain" is independent variable(molecular descriptors) matrix.
# In Xtrain matrice,molecules are in rows and descriptors are in columns.
# m is the number of molecules or samples.
# "average_R2traincv", "Min_R2traincv" and "Max_R2traincv", respectively, are the mean, minimum and maximum of obtained R2 for "repeat" time run of program.
# "average_RMSEtraincv" is the mean of obtained RMSE for "repeat" time run of program.
########################################################################################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: Average_R2train_cv,Average_RMSEtrain_cv,MAX_R2train_cv,MIN_R2train_cv=Kfold_LMO_CV(Ytrain,Xtrain,NG,repeats)
#######################################################################################################################################################################
def Kfold_LMO_CV(Ytrain,Xtrain,NG,repeats):
    import numpy as np
    All_R2_CV=np.zeros((repeats,1))
    All_RMSE_CV=np.zeros((repeats,1))
    m=len(Ytrain)
    Ones_train=np.ones(((m),1))
    Ytest_cv_pred_all_samples=np.zeros(((m),1))
    Ytrain=Ytrain.reshape(m,1)
    for i in range(0,repeats):
        index_train=np.arange(m)
        np.random.shuffle(index_train)
        Xtrain=Xtrain[index_train,:]
        Ytrain=Ytrain[index_train]
        groups=np.arange(m)%NG
        for group in range(0,NG):
            Train_NG=np.where(groups!=group)
            Train_NG=np.asarray(Train_NG)
            Train_NG=np.transpose(Train_NG)
            Train_NG=Train_NG.reshape(len(Train_NG),)
            Test_NG=np.where(groups==group)
            Test_NG=np.asarray(Test_NG)
            Test_NG=np.transpose(Test_NG)
            Test_NG=Test_NG.reshape(len(Test_NG),)
            Xtrain_cv=Xtrain[Train_NG,:]
            Ytrain_cv=Ytrain[Train_NG]
            Xtest_cv=Xtrain[Test_NG,:]
            Ytest_cv=Ytrain[Test_NG]
            LY_train=len(Ytrain_cv)
            LY_test=len(Ytest_cv)
            ones_LY_train=np.ones((LY_train,1))
            ones_LY_test=np.ones((LY_test,1))
            Xtrain_cv=np.hstack((ones_LY_train,Xtrain_cv))
            b=np.linalg.inv(np.transpose(Xtrain_cv)@Xtrain_cv)@np.transpose(Xtrain_cv)@Ytrain_cv
            Xtest_cv=np.hstack((ones_LY_test,Xtest_cv))
            Ytest_cv_pred=Xtest_cv@b
            Ytest_cv_pred_all_samples[Test_NG]=Ytest_cv_pred
        residualstest=Ytrain-Ytest_cv_pred_all_samples
        All_RMSE_CV[i]=np.std(residualstest)
        Ytrain_bar=np.mean(Ytrain)*Ones_train
        z=Ytrain-Ytrain_bar
        d=Ytrain-Ytest_cv_pred_all_samples
        All_R2_CV[i]=1-((np.transpose(d)@d)/(np.transpose(z)@z))
    Average_R2train_cv=np.mean(All_R2_CV)
    Average_RMSEtrain_cv=np.mean(All_RMSE_CV)
    MAX_R2train_cv=np.max(All_R2_CV)
    MIN_R2train_cv=np.min(All_R2_CV)
    return Average_R2train_cv,Average_RMSEtrain_cv,MAX_R2train_cv,MIN_R2train_cv
        
        
        













