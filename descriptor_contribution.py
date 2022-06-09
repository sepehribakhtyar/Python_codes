# This program calculates the contribution of each descriptor in the
# building of model.
# Xtrain is descriptor matrix for train set. In this matrix, molecules
# or objects are in rows and independent variables are in columns.
# Xtrain has m rows and n columns.
# Ytrain is a column vector contaning dependent varible values.
###########################################################################################################
# This function has been written based on following refernce:
# F. Liu, Y. Liang, C. Caob, N. Zhoua, QSPR study of GC retention indices for saturated esters on seven
# stationary phases based on novel topological indices, Talanta 72 (2007)1307-1315.
##############################################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: fraction_contribution=descriptor_contribution(Xtrain,Ytrain)
##############################################################################################################
def descriptor_contribution(Xtrain,Ytrain):
     import numpy as np
     DIM_train=np.shape(Xtrain)
     Rows_train=DIM_train[0]
     Columns_train=DIM_train[1]
     Ones_train=np.ones((Rows_train,1))
     Xtrain_new=np.hstack((Ones_train,Xtrain))
     b=np.linalg.inv(np.transpose(Xtrain_new)@Xtrain_new)@np.transpose(Xtrain_new)@Ytrain
     Ytrain_pred=Xtrain_new@b
     Ytrain_bar=np.mean(Ytrain)*Ones_train
     w=Ytrain-Ytrain_bar
     SST_train=np.transpose(w)@w
     A=Ytrain_pred-Ytrain_bar
     SSR_train=np.transpose(A)@A
     R2_train=SSR_train/SST_train
     meandescriptors=np.transpose(np.mean(Xtrain,0))
     relative_contribution=np.transpose(np.zeros(Columns_train))
     fraction_contribution=np.transpose(np.zeros(Columns_train))
     b=np.delete(b,0)
     bnew=b
     for i in range(0,Columns_train):
         relative_contribution[i]=bnew[i]*meandescriptors[i]
         relative_contribution[i]=abs(relative_contribution[i])
     for i in range(0,Columns_train):
         fraction_contribution[i]=((R2_train*relative_contribution[i])/(np.sum(relative_contribution)))*100
     return fraction_contribution
 









