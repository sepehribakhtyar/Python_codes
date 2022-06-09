####################################################################################################################################
# input arguments
# Xtrain: Independent variables for train set (m*n matrix). Variables are in columns and objects (molecules) are in rows. 
# Ytrain: Dependent variables (a m*z matrix) for train set (m*1 matrix)
# Xtest: Independent variables matrix for test set (r*n matrix)
# Ytest: Dependent variables matrix for train set (r*z matrix)
# NPC: number of PLS components
# it: number of iterations
# tol: tolerance for convergence
# In both Ytrain and Xtrain matrices, samples are in rows and variables are in columns.
###############################################################################################################################
# Calculated Value:
# P: matrix with loadings for Xtrain
# T: matrix with scores for Xtrain
# Q: matrix with loadings for Ytrain
# U: matrix with scores for Ytrain
# W: weights for Xtrain
# C: weights for Ytrain
# D: D-matrix within the algorithm 
# B: final regression coefficients
# Ytrain_pred: Predicted properties (Dependent variables) for train set
# Ytest_pred: Predicted properties (Dependent variables) for test set
#######################################################################################################################
# Code has been written based on PLSR code in following book:
# Introduction to Multivariate Statistical Analysis in Chemometrics" written by K. Varmuza and P. Filzmoser (2009).
######################################################################################################################
# MATHEMATICAL ASPECTS:
# Y=(XB)+E
# X=(T@np.transpose(P))+EX
# Y=(U@np.transpose(Q))+EY
# U=(T@D)+H
# E and H are residual matrices
####################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command:B,T,P,W,U,Q,C,D,Ytrain_pred,Ytest_pred=PLSR2_NIPALS(Xtrain, Ytrain, Xtest, Ytest, NPC, it = 1000, tol = 1e-08)
############################################################################################################################################################
def PLSR2_NIPALS(Xtrain, Ytrain, Xtest, Ytest, NPC, it = 1000, tol = 1e-08):
    import numpy as np
    DIM_Xtrain=np.shape(Xtrain)
    Rows_Xtrain=DIM_Xtrain[0]
    Columns_Xtrain=DIM_Xtrain[1]
    DIM_Ytrain=np.shape(Ytrain)
    Rows_Ytrain=DIM_Ytrain[0]
    Columns_Ytrain=DIM_Ytrain[1]
    T=np.zeros((Rows_Xtrain,NPC)) 
    P=np.zeros((Columns_Xtrain,NPC))
    W=np.zeros((Columns_Xtrain,NPC))
    U=np.zeros((Rows_Ytrain,NPC))
    Q=np.zeros((Columns_Ytrain,NPC))
    C=np.zeros((Columns_Ytrain,NPC))
    D=np.zeros((NPC,1))
    X=Xtrain
    Y=Ytrain
    for i in range(0,NPC):
        nr=0
        u=Y[:,[0]]
        ende=False
        while (not ende):
            nr=nr+1
            w=X.T@u
            w=w/np.linalg.norm(w)
            t=X@w
            c=(Y.T@t)/(t.T@t)
            unew=Y@c
            deltau=unew-u
            unorm=np.sqrt(deltau.T@deltau)
            if unorm<tol:
                ende=True
            elif nr>it:
                ende=True
                print("WARNING!, Iterations stopped  without convergence!")
            u=unew
        p=(X.T@t)/(t.T@t)
        q=(Y.T@u)/(u.T@u)
        d=(u.T@t)/(t.T@t) # PLSR inner relation
        X=X-(t @ p.T)
        Y=Y-((t@c.T)*d)
        T[:,[i]]=t
        P[:,[i]]=p
        W[:,[i]]=w
        U[:,[i]]=u
        Q[:,[i]]=q
        C[:,[i]]=c
        D[[i]]=d
    B=W@np.linalg.inv(P.T@W)@C.T
    Ytrain_pred=Xtrain@B
    Ytest_pred=Xtest@B
    return(B,T,P,W,U,Q,C,D,Ytrain_pred,Ytest_pred)
            










