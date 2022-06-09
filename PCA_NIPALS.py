# This function runs principal component analysis (PCA) by using NIPALS algorithm and has been written by Dr.Sepehri from university of Kurdistan.
# X is an m-by-n matrix that has m sample and n variables (or molecular descriptors).
# NPC is the number of principal components.
# Used NIPALS algorithm is found in following reference:
# Kim H. Esbensen, Brad Swarbrick, Multivariate Data Analysis, CAMO Software AS, 2018
#############################################################################################
# To run script:
# 1- Source script
# 2- Copy following command and paste it in Consoleand then push Enter key:
# Command: T, P, Explained_variance=PCA_NIPALS(X,NPC,it=1000,tol=0.0001) 
###################################################################################################
def PCA_NIPALS(X,NPC,it=1000,tol=0.0001):
    import numpy as np
    DIM_X=np.shape(X)
    Rows_X=DIM_X[0]
    Columns_X=DIM_X[1]
    # X Mean centering calculations
    mX=np.mean(X,axis=0).reshape(1,Columns_X)
    meanmatrix=np.repeat(mX,Rows_X,axis=0)
    meancenteredX=X-meanmatrix # Xh is meancentered X matrix
    Xh=meancenteredX
    nr=0 # The number of repeats for while loop 
    T=np.zeros((Rows_X,NPC)) # T is scores matrix
    P=np.zeros((Columns_X,NPC)) # Loading matrix
    Explained_variance=np.zeros((NPC,1))
    # Calculation loadings and scores for mean centered X matrix.
    for i in range(0,NPC):
        th=Xh[:,[1]]
        ende=False
        while (not ende):
            nr=nr+1
            ph=((Xh.T@th)/(th.T@th))
            ph=ph/np.linalg.norm(ph)
            thnew=(Xh@ph)/(ph.T@ph)
            prec=np.transpose(thnew-th)@(thnew-th)
            th=thnew
            if prec<=tol**2:
                ende=True
            elif it<=nr:
                ende=True
                print("Iteration stops without convergence")
        Xh=Xh-(th@ph.T)
        T[:,[i]] = th
        P[:,[i]] = ph
    Total_variance= np.sum(np.var(X, axis=0))
    for i in range (0,NPC):
        Explained_variance[i]=(np.var(T[:,[i]])/Total_variance)*100
    return T, P, Explained_variance


