# the data is stored in 'arr2d_X.' 
# The sample size is set to n=200.
# Initilaize EM algorithm
# For simplicity, assume the number of clusters K is 2 (Note that the choice of K=2 is for ease of visualization and not based on any other specific reasons.)
# HereI naming conventions such as 'list_' or 'array_' are omitted.

import numpy as np

n = 200
muest = np.array([[0.0,0.0],[10.0,0.0]])
covest = np.array([[[1.0,0.0],[0.0,1.0]],[[1.0,0.0],[0.0,1.0]]])
west = np.array([0.5, 0.5])
K = 2

# pp contains the class membership probabaities for each data
pp = np.zeros(n*K, dtype=np.float64)
pp = pp.reshape([n, K])
nk = np.zeros(K, dtype=np.float64)




#The itteration of the EM algorithm.
from scipy.stats import multivariate_normal
for loop in range(1): #number of itterations set to 1
    #Compute pp
    for j in range(n):
        s = 0
        for i in range(K):
            pp[j,i] = west[i]*multivariate_normal.pdf(x=arr2d_X[j,0:K], mean=muest[i], cov=covest[i])
            s += pp[j,i]
        for i in range(K):
            pp[j,i] /= s
            
    #compute parameters
    #compute nk (number of data)
    for i in range(K):
        nk[i] = 0
        for j in range(n):
            nk[i] += pp[j,i]
    #compute muest (mean vectors)
    for i in range(K):
        muest[i] = np.zeros(K, dtype=np.float64)
        for j in range(n):
            muest[i] += pp[j,i]*arr2d_X[j,0:K]
        muest[i] /= nk[i]

    #compute covest (variance-covariance matricies)
    for i in range(K):
        covest[i] = np.zeros(K*K, dtype=np.float64).reshape([K,K])
        for j in range(n):
            d = np.array([arr2d_X[j,0:K]-muest[i]])
            covest[i] += pp[j,i]*(d.T.dot(d))
        covest[i] /= nk[i]
        
    #compute west (weights for classes)
    for i in range(K):
        west[i] = nk[i]/n
        
plt.scatter(arr2d_X[:,0], arr2d_X[:,1], c=pp[:,1], cmap="bwr")





# display ellipses if necessary (note that ellipse calculations are a bit complex)
plt.figure(figsize=(6,6))
ar_x0 = np.arange(-6, 10, 0.01)
        
for i in range(K):
    PM = np.linalg.inv(covest[i])
    for beta in np.arange(0,4.8,0.3):
        C = -np.log(covest[i][0,0]*covest[i][1,1]-covest[i][0,1]**2) + beta

        x0p = []
        x1p = []
        x1m = []        
        flag = 0
        for x0 in ar_x0:
            D = ((x0-muest[i][0])**2)*(PM[0,1]**2-PM[0,0]*PM[1,1])+C
            tmp = -PM[0,1]*(x0-muest[i][0])/PM[1,1]+muest[i][1]
            if D>=0:
                if flag==0:
                    x0p += [x0]
                    x1p += [tmp]
                    x1m += [tmp]
                    flag = 1
                x0p += [x0]
                x1p += [tmp+np.sqrt(D)/PM[1,1]]
                x1m += [tmp-np.sqrt(D)/PM[1,1]]
            else:
                if flag==1:
                    x0p += [x0]
                    x1p += [tmp]
                    x1m += [tmp]
                    flag = 2
        if i==0:
            plt.plot(x0p, x1p, color="blue", linewidth=1, alpha=0.3)
            plt.plot(x0p, x1m, color="blue", linewidth=1, alpha=0.3)
        else:
            plt.plot(x0p, x1p, color="red", linewidth=1, alpha=0.3)
            plt.plot(x0p, x1m, color="red", linewidth=1, alpha=0.3)
    
plt.scatter(arr2d_X[:,0], arr2d_X[:,1], c=pp[:,1], cmap="bwr")