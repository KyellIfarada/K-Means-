from Precode import *
import numpy as np
import pandas as pd
print(pd.__version__)
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
data = np.load('AllSamples.npy')

k1,i_point1,k2,i_point2 = initial_S1('9328') # please replace 0111 with your last four digit of your ID
print(k1)
print(k2)

# Set Initital Pts 
def Kmeans(A,k):
    if (k == 3) :
        CentroidList = i_point1

    elif (k == 5) :
        CentroidList = i_point2
    else :
        CentroidList = A[np.random.choice(300,k,replace =False)]

    NearestCentroid = np.zeros(300).astype(float)
   #Find Difference in data points and centroids provided and randomly generated until convergence which is found by if the centroid that is generated is different. 
    while True:
        PreviousCentroidNearest = NearestCentroid.copy()
        distanceOfPointsFromCentroids = cdist(A,CentroidList)
        NearestCentroid = np.argmin(distanceOfPointsFromCentroids,axis = 1)
    
        for b in range(k):
            CentroidList[b, : ] = A[NearestCentroid == b].mean(axis = 0, dtype= float)
            
        if all(NearestCentroid == PreviousCentroidNearest):
            break
    FinalCentroid = NearestCentroid
    ObjectiveFunction = np.zeros((1,2))
	
	
	# Find Objective Loss of Each Data point from its centroid per each Cluster 
    for i in range(k):
        ObjectiveFunction  += (np.sum((A[FinalCentroid == i] - CentroidList[i,:])**2))

    
    return ObjectiveFunction, CentroidList
    
    
ObjectiveFunction2,CentroidList2 = Kmeans(data,2)
ObjectiveFunction3,CentroidList3 = Kmeans(data,3)
ObjectiveFunction4,CentroidList4 = Kmeans(data,4)
ObjectiveFunction5,CentroidList5 = Kmeans(data,5)
ObjectiveFunction6,CentroidList6 = Kmeans(data,6)
ObjectiveFunction7,CentroidList7 = Kmeans(data,7)
ObjectiveFunction8,CentroidList8 = Kmeans(data,8)
ObjectiveFunction9,CentroidList9 = Kmeans(data,9)
ObjectiveFunction10,CentroidList10 = Kmeans(data,10)
print('ObjLoss2:',ObjectiveFunction2,'CentroidList2',CentroidList2)
print('ObjLoss3:',ObjectiveFunction3,'CentroidList3:',CentroidList3)
print('ObjLoss4:',ObjectiveFunction4,'CentroidList4:',CentroidList4)
print('ObjLoss5:',ObjectiveFunction5,'CentroidList5:',CentroidList5)
print('ObjLoss6:',ObjectiveFunction6,'CentroidList5:',CentroidList6)
print('ObjLoss7:',ObjectiveFunction7,'CentroidList5:',CentroidList7)
print('ObjLoss8:',ObjectiveFunction8,'CentroidList5:',CentroidList8)
print('ObjLoss9:',ObjectiveFunction9,'CentroidList5:',CentroidList9)
print('ObjLoss10:',ObjectiveFunction10,'CentroidList5:',CentroidList10)
    
  plt.title("CentroidK3") 
plt.xlabel("x") 
plt.ylabel("y ")

plt.plot(CentroidList3) 
plt.show()

plt.title("CentroidK10") 
plt.xlabel("x") 
plt.ylabel("y ")

plt.plot(CentroidList10) 
plt.show()

ObjFunctionSet= [[1921.03348586 ,1921.03348586] , [1338.10760165, 1338.10760165] ,[1115.53448124 ,1115.53448124], [592.93757297, 592.93757297], [476.11875168, 476.11875168], [399.37361987, 399.37361987] , [289.86570081, 289.86570081],[232.27848278, 232.27848278], [182.85929641, 182.85929641]] 

plt.title("ObjLossStrat1") 
plt.xlabel("x") 
plt.ylabel("y ")
plt.plot(ObjFunctionSet)
plt.show()