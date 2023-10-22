  from Precode2 import *
import numpy as np
import pandas as pd
print(pd.__version__)
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import math
import numpy.ma as ma

data = np.load('AllSamples.npy')


k1,i_point1,k2,i_point2 = initial_S2('9328') # please replace 0111 with your last four digit of your ID

print(k1)
print(i_point1)
print(k2)
print(i_point2)
print(data.shape)

# initialize the centroids list 
 

 

def CreateCentroidList(data,k):
#create intitial centroids 
        CentroidList = np.empty((1,2)) 
        if    k == 4  :
            CentroidList[0][0],CentroidList[0][1]  = i_point1[0], i_point1[1]

        elif  k == 6 : 
            CentroidList[0][0],CentroidList[0][1]  = i_point2[0] , i_point2[1]    

        else:
            CentroidList = data[np.random.choice(300,1,replace =False)]

        ## compute remaining k - 1 centroids

        


        for i in range(k - 1):

            #distance of data pts from CentroidList
            distanceofdatafromcentroids = cdist(data , CentroidList, metric ='euclidean')

            #AverageArray Distance per datapoint from previous centroids

            avgarraydistanceperdatapt = np.mean(distanceofdatafromcentroids, axis = 1, dtype=float)

            #Choose datapt with maximum average distance from centroids

            #Create IndexValue for datapoint with max average distance from centroids
            NewCentroididx =  np.argmax(avgarraydistanceperdatapt, axis = 0)

            #Access CentroidValue at index and resize for appending value to CentroidList
            NewCentroidValue =np.resize((data[NewCentroididx,:]), (1,2))

            CentroidList = np.append(CentroidList,NewCentroidValue,axis =0)
       
        return CentroidList

#Now you need to calcualte the average final clusters 

def Kmeans(A,k):

# attempt to apply masking to data points to get rid of 'bad' data 
    masking = np.ma.masked_invalid(A)
    RevisedArray =ma.fix_invalid(masking, mask=False, copy=True, fill_value=A[np.random.choice(300,1,replace =False)])

    if  k == 4 :
        CentroidList = np.array(CreateCentroidList(A, 4), dtype = float)
        
    elif k == 6 :
        CentroidList = np.array(CreateCentroidList(A, 6), dtype = float )
   
    else :       
        CentroidList = np.array(CreateCentroidList(A, k), dtype = float)


        
    
    NearestCentroid = np.zeros(300).astype(float)
   # find distance between existing intitial centroid list and data and then find the nearest centroid for each datapt , then recalculate centroids until no change . 
    while True:
        PreviousCentroidNearest = NearestCentroid.copy()
        distanceOfPointsFromCentroids = cdist(A,CentroidList)
        NearestCentroid = np.nanargmin(distanceOfPointsFromCentroids,axis = 1)
    
        for b in range(k):
            CentroidList[b, : ] = RevisedArray[NearestCentroid == b].mean(axis = 0, dtype= float)
            if all(CentroidList[b, : ] == [0,0]):
                CentroidList[b, : ] ==  A[np.random.choice(300,1,replace =False)]
                
        if all(NearestCentroid == PreviousCentroidNearest):
            break
            
    FinalCentroid = NearestCentroid
    ObjectiveFunction = np.zeros((1,2))
    
	
	
	# Find loss of in data per each centroid per cluster 
    for i in range(k):
        ObjectiveFunction  += np.sum((A[FinalCentroid == i] - CentroidList[i,:])**2)
         
    return ObjectiveFunction,CentroidList


# intitializing functions and printing/graphing results
ObjectiveFunction2,CentroidList2 = Kmeans(data,2)
ObjectiveFunction4,CentroidList4 = Kmeans(data,4)
ObjectiveFunction3,CentroidList3 = Kmeans(data,3)
ObjectiveFunction5,CentroidList5 = Kmeans(data,5)
ObjectiveFunction6,CentroidList6 = Kmeans(data,6)
ObjectiveFunction7,CentroidList7 = Kmeans(data,7)
ObjectiveFunction8,CentroidList8 = Kmeans(data,8)
ObjectiveFunction9,CentroidList9 = Kmeans(data,9)
ObjectiveFunction10,CentroidList10 = Kmeans(data,10)

print(ObjectiveFunction2,CentroidList2)
print(ObjectiveFunction3,CentroidList3)
print('ObjLoss4:',ObjectiveFunction4,'CentroidList4:',CentroidList4)
print(ObjectiveFunction5,CentroidList5)
print('ObjLoss6:',ObjectiveFunction6,'CentroidList6:',CentroidList6)
print(ObjectiveFunction7,CentroidList7)
print(ObjectiveFunction8,CentroidList8)
print(ObjectiveFunction9,CentroidList9)
print(ObjectiveFunction10,CentroidList10)




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


plt.title("CentroidK3") 
plt.xlabel("x") 
plt.ylabel("y ")

plt.plot(CentroidList3) 
plt.show()

plt.title("CentroidK10") 
plt.xlabel("x") 
plt.ylabel("y ")


Objlosss = [[1921.03348586, 1921.03348586],[1293.77745239 ,1293.77745239], [805.11664575, 805.11664575], [613.28243921, 613.28243921], [592.52838426, 592.52838426], [469.13171567,469.13171567], [476.11875168 ,476.11875168], [476.11875168, 476.11875168]]
plt.title("ObjecLossStrat2")
plt.xlabel("x") 
plt.ylabel("y ")

plt.plot(Objlosss)
plt.show()