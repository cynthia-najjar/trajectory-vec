# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:36:16 2019

@author: HP-USER
"""


import pickle as cPickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn . model_selection import train_test_split
from sklearn . metrics import confusion_matrix
from sklearn . metrics import classification_report


sampleNum = 10

def vecClusterAnalysis():
    print ('Our Method')
    trVecs = []
    #trs = cPickle.load(open('./simulated_data/sim_traj_vec_normal_reverse','rb'))
    trs = cPickle.load(open('./simulated_data/sim_traj_vec_normal_humanMvtMainGauche_reverse','rb'))
   
    for tr in trs:
        trVecs.append(tr[0][0])


    #print(trVecs)
    X = np.array(trVecs)
   # print(X)
   # print(X[:,0])
   # print(X[:,1])
   # X_train , X_test , y_train , y_test = train_test_split ( df . iloc [: ,0: -1] , df . iloc
#[: , -1] , test_size =0.2)
    X_train , X_test , y_train , y_test = train_test_split(X[:,0: -1],X[:,-1] , test_size =0.2)

    kmeans = KMeans(n_clusters=20, random_state=2016)
    #y_kmeans5 = kmeans5.fit_predict(trVecs)
    clusters = kmeans.fit(trVecs).labels_.tolist()
    plt.scatter(X[:,0],X[:,1], c=clusters, cmap='rainbow')
    

    
    clustersTrain = kmeans.fit(X_train , y_train).labels_.tolist()
    #print(clustersTrain)
    y_pred  = kmeans.predict(X_test)
    #plt.scatter(y_test,y_pred, c=kmeans.labels_, cmap='rainbow')
    print(classification_report(y_test.round(),y_pred))

  
    
    #km = KMeans(n_clusters=3, random_state=2016)
    #clusters = km.fit(trVecs).labels_.tolist()
   
    
    #centers = km.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c=color, s=200, alpha=0.5);
    
    print("CLUSTERS")
    print(clusters)
    all = 0.
   
    strt = 0
    sampleNum = 5
    for c in range(1,21):
        m = 0.
        print(c)
        print("start")
        print(strt)
        print("end")
        print(sampleNum)
        l = []
        print("llllll")
        print(l)
       
        itemTocluster = set(clusters[strt:sampleNum])
        print(itemTocluster)
        print('Cluster:  '+ str(c) +' --> ')
        for i in itemTocluster:
            l.append([i,clusters[strt:sampleNum].count(i)])
        print(str(l))
        m = max([te[1] for te in l])
        print("MAXXX",m)
        all = all + m
        print (float(m)/sampleNum)

        
        strt = sampleNum
        sampleNum += 5
    
    print ('overall')
    print (all/(sampleNum))
    #print ('---------------------------------')


if __name__ == '__main__':
    vecClusterAnalysis()