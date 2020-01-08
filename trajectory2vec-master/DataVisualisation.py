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
    trs = cPickle.load(open('./simulated_data/sim_traj_vec_normal_reverse','rb'))
  
    for tr in trs:
        trVecs.append(tr[0][0])
        
   
    X = np.array(trVecs)

   # X_train , X_test , y_train , y_test = train_test_split ( df . iloc [: ,0: -1] , df . iloc
#[: , -1] , test_size =0.2)
    X_train , X_test , y_train , y_test = train_test_split(X[:,0: -1],X[:,-1] , test_size =0.2)

    kmeans = KMeans(n_clusters=3, random_state=2016)
    clusters = kmeans.fit(X).labels_.tolist()
    plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    
    
    clustersTrain = kmeans.fit(X_train , y_train).labels_.tolist()
    #print(clustersTrain)
    y_pred  = kmeans.predict(X_test)
    #plt.scatter(y_test,y_pred, c=kmeans.labels_, cmap='rainbow')
    print(classification_report(y_test.round(),y_pred))

  
    
    #km = KMeans(n_clusters=3, random_state=2016)
    #clusters = km.fit(trVecs).labels_.tolist()
   
    
    #centers = km.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c=color, s=200, alpha=0.5);
    
    #print("clusters")
    print(clusters)
    all = 0.
   
    item = set(clusters[:sampleNum])
    
    print(item)
    l = []
    for i in item:
        l.append([i,clusters[:sampleNum].count(i)])
        
    
    print ('Straight:  '+ str(l))
    m = max([te[1] for te in l])
    print("----------------------")
    print(m)
    all = all + m
    print (float(m)/sampleNum)
 

    m = 0.
    item = set(clusters[sampleNum:sampleNum*2])
    l = []
    for i in item:
        l.append([i,clusters[sampleNum:sampleNum*2].count(i)])
    print ('Circling:  '+ str(l))
    m = max([te[1] for te in l])
    all = all + m
    print (float(m)/sampleNum)

    m = 0.
    item = set(clusters[sampleNum*2:sampleNum*3])
    l = []
    for i in item:
        l.append([i,clusters[sampleNum*2:sampleNum*3].count(i)])
    m = max([te[1] for te in l])
    print ('bending:   '+ str(l))
    all = all + m
    print (float(m)/sampleNum)
    print ('overall')
    print (all/(sampleNum*3))
    #print ('---------------------------------')


if __name__ == '__main__':
    vecClusterAnalysis()