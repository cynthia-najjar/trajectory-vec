# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:47:04 2019

@author: HP-USER
"""
import random
import pickle as cPickle  
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

random.seed(2016)
sampleNum = 10
speedPreSec = 5
secPreCircle = 3000
a = 10000
b = 500
def sim_data(sampleNum = sampleNum,speedPreSec = speedPreSec,
    secPreCircle = secPreCircle, a = a, b = b):
    ta = math.pi*2
    noise = 50
    minLength = 2500
    maxLength = 5000
    minSample = 20
    maxSample = 50
    simData = []

    a = a
    b = b
    corrInCircleX = []
    corrInCircleY = []
    for i in range(secPreCircle):
        corrInCircleX.append(a*(math.sin(((i*2*math.pi)/secPreCircle)+1.5*math.pi)+1))
        corrInCircleY.append(b*math.cos(((i*2*math.pi)/secPreCircle)+0.5*math.pi))
    #plt.plot(corrInCircleX,corrInCircleY,'*')
    #plt.show()

    # genreate Bending
    # timeInterval, x, y
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength,maxLength)))
        sampleData = [[0,0,0]]
        j = 0
        previous = [0,0,0]
        while j< seqTimeLength:
            delta_t = random.randint(minSample,maxSample)
            x = previous[1]+ random.gauss((delta_t*speedPreSec),noise)
            y = 500 * math.sin(j/(100*math.pi))
            # random.gauss(0,50)
            j += delta_t
            sampleData.append([j,x,y])
            previous = [j,x,y]
        angle = random.random()*ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0],x1,y1])
            
        times = []
        values = []
        for i in turnSample:
            times.append(i[0])
            values.append(i[1])
        
        sns.set(style="whitegrid")
    
        #values = firstTraj[1]
       
        #print(times)
        #print(values)
        #dates = firstTraj[0]
        data = pd.DataFrame(values, times, columns=[''])
        #print(data)
        #data = data.rolling(7).mean()
        
        sns.lineplot(data=data, palette="tab10", linewidth=2.5) 
        simData.append(turnSample)

    #cPickle.dump(simData,open('./simulated_data/sim_trajectories','wb'))
   
    return simData

if __name__ == '__main__':
    sim_data()