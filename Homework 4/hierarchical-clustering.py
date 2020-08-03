#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:13:39 2020

@author: rayansami
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering 




def hierarchicalClusteringWithMIN(data):

    clustering= AgglomerativeClustering(linkage = 'single') 
    # Show clustering using MIN
    plt.figure(figsize =(8,8)) 
    plt.scatter(data[:,0], data[:,1], c = clustering.fit_predict(data),cmap='rainbow') 
    plt.title("Clustering with MIN intercluster similarity")    
    plt.show()

def hierarchicalClusteringWithMAX(data):
    clustering= AgglomerativeClustering(linkage = 'complete') 
    # Show clustering using MAX
    plt.figure(figsize =(8,8)) 
    plt.scatter(data[:,0], data[:,1], c = clustering.fit_predict(data),cmap='rainbow') 
    plt.title("Clustering with MAX intercluster similarity")    
    plt.show()


def hierarchicalClusteringWithGroupAverage(data):
    clustering= AgglomerativeClustering(linkage = 'average') 
    # Show clustering using Group Average
    plt.figure(figsize =(8,8)) 
    plt.scatter(data[:,0], data[:,1], c = clustering.fit_predict(data),cmap='rainbow') 
    plt.title("Clustering with Group Average intercluster similarity")    
    plt.show()

def hierarchicalClusteringWithCentroidDistance(data):
    clustering= AgglomerativeClustering(linkage = 'ward') 
    # Show clustering using Distance Between Centroids
    plt.figure(figsize =(8,8)) 
    plt.scatter(data[:,0], data[:,1], c = clustering.fit_predict(data),cmap='rainbow') 
    plt.title("Clustering with Distance Between Centroids intercluster similarity")    
    plt.show()
    
def main():
    #Read files
    data = np.loadtxt("B.txt",delimiter = ' ')
    X = data[:,0]
    Y = data[:,1]
    
    # Show the data points
    plt.figure(figsize =(8,8)) 
    plt.scatter(X, Y,c='red')
    plt.show()
    
    hierarchicalClusteringWithMIN(data)
    hierarchicalClusteringWithMAX(data)
    hierarchicalClusteringWithGroupAverage(data)
    hierarchicalClusteringWithCentroidDistance(data)
    
    """
        Notes
        
        As there is low noise and outliers in this dataset, we can say
        MIN performed the best by dividing the two clusters precisely.
    """
    
if __name__ == "__main__":
    main()