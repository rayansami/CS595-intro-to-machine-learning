#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:11:48 2020

@author: rayansami
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

   
def initialCentroids(X,clusters): 
    np.random.RandomState(30)
    randomIds = np.random.permutation(X.shape[0])
    centroids = X[randomIds[:clusters]]
    return centroids

def computeDistance(X,centroids,clusters):
    distance = np.zeros((X.shape[0],clusters))
    for k in range(clusters):
        row_norm = norm(X - centroids[k, :], axis=1)
        distance[:, k] = np.square(row_norm)
    return distance

def findClosestCluster(distance):
    return np.argmin(distance,axis=1)
    
def computeCentroids(X,labels,clusters):
    centroids = np.zeros((clusters, X.shape[1]))
    for k in range(clusters):
        centroids[k, :] = np.mean(X[labels == k, :], axis=0)
    return centroids

def computerSSE(X,labels,centroids,clusters):
    distance = np.zeros(X.shape[0])
    for k in range(clusters):
        distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
    return np.sum(np.square(distance))
    
    
def runKmeans(X,clusters,max_iter):
    centroids = initialCentroids(X,clusters)
    labels = np.zeros(X.shape[0] - clusters)
    for iter in range(max_iter):
        # Step 1: Form k clusters by assigning all the points to the closest centroid
        oldCentroids = centroids
        distance = computeDistance(X,oldCentroids,clusters)
        labels = findClosestCluster(distance)
        
        # Step 2: Recompute the centroid of each cluster
        centroids = computeCentroids(X,labels,clusters)        
        if np.all(oldCentroids == centroids):
            break
    error = computerSSE(X,labels,centroids,clusters)
    return error

def runKmeansFor3(X,clusters,max_iter):
    centroids = initialCentroids(X,clusters)
    labels = np.zeros(X.shape[0] - clusters)
    for iter in range(max_iter):
        oldCentroids = centroids
        distance = computeDistance(X,oldCentroids,clusters)
        labels = findClosestCluster(distance)
        centroids = computeCentroids(X,labels,clusters)                
        if np.all(oldCentroids == centroids):
            break
        
    return labels,X
    
def main():
    data = np.loadtxt("A.txt",delimiter = ' ')
    
    """
        Solving HW problem 1.1
    """
    max_iter = 35
    sse = [runKmeans(data,k,max_iter) for k in range(2,11)]
    
    plt.plot(range(2,11),sse,'-o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Optimal number of clusters')
    plt.show()  
    
    """
        Solving HW problem 1.2
    """
    set_cluster = 3
    s = runKmeansFor3(data,set_cluster,max_iter)
    label = s[0]
    points = s[1]
    plt.scatter(points[:,0], points[:,1], c=label,s=30, edgecolors="red")
    
if __name__ == "__main__":
    main()