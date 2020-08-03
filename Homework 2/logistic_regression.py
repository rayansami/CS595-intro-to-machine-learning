import numpy as np

def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z) entrywise
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    #p = np.full(z.shape, 0.5)
    p = (1/(1 + np.exp(-z)))
    return p

def cost_function(X, y, w):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        w      parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    cost = 0
    probability = logistic(X.dot(w)) 
    cost = - np.sum( y* np.log(probability) + (1-y)*np.log((1-probability)))
    
    return cost

def gradient_descent( X, y, w0, alpha, iters ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        w          initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        w          learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """  

    # REPLACE CODE BELOW WITH CORRECT CODE
    w = w0

    m = X.shape[0]
    J_history = []
    for itr in range(iters):
        gradient = (1/m) * np.dot(X.T,(1 / (1+np.exp(-X.dot(w)))) - y)
        w -= alpha * gradient           
        probability = logistic(X.dot(w)) 
        cost = - np.sum( y* np.log(probability) + (1-y)*np.log((1-probability)))
        J_history.append(cost)        

    return w, J_history