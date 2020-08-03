import numpy as np

def cost_function( x, y, w0, w1 ):
    """Compute the squared error cost function

    Inputs:
    x        vector of length m containing x values
    y        vector of length m containing y values
    w0  (scalar) intercept parameter
    w1  (scalar) slope parameter

    Returns:
    cost     (scalar) the cost
    """

    #cost = 1.0    
    cost = 0.0
    ##################################################
    # TODO: write code here to compute cost correctly
    ##################################################
    for i in range(len(x)):
        cost += ( (w0 + x[i] * w1) - y[i]) ** 2
        
    return cost/2


def gradient(x, y, w0, w1):
    """Compute the partial derivative of the squared error cost function

    Inputs:
    x          vector of length m containing x values
    y          vector of length m containing y values
    w0    (scalar) intercept parameter
    w1    (scalar) slope parameter

    Returns:
    d_w0  (scalar) Partial derivative of cost function wrt w0
    d_w1  (scalar) Partial derivative of cost function wrt w1
    """

    d_w0 = w0
    d_w1 = w1
    #learning_rate = 0.01
    ##################################################
    # TODO: write code here to compute partial derivatives correctly
    ##################################################
    #for i in range(len(x)):
    #    d_w0 += 2*x[i] * ((w0 + x[i] * w1) - y[i]) 
    #    d_w1 += -2 * (y[i] - (w0 + x[i] * w1))
    y_pred = w0 + x*w1    
    d_w0 = 2*np.sum((y_pred- y))
    d_w1 = 2*np.sum(x*(y_pred-y))
    
    return d_w0/2, d_w1/2 # return is a tuple