import numpy as np

"""
Attempt at creating an L-Layer neural network to optimize reordering data for businesses.
"""

def init_params(dims):
    """
    Initialize Parameters
    
    """
    
    np.random.seed(3)
    params = {}
    L = len(dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(dims[l], dims[l-1])
        params['b' + str(l)] = np.zeros((dims[l], 1))
        
    return params

def linear_forward(A, W, b):
    """
    Implement the linear part of forward propagation for single layer.
    
    """
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache

def sigmoid(Z):
    """
    Sigmoid function.
    
    """
    val = 1/(1+np.exp(-Z))
    return val, Z

def relu(Z):
    """
    Rectified Linear Units function
    
    """
    return np.maximum(0, Z), Z

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement forward propagation for one layer

    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the whole model.
    """

    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function.
    
    """
    
    m = Y.shape[1]
    AL[AL == 0] = 0.00000001
    Y[Y == 0] = 0.00000001
    cost = (-1/m)*np.sum(((Y*np.log(AL)) + (1-Y)*np.log(1-AL)))
    cost = np.squeeze(cost)
    
    return cost

def sigmoid_backward(dA, activation_cache):
    """
    Backwards sigmoid function
    """
    val, Z = sigmoid(activation_cache)
    dZ = dA*(val)*(1-val)
    return dZ

def relu_backward(dA, activation_cache):
    """
    Backwards rectufied linear units function
    
    """
    x = activation_cache
    x[x<=0] = 0
    x[x>0] = 1
    return x

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for each layer. 
    
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the whole model
    
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    AL[AL==0] = 0.00000001
    Y[Y == 0] = 0.00000001
    Y = np.resize(Y, AL.shape) 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
 
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_params(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    """
    
    L = len(params) // 2 # number of layers in the neural network

    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate*grads["dW"+ str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate*grads["db"+ str(l+1)]

    return params

def L_layer_model(X, Y, dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network
    
    """

    np.random.seed(1)
    costs = []                         
    
    params = init_params(dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, params)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        params = update_params(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(cost)
            costs.append(cost)
    
    return params

def run():
    """
    Attempt at running a test on training the model, final step doesn't work.
    
    """
    X = np.random.randint(100, 1000, (104, 50)) #test values to see if implementation works
    Y = np.random.randint(100, 1000, (1, 50)) 
    dims = [X.shape[0], 5, 5, 1]
    params = L_layer_model(X, Y, dims)
    print(params)