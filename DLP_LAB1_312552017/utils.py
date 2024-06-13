import numpy as np
'''
Regression Loss Functions
'''
def mean_square_error(y,t,diff=False) -> float:
    '''
    A Linear Regression Loss Function
    '''
    if diff:
        return np.mean(2.0 * (y-t)) 
    return np.mean((y-t)**2.0)
'''
Classification Loss Functions
'''
def categorical_cross_entropy(y,t,diff=False) -> float:
    '''
    A Classification Loss Function
    '''
    det = 1e-7
    if diff:
        return -(t / y) 
    return -np.sum(t * np.log(y + det))

def binary_cross_entropy(y,t,diff=False) -> float:
    '''
    A Binary Classification Loss Function
    '''
    det = 1e-7
    if diff:
        return -(t / y) + (1-t + det) / (1-y + det)
    return -np.sum(t * np.log(y + det) + (1-t) * np.log(1-y + det))


'''
Regression Activation Functions
'''
def sigmoid(x,diff=False) -> np.ndarray:
    '''
    A Logistic Regression(Binary Classifier) Activation Function
    '''  
    if diff:
        return np.multiply(x, (1.0-x))  # x is the output of the sigmoid function
    return 1.0 / (1.0+np.exp(-x))
def relu(x,diff=False) -> np.ndarray: 
    if diff:
        return 1.0*(x>0.0)
    return np.maximum(x,0.0)
def tanh(x,diff=False) -> np.ndarray:
    if diff:
        return 1.0 / np.cosh(x) ** 2
        # cosh(x) = (e^x + e^-x )/2
    return np.tanh(x)
def linear(x,diff=False) -> np.ndarray:
    if diff:
        return np.ones(x.shape)
    return x
'''
Classification Activation Functions
'''
def softmax(x, diff=False) -> np.ndarray:
    '''
    A Classification Activation Function,  used for multi-class
    '''
    c = -np.max(x)   
    # the output will be [[1.]] if output layer neuron is 1 in generate_linear case 
    return np.exp(x+c)/sum(np.exp(x+c))


'''
Optimizer Functions
'''
def SGD(learning_rate, gradient) -> float:
    '''
    take only the current gradient into consideration,
    which means the next step is not directly straight to the destination 
    '''
    return learning_rate * gradient 
def momentum(learning_rate, gradient, beta, v) -> float:
    r = learning_rate * gradient
    v = beta * v  - r # m is zero at first
    return -v
'''
Supervised Learning algorithms
'''
def logistic_regression(output, pred) -> list:
    '''
    sigmoid 
    ''' 
    if output >= 0.5:
        pred.append(1)
    else:
        pred.append(0)
    return pred
def classification(output, pred) -> list: 
    '''
    [[0.96],[0.04]] -> class 0, class 1
    class 0, class 1 -> 1, 0
    ''' 
    output = np.argmax(output)
    if output == 0:
        pred.append(1)
    elif output == 1:
        pred.append(0) 
    return pred