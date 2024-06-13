import utils
import numpy as np
 
class Seqential():
    def __init__(self):
        self.input_dim = 0 # dimension of input

        self.layers = [] # list of layer, W1, W2, ...
        self.weights = {}
        self.biases = {}
        self.z = {} # wx + b
        self.activation = {}
        self.a = {} # activation(z)
        self.gradient = {}
        self.partial_c_z = {}
        
 
        # trained technique 
        self.loss = "binary_cross_entropy"
        self.optimizer = "sgd"
        self.lr = 0.01

        # momentum
        self.momentum_beta = 0.5
        self.v = {}
    def show_layers(self):
        print(f"There're {len(self.layers)} layers in this model...") 
        for layer in self.layers: 
            print(self.weights[layer], '\n')
            print(self.biases[layer], '\n')
            print(self.activation[layer], '\n')
            print(self.gradient[layer], '\n') 
    def add_dense(self, units, input_dim="", activation=""):
        """
        Initialize a new layer with weights and biases including its dimension and value
        """

        # input dimension
        if self.input_dim == 0 and input_dim == "":
            raise Exception("input_dim is required")
        elif self.input_dim == 0:
            self.input_dim = input_dim
        
        # init layer
        layer = "W" + str(len(self.layers) + 1)
        self.layers.append(layer)
        
        self.init_dense(self.input_dim, units, activation)
        self.input_dim = units
    def init_dense(self, input_dim, output_dim, activation): 
        '''
        Randomly assign weights and biases to the newest layer,
        assign activation function,
        and initalize the gradient and momentum
        '''
        # newest layer
        layer = self.layers[-1]
        # random between 0 and 1
        np.random.seed(0)
        self.weights[layer] = (np.random.rand(output_dim, input_dim)) 
        self.biases[layer] =  (np.random.rand(output_dim, 1)) 

        # activation function
        if activation == "sigmoid":
            self.activation[layer] = utils.sigmoid
        elif activation == "relu":
            self.activation[layer] = utils.relu
        elif activation == "tanh":
            self.activation[layer] = utils.tanh
        elif activation == "softmax":
            self.activation[layer] = utils.softmax
        else:
            self.activation[layer] = utils.linear

        # gradient of weights and biases
        self.gradient[layer] = [np.zeros(self.weights[layer].shape), 
                                np.zeros(self.biases[layer].shape)]
        
        # momentum
        self.v[layer] = [np.zeros(self.weights[layer].shape), 
                  np.zeros(self.biases[layer].shape)]
        
    def next_layer(self, layer) -> str:
        '''
        Access the key of next layer (W1 -> W2 -> W3 ...)
        '''
        numeric_layer = int(layer[1:]) # extract numeric layer
        new_numeric_layer = numeric_layer + 1
        next_layer = layer[0] + str(new_numeric_layer)
        return next_layer
    
    def forward(self, x) -> np.ndarray: 
        for layer in self.layers:  
            self.z[layer] = np.dot(self.weights[layer], x) + self.biases[layer]  

            x = self.activation[layer](self.z[layer])
            
            self.a[layer] = x  

        return self.a[self.layers[-1]]
    def backward(self, pred, t):
        """
        calculate the delta

        delta(output layer) = partial_c_z = p(y_z) * partial_loss_y
        delta_l(except for output layer) = partial_c_z = p(y_z) * wT dot delta_l_next

        """ 

        last_layer = self.layers[-1]
 
        partial_c_y = self.loss(pred, t, True)
  
        partial_y_z = self.activation[last_layer](self.a[last_layer], True)
 
        self.partial_c_z[last_layer] = np.multiply(partial_c_y, partial_y_z)

        # If the activation of output layer is softmax,
        # the partial c over z is pred - t
        if self.activation[last_layer] == utils.softmax:
            self.partial_c_z[last_layer] = pred - t     

 
        for layer in reversed(self.layers):
            if layer == last_layer:
                continue
            partial_y_z = self.activation[layer](self.a[layer], True)
 
            next_layer = self.next_layer(layer)
            self.partial_c_z[layer] = np.multiply(
                                        partial_y_z,
                                        np.dot(self.weights[next_layer].T, self.partial_c_z[next_layer]) 
                                    )
        
    def gradient_descent(self, x, pred, t):
        """
        forward and backward to calculate the gradient, and then descent
        """
        self.backward(pred, t)
 
        for layer in self.layers:
            # calculate the gradient  
            self.gradient[layer][0] = np.multiply(x.T, self.partial_c_z[layer])  
            self.gradient[layer][1] = self.partial_c_z[layer]

            #print(f"grad of weight: {self.gradient[layer][0]}")
            #print(f"grad of bias: {self.gradient[layer][1]}")
             
            x = self.a[layer]
 
        for layer in self.layers:
            # gradient descent
            step_weight = 0
            step_bise = 0
            
            if self.optimizer == utils.SGD:
                step_weight = (self.optimizer(self.lr, self.gradient[layer][0])) 
                step_bise  = (self.optimizer(self.lr, self.gradient[layer][1])) 
            elif self.optimizer == utils.momentum:
                step_weight = (self.optimizer(self.lr, self.gradient[layer][0], self.momentum_beta, self.v[layer][0])) 
                step_bise = (self.optimizer(self.lr, self.gradient[layer][1], self.momentum_beta, self.v[layer][1])) 
     
            self.weights[layer] -= step_weight
            self.biases[layer] -= step_bise
 
 
    def compile(self, loss, optimizer, lr):
        if loss == "mean_square_error":
            self.loss = utils.mean_square_error
        elif loss == "binary_cross_entropy":
            self.loss = utils.binary_cross_entropy
        elif loss == "categorical_cross_entropy":
            self.loss = utils.categorical_cross_entropy
        if optimizer == "momentum":
            self.optimizer = utils.momentum
        elif optimizer == "sgd":
            self.optimizer = utils.SGD
        else:
            raise Exception("optimizer is required")
        self.lr = lr
        
    def fit(self, xs, ts, epochs) -> list:
        last_layer = self.layers[-1]
 
        loss_list = []
        for epoch in range(1,epochs+1):
            tol_loss = 0
            for x, t in zip(xs, ts):   
                # one-hot encoding
                if self.activation[last_layer] == utils.softmax:
                    if t.item() == 1:
                        t = np.array([[1],[0]])
                    else:
                        t = np.array([[0],[1]])       
                pred = self.forward(x.reshape(-1, 1)) 
                 
                self.gradient_descent(x, pred, t) 
                tol_loss += self.loss(pred, t)
 

            avg_loss = tol_loss / len(xs)
            loss_list.append(avg_loss)
            if (epoch) % 500 == 0:
                print(f"epoch {epoch} loss : {avg_loss}")
        return loss_list
 
    def predict(self, xs, ts, algorithm='logistic_regression'):
        i = 0
        pred = []
        tol_loss = 0
        output_copy = []
        for x in xs:
            output = self.forward(x.reshape(-1, 1)) 
            
            # Logistic Regression with Sigmoid Function
            if algorithm == "logistic_regression":
                output_copy.append(output[0][0])
                pred = utils.logistic_regression(output, pred)
                 

            elif algorithm == "classification":
                output_copy.append(output) 
                pred = utils.classification(output, pred)
                 
            tol_loss += self.loss(pred[i], ts[i])
            i = i + 1 
        y = []
        for i in ts:
            y.append(i[0])
        true_positive = 0
        for a, b in zip(pred, y):
            if a == b:
                true_positive += 1 

        j = 0

        if algorithm == "logistic_regression":
            for a, b in zip(y, output_copy):
                j = j + 1   
                print(f"Iter{j} |   Ground True: {a:.1f} |   prediction: {b:.5f}")
        elif algorithm == "classification": 
            for a, b in zip(y, output_copy):
                j = j + 1   
                if a == 1:
                    a = np.array([[1],[0]])
                else:
                    a = np.array([[0],[1]])
                print(f"Iter{j} |   Ground True: {a[0][0]:.1f}, {a[1][0]:.1f} |   prediction: {b[0][0]:.5f}, {b[1][0]:.5f}")
        avg_loss = tol_loss/len(xs)
        acc = true_positive / len(ts) * 100
        print(f'loss={avg_loss:.5f} accuracy={acc:.2f}%')
        return pred

