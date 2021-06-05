"""
network.py
~~~~~~~~~~
Here we have a module to (1)LOAD MNIST DATA and work with (2)NEURAL NETWORKS

(1) Data are loaded and presented in a comfortable form through the function
"load_wrap_data" the only one we will use. The functions load_data and
vectorized_result are just auxiliary to load_wrap_data.

(2)The class Network define the fedd forward network object. On it then we
have manymethods, most of which are auxiliary of
IGD(=incremental gradient descent) in order to implement the incremental
gradient descent learning algorithm on the net.
Gradients are calculated using backpropagation.
"""

###Libraries
from keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt
import copy



#part1111111111111111111111111111111111111111111111111111111111111111111111111

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
def devectorize(j):
    """does the opposite of ``vectorized_result``"""
    if np.size(j)!=1:
        j = list(j).index(1)
    return j

def load_data():
    """Return the MNIST data as a tuple containing the training data
    and the test data.
    """
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    """
    train_X: (60000, 28, 28) | train_y: (60000,)
    test_X:  (10000, 28, 28) | test_y:  (10000,)
    """
    return (train_X, train_y), (test_X, test_y)


def load_wrap_data():
    """Return a tuple containing ``(train_data,
    test_data)``. Based on ``load_data``, makes the format more convenient.

    ``train_data`` is a list containing 60,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional np.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    np.ndarray representing the unit VECTOR (0, ... , 1 , ... , 0)
    corresponding to the correct digit for ``x``.

    Similarly for ``test_data``. ``x`` is a 784-dimensional
    np.ndarry with the input image, and ``y`` is the corresponding
    classification, i.e., the DIGIT VALUE (integer) relative to ``x``.

    NOTE that we are using DIFFERENT FORMATS for train_data and test_data.
    """
    tr_d, te_d = load_data()
    train_inputs = np.array([np.reshape(x, (784, 1)) for x in tr_d[0]])
    train_results = np.array([vectorized_result(y) for y in tr_d[1]])
    train_data = [(x, y) for x, y in zip(train_inputs, train_results)]
    #np.shape(train_inputs): (60000, 784, 1)
    #np.shape(train_results): (60000, 10, 1)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = [(x, y) for x, y in zip(test_inputs, te_d[1])]
    #np.shape(test_inputs): (10000, 784, 1)
    #np.shape(test_results): (10000, 1)
    return (train_data, test_data)

###################################################################
#part222222222222222222222222222222222222222222222222222222222222222222222


"""
NB: data are received through:

data = load_wrap_data() = (train_data, test_data)

train_data = data[0]
test_data = data[1]

train_data = [ (x, y), .... , (x, y) ]
where np.shape(x)=(784,1)  np.shape(y)=(10,1)

test_data = [ (x, y), .... , (x, y) ]
where np.shape(x)=(784,1)  np.shape(y)=(1,)

"""

#### Transition function
def transition(x):
    """The transition function."""
    #return np.tanh(x)
    return 1.0/(1.0 + np.exp(-x))

def transition_prime(x):
    """Derivative of the transition function."""
    #return 1-np.tanh(x)**2
    return transition(x)*(1.0-transition(x))
    

class Network(object):

    def __init__(self, sizes):
        """we set a list [u_1, ... , u_n] in which the number u_i tells how many
        neurons there will be in the layer i. For example [784, 30, 20, 10] would
        be 4 layers of respectively 784(input), 30, 20, and 10(output) neurons.
        The biases and weights for the network are initialized randomly, using
        a Gaussian distribution with mean 0, and variance 1.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        #a bias is a COLUMN VECTOR who represent the biases for all the neurons in a layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #weights are a list of matrixes (layer^(l) x layer^(l-1)).
        #In each matrix therefore w[i, j] is the j-th weight of the i-th neuron
        #for the respective layer, remembering we start to count them from 0
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]        

    def feedforward(self, a):
        """Return the output a^(L) of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = transition(np.dot(w, a)+b)
        return a

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives of the function C,
        (BUT JUST RESPECT TO a^(L), so the activation vector of the final layer)
        i.e. a^(L)-y. """
        return (output_activations-y)

    #revision
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result aka the index of the
        neuron in the final layer which has the highest activation."""
        test1_data = [(x, devectorize(y)) for x, y in test_data]
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test1_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def err(self, test_data):
        """evaluate the error function of the network based on test_data"""
        test1_data = [(self.feedforward(x), y) for x, y in test_data]
        return (sum(sum((x-y)**2 for (x, y) in test1_data)))*0.5

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function E = C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-byiayer lists of np.arrays, the same shape as to ``self.biases``
        and ``self.weights``.
        x is the input and y is the desired output, given in the vectorized form"""
        #initializing nablas, that are in reality np.arrays
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations VECTORS, layer by layer
        zs = [] # list to store all the z VECTORS, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = transition(z)
            activations.append(activation)
        """
        here we apply the BP1 formula:
        delta_L = nabla_(a^L)C_x hadamarproduct(*) transition_prime(z^L) =
        = nabla_(a^L)E hadarmarproduct(*) transition_prime(z^L)
        """
        #hadamar/schur product in numpy = 
        delt = self.cost_derivative(activations[-1], y) * transition_prime(zs[-1])
        nabla_b[-1] = delt
        nabla_w[-1] = np.dot(delt, activations[-2].transpose())
        """
        here we apply the BP2 formula:
        delta_L = [ (W^(l+1))^T * delta_L_(l+1)) hadamarproduct(*) transition_prime(z^l)
        used to find out delta from the other elemtns on the RHS.
        And we exploit the formulas BP3 and BP4:
        (de C_x)/(de b) = delta
        (de C_x)/(de w^l) = a^(l-1) @ (delta)^T
        """
        #notice that we use the feature of python of using negative indexes
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = transition_prime(z)
            delt = np.dot(self.weights[-i+1].transpose(), delt) * sp
            nabla_b[-i] = delt
            nabla_w[-i] = np.dot(delt, activations[-i-1].transpose())
        return (np.array(nabla_b), np.array(nabla_w))

    def update(self, x, y, eps=0.0001):
        """
        made to update the biases and weights of the net after the presentation
        of one example x and the respective desired result y
        """
        nabla_b, nabla_w = self.backprop(x, y)
        self.biases = self.biases - eps*nabla_b
        self.weights = self.weights - eps*nabla_w
        
    def IGD(self, train_data, epochs=40000, eps=0.01, test_data=None):
        """Train the neural network using incremental gradient descent.
        NB: data are received through: data = load_wrap_data() = (train_data, test_data)
        train_data = [ (x, y), ... , (x, y) ] where np.shape(x)=(784,1)  np.shape(y)=(10,1)
        test_data = [ (x, y), ... , (x, y) ] where np.shape(x)=(784,1)  np.shape(y)=(1,)
            If ``test_data`` is provided then the network will be evaluated against
        the test data after each batch of 20 epochs, and partial progress printed out.
        This is useful for tracking progress, but slows things down."""
        random.shuffle(train_data)
        counter = -1
        if test_data:
            n_test = len(test_data)
            progresses_err = [self.err(test_data)/n_test]
            progresses_percent = [self.evaluate(test_data)/n_test]
        for epoch in range(epochs):
            index = random.randrange(0,len(train_data))
            x, y = train_data.pop(index)
            self.update(x, y, eps)
            counter += 1
            if test_data:
                if counter%200 == 0:
                    progresses_err.append(self.err(test_data)/n_test)
                    progresses_percent.append(self.evaluate(test_data)/n_test)
                    print ("Epoch {0}:\n\t{1} / {2}\n\t{3}".format(epoch, self.evaluate(test_data), n_test, self.err(test_data)))
            else:
                if counter%50 == 0:
                    print("Epoch {0} complete".format(epoch))
        if test_data:
            return progresses_err, progresses_percent
        
    def transcribe_IGD(self, train_data, eps, test_data, epochs=40000):
        "used to transcribe in the file all the results regarding igd with certain parameters"
        initial = (self.evaluate(test_data)/len(test_data), self.err(test_data))
        progresses = self.IGD(train_data, epochs, eps)
        final = (self.evaluate(test_data)/len(test_data), self.err(test_data))
        separator = "#"*40
        record = "\nArchitecture: "+ str(self.sizes)+ "\neps: "+ str(eps)+ "\nInitial evaluation: "+ str(initial)+ "\nFinal evaluation: "+ str(final)+"\nEpochs: "+str(epochs)
        #record = record+"\nErr every 50 epochs: "+str(progresses[0])+"\nPercent every 50 epochs: "+str(progresses[1])
        print(record)
        #write to file records
        with open('records_on_test.txt', 'a') as records:
            records.write("\n")
            records.write(separator)
            records.write(record)
        
    
    def transcribe_train_IGD(self, train_data, eps, epochs=40000, test_data=None):
            """used to transcribe in the file all the results regarding igd with certain parameters
            but ATTENTION: this time regarding train_data"""
            #let's organise precisely data:
            train1_data = copy.deepcopy(train_data)
            if test_data:
                random.shuffle(test_data)
                test_data = test_data[0:10000]
            else:
                test_data = train_data
            initial = self.evaluate(test_data)/len(test_data), self.err(test_data)/len(test_data)
            #gres_err, gres_perc =
            self.IGD(train1_data, epochs, eps)
            
            final = self.evaluate(test_data)/len(test_data), self.err(test_data)/len(test_data)
            separator = ">"*40
            record = "\nArchitecture: "+ str(self.sizes)+ "\neps: "+ str(eps)+ "\nInitial evaluation: "+ str(initial)+ "\nFinal evaluation: "+ str(final)+"\nEpochs: "+str(epochs)
            #record = record+"\nErr every 50 epochs: "+str(progresses[0])+"\nPercent every 50 epochs: "+str(progresses[1])
            #illustrativo
            print(record)
#             data1 = np.arange(0, epochs, 50)
#             fig, (ax1, ax2) = plt.subplots(1, 2)
#             ax1.plot(data1, err)
#             ax1.plt.ylabel('error')
#             ax1.plt.show()
#             ax2.plot(data1, gres_perc)
#             ax2.plt.ylabel('percentage')
#             ax2.plt.show()
            #write to file records
            with open('records.txt', 'a') as records:
                records.write("\n")
                records.write(separator)
                records.write(record)


    def transcribe_train_IGD_mod(self, train_data, eps, epochs, stop=1 ,test_data=None):
            """used to transcribe in the file all the results regarding igd with certain parameters
            but ATTENTION: this time regarding train_data"""
            #let's organise precisely data:
            train1_data = copy.deepcopy(train_data)
            if test_data:
                random.shuffle(test_data)
                test_data = test_data[0:10000]
            else:
                test_data = train_data
            initial = self.evaluate(test_data)/len(test_data), self.err(test_data)/len(test_data)
            #gres_err, gres_perc =
            mid_epochs = int(epochs*stop)
            self.IGD(train1_data, mid_epochs, eps)
            middle = self.evaluate(test_data)/len(test_data), self.err(test_data)/len(test_data)
            self.IGD(train1_data, epochs-mid_epochs, eps*0.1)
            final = self.evaluate(test_data)/len(test_data), self.err(test_data)/len(test_data)
            separator = ">"*40
            record = "\nArchitecture: "+ str(self.sizes)+ "\neps: "+ str(eps)+"\nStop:"+str(stop)+"\nMid_epochs: "+str(mid_epochs)+ "\nInitial evaluation: "+ str(initial)+"\nMiddle evaluation: "+str(middle)+ "\nFinal evaluation: "+ str(final)+"\nEpochs: "+str(epochs)
            #record = record+"\nErr every 50 epochs: "+str(progresses[0])+"\nPercent every 50 epochs: "+str(progresses[1])
            #illustrativo
            print(record)
#             data1 = np.arange(0, epochs, 50)
#             fig, (ax1, ax2) = plt.subplots(1, 2)
#             ax1.plot(data1, err)
#             ax1.plt.ylabel('error')
#             ax1.plt.show()
#             ax2.plot(data1, gres_perc)
#             ax2.plt.ylabel('percentage')
#             ax2.plt.show()
            #write to file records
            with open('records_mod.txt', 'a') as records:
                records.write("\n")
                records.write(separator)
                records.write(record)
