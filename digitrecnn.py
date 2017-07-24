import numpy as np
from scipy import misc, optimize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os, os.path

#Function to calculate sigmoid of values
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Function to compute sigmoid gradient for back propagation
def sigmoidGrad(x):
    sigx = sigmoid(x)
    #return sigx * (1 - sigx)
    return x * (1 - x)

#Function to compute cost and gradient
def costFunc(params):

    #Reshape the weights from unrolled vector to matrices
    weights1 = params[:(hidden_layer * (input_layer + 1))].reshape(hidden_layer, input_layer + 1)
    weights2 = params[(hidden_layer * (input_layer + 1)):].reshape(output_layer, hidden_layer + 1)
    

    #Forward Propagation
    a1 = np.c_[np.ones(m), data]
    z2 = np.dot(a1, weights1.T)
    a2 = np.c_[np.ones(m), sigmoid(z2)]
    z3 = np.dot(a2, weights2.T)
    a3 = sigmoid(z3)

    #Back Propagation
    d3 = a3 - y_mat
    d2 = np.dot(d3, weights2) * sigmoidGrad(a2)
    D1 = np.dot(d2[:,1:].T, a1) / m
    D2 = np.dot(d3.T, a2) / m

    D1[:, 1:] += (lmb / m * weights1[:,1:])
    D2[:, 1:] += (lmb / m * weights2[:,1:])

    cost = np.sum(- y_mat * np.log(a3) - (1 - y_mat) * np.log(1 - a3)) / m + (lmb / (2 * m)) * (np.sum(weights1[:, 1:] ** 2) + np.sum(weights2[:, 1:] ** 2))
    grad = np.concatenate([D1.reshape(-1), D2.reshape(-1)])
    return cost, grad

#Function to return cost
def cost(params):
 return costFunc(params)[0]
 
#Function to return gradient
def grad(params):
 return costFunc(params)[1]

#Function to classify the images after training ANN
def classify(weights):
    weights1 = weights[:(hidden_layer * (input_layer + 1))].reshape(hidden_layer, input_layer + 1)
    weights2 = weights[(hidden_layer * (input_layer + 1)):].reshape(output_layer, hidden_layer + 1)

    a1 = np.c_[np.ones(m), data]
    z2 = np.dot(a1, weights1.T)
    a2 = np.c_[np.ones(m), sigmoid(z2)]
    z3 = np.dot(a2, weights2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3, axis = 1)

os.chdir("D:\\Workspace\\Python\\DigitRec\\mnist-train-images-tiff") #change working directory to image location

#Count the number of images
m = len([name for name in os.listdir('.') if os.path.isfile(name)])   #Number of training examples

data = np.zeros(shape = (m, 784))

i = 0

#Scan current directory for all images, unroll them into a row vector and form a matrix from those vectors
#Images are 28 x 28
for root, dirs, files in os.walk("."):
    for name in files:
        imgpath = os.path.join(root, name)
        img = misc.imread(imgpath)
        data[i, :] = img.reshape(-1)
        i += 1

#Scale data in the range [-1, 1]
minval = np.amin(data)
maxval = np.amax(data)
data = (data - minval) / (maxval - minval)

#Read labels from text file
os.chdir("D:\\Workspace\\Python\\DigitRec")
y = np.loadtxt('mnist-train-labels.txt') 
y = [int(item) for item in y]

#Initialize various parameters
lmb = 0.001
input_layer = 784
hidden_layer = 50
output_layer = 10
epsilon = 0.12

#Initialize weights to random values
weights1 = np.random.randn(hidden_layer, input_layer + 1) * 2 * epsilon - epsilon
weights2 = np.random.randn(output_layer, hidden_layer + 1) * 2 * epsilon - epsilon

#Calculate y matrix from the input labels
y_mat = np.eye(output_layer)[y]

#Unroll initial weights to form a vector
params = np.concatenate([weights1.reshape(-1), weights2.reshape(-1)])

#Pass the parameters to the built-in fmin_cg optimizer to train the weights
weights = optimize.fmin_cg(cost, x0 = params, fprime = grad, full_output = 1)

#Classify the training images and check the accuracy of the ANN
res = classify(weights[0])
print("The cost is",weights[1])
print("The accuracy is",(accuracy_score(y, res)) * 100, "%")