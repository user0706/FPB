import numpy as np
import pandas as pd
import time
import csv

def sigmoid(x):  
    return 1/(1+np.exp(-x))
def sigmoidDerivation(x):  
    return sigmoid(x)*(1-sigmoid(x))

file_load = pd.read_csv("data.csv")
customerDecisions = list(file_load['buy_computer'])
rows = csv.reader(open('data.csv'))
setOfSamples = list(rows)[1:]
for i in setOfSamples:
    del i[-1]
    for j in range(len(i)):
        i[j] = float(i[j])

arrayOfSamples = np.array(setOfSamples)  
labels = np.array([customerDecisions])  
labels = labels.reshape(len(customerDecisions),1)

visitor_info = [float(input('Please enter a value for {}: '.format(i))) for i in ['age', 'income', 'employed']]

np.random.seed(50)  
weights = np.random.rand(3,1)
bias = np.random.rand(1)  
learningRate = 0.05 
iteration = 20000

for epoch in range(iteration):  
    inputs = arrayOfSamples

    # Feedforward
    Z = np.dot(arrayOfSamples, weights) + bias
    s = sigmoid(Z)

    # Backpropagation
    error = s - labels
    dpred_ds = sigmoidDerivation(s)

    s_delta = error * dpred_ds

    inputs = arrayOfSamples.T
    weights -= learningRate * np.dot(inputs, s_delta)

    for num in s_delta:
        bias -= learningRate * num

instance = np.array(visitor_info)  
result = sigmoid(np.dot(instance, weights) + bias)  

if result >= 0.5:
    print('\nThere is a possibility that the visitor will buy a computer.')
else:
    print('\nThe chance that the visitor will buy a computer is small.')
