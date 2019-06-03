import numpy as np
import pandas as pd
import time
#from progress.bar import ChargingBar
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

np.random.seed(50)  
weights = np.random.rand(3,1)
bias = np.random.rand(1)  
learningRate = 0.05 
iteration = 20000
#bar = ChargingBar('Processing', max=iteration)

for epoch in range(iteration):  
    inputs = arrayOfSamples

    # feedforward step1
    XW = np.dot(arrayOfSamples, weights) + bias

    #feedforward step2
    z = sigmoid(XW)

    # backpropagation step 1
    error = z - labels

    #print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoidDerivation(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = arrayOfSamples.T
    weights -= learningRate * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= learningRate * num
    
    #bar.next()
#bar.finish()

single_point = np.array([19,1,1])  
result = sigmoid(np.dot(single_point, weights) + bias)  

print('')
if result >= 0.5:
    print('Da, kupice')
else:
    print('Ne, nece kupiti')