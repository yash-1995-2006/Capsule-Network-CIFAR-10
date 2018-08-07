import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getTrainData():
    s = 'cifar-10-batches-py/data_batch_'
    trainX = []
    trainY = []
    for i in range(1,6):
        d = unpickle(s + str(i))
        if trainX == []:
            trainX = d[b'data']
            trainY = np.array(d[b'labels'])
        else:
            trainX = np.vstack((trainX, d[b'data']))
            trainY = np.vstack((trainY, d[b'labels']))
    return trainX, trainY.reshape((-1))

def getTestData():
    s = 'cifar-10-batches-py/test_batch'
    d = unpickle(s)
    return d[b'data'], np.array(d[b'labels'])

def getOneHot(arr):
    oh = np.zeros((arr.shape[0], max(arr) + 1))
    oh[np.arange(arr.shape[0]),arr] = 1
    return oh

def getData():
    trainX, trainY = getTrainData()
    testX, testY = getTestData()
    return reshapeToImage(trainX), getOneHot(trainY), reshapeToImage(testX), getOneHot(testY)

def reshapeToImage(data):
    reshaped = np.reshape(data,(data.shape[0],3,32,32)).transpose((0,2,3,1))
    return reshaped