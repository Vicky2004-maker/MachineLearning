import numpy as np
from keras.datasets import mnist


#from tensorflow.python.keras.datasets import mnist


# %%
def load_mnist_data():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    return data, labels


def load_az_dataset(datasetPath):
    data = []
    labels = []

    for row in open(datasetPath):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype='uint8')
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype='int')

    return data, labels


# %%
digitsData, digitsLabel = load_mnist_data()
azData, azLabels = load_az_dataset("E:/Dataset/Handwritten Alphabets/A_Z Handwritten Data.csv")

# %%
