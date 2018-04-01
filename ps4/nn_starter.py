import json

import numpy as np
import matplotlib.pyplot as plt


def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def softmax(x):
    """
    Compute softmax function for input.
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE

    shifted_x = x - np.max(x, axis=1, keepdims=True)
    margins = np.exp(shifted_x)

    ### END YOUR CODE
    return margins / np.sum(margins, axis=1, keepdims=True)


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = np.zeros_like(x, dtype=float)
    pos = (x > 0)
    neg = (x <= 0)
    s[pos] = 1 / (1 + np.exp(-x[pos]))
    s[neg] = np.exp(x[neg]) / (1 + np.exp(x[neg]))
    ### END YOUR CODE
    return s


def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h = sigmoid(data.dot(W1) + b1)
    y = softmax(h.dot(W2) + b2)

    cost = -np.sum(labels * np.log(y), axis=1)

    ### END YOUR CODE
    return h, y, cost


def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h, y_hat, cost = forward_prop(data, labels, params)
    B = h.shape[0]

    delta1 = y_hat - labels  # 1000 * 10
    gradb2 = np.mean(delta1, axis=0, keepdims=True)
    gradW2 = np.dot(h.T, delta1) / B

    delta2 = delta1.dot(W2.T) * h * (1 - h)
    gradb1 = np.mean(delta2, axis=0, keepdims=True)
    gradW1 = np.dot(data.T, delta2) / B

    if 'lambda' in params:
        gradW2 += 2 * params['lambda'] * W2
        gradW1 += 2 * params['lambda'] * W1

    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad


def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    a, b = trainLabels.shape
    num_hidden = 300
    learning_rate = 5
    params = {}

    ### YOUR CODE HERE
    params['W1'] = np.random.standard_normal((n, num_hidden))
    params['b1'] = np.zeros((1, num_hidden))
    params['W2'] = np.random.standard_normal((num_hidden, b))
    params['b2'] = np.zeros((1, b))

    params['lambda'] = 0.0001

    train_loss, dev_loss, train_acc, dev_acc = [], [], [], []
    B = 1000
    for epoch in range(30):
        for batch in range(m // B):
            grad = backward_prop(trainData[B * batch: B * (batch + 1)], trainLabels[B * batch: B * (batch + 1)], params)
            for item in grad:
                params[item] -= grad[item] * learning_rate
        _, y_hat, cost = forward_prop(trainData, trainLabels, params)
        train_loss.append(np.mean(cost))
        train_acc.append(compute_accuracy(y_hat, trainLabels))

        _, y_hat, cost = forward_prop(devData, devLabels, params)
        dev_loss.append(np.mean(cost))
        dev_acc.append(compute_accuracy(y_hat, devLabels))

    ### END YOUR CODE
    return params, train_loss, dev_loss, train_acc, dev_acc


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]

    devData = trainData[0:10000, :]
    devLabels = trainLabels[0:10000, :]
    trainData = trainData[10000:, :]
    trainLabels = trainLabels[10000:, :]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std

    params, a, b, c, d = nn_train(trainData, trainLabels, devData, devLabels)

    # plt.scatter(list(range(len(p[1]))), p[1], c='r')
    # plt.scatter(list(range(len(p[2]))), p[2], c='b')
    # plt.scatter(range(len(p[2])), p[2], c='b')

    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print('Test accuracy: %f' % accuracy)


if __name__ == '__main__':
    main()
