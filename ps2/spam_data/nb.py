import numpy as np
import scipy
import matplotlib.pyplot as plt


def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)


def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    state['prior'] = np.sum(category) / N

    pos_matrix = matrix[category == 1]
    neg_matrix = matrix[category == 0]

    # might be wrong
    state['post1'] = (np.sum(pos_matrix, axis=0) + 1) / (np.sum(pos_matrix) + pos_matrix.shape[0])
    state['post0'] = (np.sum(neg_matrix, axis=0) + 1) / (np.sum(neg_matrix) + neg_matrix.shape[0])

    ###################
    return state


def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################

    true_prob = np.sum(matrix * np.log(state['post1']), axis=1) + np.log(state['prior'])
    false_prob = np.sum(matrix * np.log(state['post0']), axis=1) + np.log(1 - state['prior'])
    output[true_prob > false_prob] = 1

    ###################
    return output


def evaluate(output, label):
    error = np.sum(output != label) / len(output)
    print('Error: %1.4f' % error)
    return error


def top_n_indicative(state, n=5):
    tmp = state['post1'] / state['post0']
    return np.argpartition(tmp, tmp.size + 1 - n)[-n:]


def test_for_size(trainMatrix, trainCategory, testMatrix, testCategory, s):
    state = nb_train(trainMatrix[:s], trainCategory[:s])
    output = nb_test(testMatrix, state)
    error = evaluate(output, testCategory)
    return error


def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)

    top_5 = [tokenlist[item] for item in top_n_indicative(state)]

    train_size = [50, 100, 200, 400, 800, 1400]
    errors = []
    for s in train_size:
        errors.append(test_for_size(trainMatrix, trainCategory, testMatrix, testCategory, s))
    plt.plot(train_size, errors)
    plt.show()
    print('over')


if __name__ == '__main__':
    main()
