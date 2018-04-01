import numpy as np
import matplotlib.pyplot as plt

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X


def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y


def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1. / m) * (X.T.dot(probs * Y))

    return grad


def cal_loss(X, Y, theta):
    margins = Y * X.dot(theta)
    probs = 1. / (1 - np.exp(margins))
    return -np.sum(probs)


def display(X, Y, theta):
    plt.scatter(x=X[:, 1][Y == 1], y=X[:, 2][Y == 1], c='r', marker='o')
    plt.scatter(x=X[:, 1][Y == -1], y=X[:, 2][Y == -1], c='b', marker='x')
    x = np.linspace(0,1,100)
    plt.plot(x, -(theta[1] * x + theta[0]) / theta[2])
    plt.show()


def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    # learning_rate = 1
    learning_rate = 50

    diffs = []

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * (grad)

        diff = np.linalg.norm(prev_theta - theta)
        if i % 10000 == 0:
            p = cal_loss(X, Y, theta)
            print('Finished %d iterations' % i)
            diffs.append(diff)
        if diff < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = load_data('data_a.txt')
    # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('data_b.txt')
    logistic_regression(Xb, Yb)

    return


# def display_a():
#     plt.scatter(x=Xa[:, 1][Ya == 1], y=Xa[:, 2][Ya == 1], c='r', marker='o')
#     plt.scatter(x=Xa[:, 1][Ya == -1], y=Xa[:, 2][Ya == -1], c='b', marker='x')
#     plt.show()
#
#
# def display_b():
#     plt.scatter(x=Xb[:, 1][Yb == 1], y=Xb[:, 2][Yb == 1], c='r', marker='o')
#     plt.scatter(x=Xb[:, 1][Yb == -1], y=Xb[:, 2][Yb == -1], c='b', marker='x')
#     plt.show()

if __name__ == '__main__':
    main()
