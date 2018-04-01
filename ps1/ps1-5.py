import numpy as np
import matplotlib.pyplot as plt

g = lambda x: 1 / (1 + np.exp(-x))


def add_intercept(X):
    return np.hstack((np.ones((X.size, 1)), np.reshape(X, (X.size, 1))))


def plot_func(fn, range):
    x = np.linspace(range[0], range[1], (range[1] - range[0]) * 100)
    plt.plot(x, fn(x))


def plot_bi():
    pass


def cal_weighted_y(x, y, tau):
    X = add_intercept(x)
    y_predicted = np.zeros(y.size)

    for i in range(y.size):
        w = np.exp((-(x[i] - x) ** 2) / (2 * tau ** 2)) / 2
        W = np.diag(w)
        theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
        y_predicted[i] = np.array([1, x[i]]).dot(theta)
    return y_predicted


def smooth_data(samples, x):
    for i in range(samples.shape[0]):
        samples[i] = cal_weighted_y(x, samples[i], tau=5)
    return samples


def split(samples, x):
    return samples[:, x <= 1200], samples[:, x >= 1300]


def predict_left(left_train, right_train, right_test):
    diff = right_train - right_test
    dis = np.sum(diff * diff, axis=1)
    k = 3
    k_smallest = np.argpartition(dis, k)[:k]

    h = np.max(dis)
    ker = np.sum(np.maximum(1 - dis[k_smallest] / h, 0))
    estimated = (np.maximum(1 - dis[k_smallest] / h, 0)).dot(left_train[k_smallest]) / np.sum(ker)
    return estimated


def plot_diff(x, y, y_predicted):
    plt.plot(x, y, c='r')
    plt.plot(x[:y_predicted.size], y_predicted, c='b')
    plt.show()


if __name__ == '__main__':
    data = np.loadtxt('quasar_train.csv', delimiter=',')
    test_data = np.loadtxt('quasar_test.csv', delimiter=',')

    x = data[0]

    x_ = add_intercept(x)

    samples = data[1:]
    test_samples = test_data[1:]

    y = samples[0]

    # theta = np.linalg.inv(x_.T.dot(x_)).dot(x_.T.dot(y))
    #
    # plot_func(lambda x: x * theta[1] + theta[0], (1150, 1600))
    # plt.scatter(x, y, c='r', marker='x')
    # plt.show()
    #
    # tau = 5
    # for tau, color in [(1, 'r'), (5, 'b'), (10, 'y'), (100, 'g'), (1000, 'c')]:
    #     weighted_y = cal_weighted_y(x, y, tau)
    #     plt.plot(x, weighted_y, c=color)
    # plt.scatter(x, y, c='r', marker='x')
    # plt.show()

    smoothed_samples = smooth_data(samples, x)

    smoothed_test_samples = smooth_data(test_samples, x)

    left_train, right_train = split(samples, x)

    left_test, right_test = split(test_samples, x)

    left_train_predicted = np.zeros(left_train.shape)

    for i in range(right_train.shape[0]):
        left_train_predicted[i] = predict_left(left_train, right_train, right_train[i])

    train_error = np.sum((left_train_predicted - left_train) ** 2, axis=1) / y.shape[0]

    left_test_predicted = np.zeros(left_test.shape)
    for i in range(right_test.shape[0]):
        left_test_predicted[i] = predict_left(left_train, right_train, right_test[i])

    test_error = np.sum((left_test_predicted - left_test) ** 2, axis=1) / y.shape[0]

    plot_diff(x, test_samples[0], left_train_predicted[0])
    plot_diff(x, test_samples[5], left_train_predicted[5])
