import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('logistic_x.txt')
y = np.loadtxt('logistic_y.txt')

x = np.hstack((np.ones((x.shape[0], 1)), x))

g = lambda x: 1 / (1 + np.exp(-x))


def logistic_regression(x, y):
    theta = np.zeros(3)

    while True:
        z = x.dot(theta) * y
        J = -np.sum(g(z)) / y.size
        grad = -x.T.dot((1 - g(z)) * y) / y.size
        hessian = x.T.dot(np.diag(g(z) * (1 - g(z)))).dot(x) / y.size
        diff = np.linalg.inv(hessian).dot(grad)
        theta -= diff
        if np.linalg.norm(diff) < 1e-15:
            break
    return theta


def plot_func(fn, range):
    x = np.linspace(range[0], range[1], (range[1] - range[0]) * 100)
    plt.plot(x, fn(x))


if __name__ == '__main__':
    theta = logistic_regression(x, y)
    plot_func(lambda x: -(theta[0] + theta[1] * x) / theta[2], [0, 8])
    plt.scatter(x[:, 1][y == -1], x[:, 2][y == -1], c='r')
    plt.scatter(x[:, 1][y == 1], x[:, 2][y == 1], c='b')
    plt.show()

