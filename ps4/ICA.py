### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np

Fs = 11025
np.seterr(all='raise')


def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))


def load_data():
    mix = np.loadtxt('mix.dat')
    return mix


def play(vec):
    sd.play(vec, Fs, blocking=True)


def g(x):
    s = np.zeros_like(x, dtype=float)
    s[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
    s[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    return s


def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')

    ######## Your code here ##########
    for alpha in anneal:
        for i in range(M):
            try:
                wx = W.dot(X[i])
                W += alpha * ((1 - 2 * g(wx)).reshape(5,1).dot(X[i].reshape((1,5))) + np.linalg.inv(W.T))
            except:
                pass

        # for i in range(200):
        #     W += alpha * ((1 - 2 * g(W.dot(X.T))).dot(X) / M + np.linalg.inv(W.T))
        #
        # wx = X.dot(W.T)
        # cost = np.sum(np.log(g(wx) * (1 - g(wx)))) + M * np.log(np.linalg.det(W))
        # print(cost)

    ###################################
    return W


def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = X.dot(W.T)

    ##################################
    return S


def main():
    X = normalize(load_data())

    # for i in range(X.shape[1]):
    #     print('Playing mixed track %d' % i)
    #     play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])
        break


if __name__ == '__main__':
    main()
