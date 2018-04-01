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


def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))


def load_data():
    mix = np.loadtxt('mix.dat')
    return mix


def play(vec):
    sd.play(vec, Fs, blocking=True)


def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    for alpha in anneal:
        totalcost = 0.0
        for i in range(M):
            wx = W.dot(X[i])
            pos = (wx > 0)
            neg = (wx <= 0)
            sig = np.zeros_like(wx, dtype=float)
            sig[pos] = 1 / (1 + np.exp(-wx[pos]))
            sig[neg] = np.exp(wx[neg]) / (1 + np.exp(wx[neg]))
            propogate = np.matrix(1 - 2 * sig).T * np.matrix(X[i])
            W = np.array(W + alpha * (propogate + np.linalg.inv(W.T)))

            # Observe the cost for each loop
            wx1 = W.T.dot(X[i])
            pos = (wx1 > 0)
            neg = (wx1 <= 0)
            sig1 = np.zeros_like(wx1, dtype=float)
            sig1[pos] = 1 / (1 + np.exp(-wx1[pos]))
            sig1[neg] = np.exp(wx1[neg]) / (1 + np.exp(wx1[neg]))
            log = np.log(sig1 * (1 - sig1))
            cost = np.sum(log) + np.log(np.linalg.det(W))
            totalcost = totalcost + cost
        print('total lost: %f' % totalcost)
    ###################################
    return W


def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = np.dot(X, W.T)
    ##################################
    return S


def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        # play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])


if __name__ == '__main__':
    main()