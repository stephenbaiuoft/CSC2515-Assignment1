# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import scipy

# part 1 gives the analytical solution to w*
# (XTAX + lamda*I )-1 XTAy

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        #print("j is {}".format(j))
        predictions =  np.array([    LRLS(x_test[i, :].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)    ])

        #
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses


def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    #test_datum.reshape(test_datum.shape[0], 1)
    test_datumL = test_datum.reshape(1, test_datum.shape[0])
    #print( "test_datumL shape is {}".format(test_datumL.shape))
    dist = l2(x_train, test_datumL)
    logexpSum = scipy.misc.logsumexp(-dist/(2*tau*tau))
    expSum = np.exp(logexpSum)
    exp = np.exp(- dist/ (2*tau*tau))

    Avector = exp/expSum
    #
    # check = np.sum(Avector)
    # print("check should be 1: ==? {}".format(check))

    A = Avector * np.identity(Avector.shape[0])

    XT = np.transpose(x_train)
    XTAy = XT.dot(A).dot(y_train)
    left1 = XT.dot(A).dot(x_train)
    left2 = lam * np.identity(XT.shape[0])
    left = left1 + left2

    # this is the optimal weight matrix
    w = np.linalg.solve(left, XTAy)
    #print("w is {}".format(w))
    y_hat = np.transpose(w).dot(test_datum)
    return y_hat


def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    is ? tau d x k?????
    '''

    partition = int(N/k)
    # for i in range(k):
    #     test_index = idx[i * partition:(i+1)*partition]
    #     train_index = np.setdiff1d(idx, test_index)
    #     x_test, y_test = x[test_index], y[test_index]
    #     x_train, y_train = x[train_index], y[train_index]
    #     print("i is {}".format(i))
    #     run_on_fold(x_test, y_test, x_train, y_train, taus)


    lossMatrix = np.array([  k_fold_helper(k, i, taus)
                          for i in range(k)])

    # should be k by 200? # of tau iteration
    print("lossMatrix shape is {}".format(lossMatrix.shape))

    # average over same tau
    losses = np.mean(lossMatrix, axis = 0)
    print("losses(after averaging) shape is {}".format(losses.shape))
    return losses


def k_fold_helper(k, i, taus):
    # x_test, y_test, x_train, y_train, taus
    partition = int(N/k)

    test_index = idx[i * partition:(i + 1) * partition]
    train_index = np.setdiff1d(idx, test_index)
    x_test, y_test = x[test_index], y[test_index]
    x_train, y_train = x[train_index], y[train_index]

    return run_on_fold(x_test, y_test, x_train, y_train, taus)



def test():
    A = np.array([[1],[3],[5]])
    plt.plot(A)

def part23():
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)

    plt.plot(losses, 'b--')
    plt.title("loss value vs different tau value, k = 5")
    plt.xlabel("taus in logspace of [10:1000]")
    plt.ylabel("k_fold loss for each tau ")
    plt.show()
    print("min loss = {}".format(losses.min()))

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value.
    #  Feel free to play with lambda as well if you wish

    # test()
    # change to to 200 later

    part23()