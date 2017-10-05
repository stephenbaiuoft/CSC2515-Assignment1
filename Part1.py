from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    y = np.array(y)
    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # ith row meaning the ith feature
        plt.plot(X[:,i], y)


    plt.tight_layout()
    plt.show()


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


def fit_regression(X, y):
    # TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    # Meaning to solve for (XTX)-1 using np.linalg.solve(XTX, I)

    # w* = (XTX)-1 XTy
    # add bias by appending 1
    bias = np.ones(shape=(X.shape[0],1))
    X_bias = np.append(bias, X, axis=1)

    X_tranpose = np.transpose(X_bias)
    XTX = np.dot(X_tranpose, X_bias)
    XTY = np.dot(X_tranpose, y)
    w = np.linalg.solve(XTX, XTY)
    print("w is: ", w)

    #print("w shape is: " , w.shape)
    print(w)
    return w

    # raise NotImplementedError()


def mse_calculate(w, X, y):
    bias = np.ones(shape=(X.shape[0],1))
    X_bias = np.append(bias, X, axis=1)
    yExpected = np.dot(X_bias, w)
    # print("yExpected shape: ", yExpected.shape)
    # print("y shape is:", y.shape)

    mseSum = np.dot( np.transpose(y - yExpected), y - yExpected)
    mse = mseSum / X.shape[0]
    print("mse value is {}".format(mse))

    ldd = np.linalg.norm(yExpected - y)
    #print("ldd is ", ldd)

    return mse

def l1_calculate(w, X, y):
    bias = np.ones(shape=(X.shape[0],1))
    X_bias = np.append(bias, X, axis=1)

    yPredicted = np.dot(X_bias, w)
    # y = np.array(y).reshape(y.shape[0], 1)

    diff = np.fabs (y - yPredicted)
    l1 = np.sum(diff)/X.shape[0]
    print("l1 loss is {}".format(l1))
    return l1

def r2_calculate(w, X, y):
    bias = np.ones(shape=(X.shape[0],1))
    X_bias = np.append(bias, X, axis=1)
    yPredicted = np.dot(X_bias, w)

    # y = np.array(y).reshape(y.shape[0], 1)
    sumOfSquare = np.dot( np.transpose(y - yPredicted), y - yPredicted)
    print("sumOfSquare is {}".format(sumOfSquare))

    yAverage = y.sum()/y.shape[0]
    #print("yaverage is: ", yAverage)
    ytot = y - yAverage
    print( "ytot vector shape: ", ytot.shape)
    sumOfTot = np.dot( np.transpose(ytot), ytot )
    print("sum is: ", sumOfTot)

    r2 = 1 - sumOfSquare/sumOfTot
    print("r2 is {}".format(r2))
    return r2

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test

    np.random.seed(12)
    X = np.random.permutation(X)

    ratio = int( X.shape[0]*0.8)
    trainData = X[0:ratio,:]
    testData = X[ratio:,:]

    # same seed for y
    np.random.seed(12)
    y = np.random.permutation(y)

    # Fit regression model
    w = fit_regression(trainData, y[0:ratio])

    # Compute fitted values, MSE, etc.
    mse = mse_calculate(w, testData, y[ratio:])

    l1 = l1_calculate(w, testData, y[ratio:])
    print("l1 loss is: ", l1)

    bias = np.ones(shape=(testData.shape[0],1))
    test_bias = np.append(bias, testData, axis=1)
    yPredicted = np.dot(test_bias, w)

    cos_similarity = cosine_similarity( yPredicted, y[ratio:])
    print("cos_similarity is{0}".format( cos_similarity) )

if __name__ == "__main__":
    main()
