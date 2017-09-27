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

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # TODO: Plot feature i against y
        # ith row meaning the ith feature
        plt.plot(X[:,i])

    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!

    # Mark one or zero?
    bias = np.zeros(shape=(X.shape[0],1))
    NX = np.append(X, bias, 1)

    w = np.linalg.solve(NX, Y)
    return w

    # raise NotImplementedError()


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test

    np.random.seed(42)
    np.random.permutation(X)
    ratio = int( X.shape[0]*0.8)
    trainData = X[ratio:,:]
    testData = X[0:ratio,:]

    # same seed for y
    np.random.seed(42)
    np.random.permutation(y)

    # Fit regression model
    w = fit_regression(trainData, y[:ratio])

    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()
