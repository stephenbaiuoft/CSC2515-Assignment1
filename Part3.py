import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

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


# TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    XT = np.transpose(X)
    n = X.shape[0]
    gradient  = ( 2/n )*(XT.dot(X).dot(w) - XT.dot(y))
    # should be # of features x 1
    return gradient


# return average gradient for k iterations
# then average
def k_batch( X, y, w, k):

    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    gradientMatrix = np.array( [
        get_batch(batch_sampler, w) for i in range(k)
    ] )

    print("gradientMatrix shape should be: k x # of features? {0}".format (gradientMatrix.shape))

    # average over k times
    k_average_batch_gradient = gradientMatrix.mean(axis = 0)
    print("k_average_batch_gradient shape should be: # of features {0}".format(k_average_batch_gradient.shape))

    return k_average_batch_gradient


# return batch_grad for 1 batch
def get_batch(batch_sampler, w):
    X_b, y_b = batch_sampler.get_batch()
    batch_grad = lin_reg_gradient(X_b, y_b, w)
    return batch_grad


# given m batch,
# compute k times, each time the weight vector
# k x 13 matrix
def km_batch_variance( X, y, w, k, m):
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, m)

    gradientMatrix = np.array( [
        get_batch(batch_sampler, w) for i in range(k)
    ] )


    #print("gradientMatrix shape should be: k x # of features? {0}".format (gradientMatrix.shape))
    T = gradientMatrix.transpose()

    weight_var = T.var( axis =  1)
    # print("weight_var shape should be: 13, each representing variances? "
    #       "{0}\n\n\n{1}".format(weight_var.shape, weight_var))

    weight_var_cus = compute_variance(T)

    # if weight_var_cus.all() == weight_var.all():
    #     print("weight_matrix python and custom written equal")

    return weight_var_cus

# compute the variance of a matrix, treating each row
# as a set of data
def compute_variance( matrix ):
    # take shape = (d,1) vector? and computes variance
    def computer_variance_vector( vector ):
        u = np.mean(vector)
        # variance definition:
        # sum over (xi - u )^2, dividied by

        variance =  np.sum( (vector - u) ** 2 )/ vector.shape[0]
        # varPython = np.var(vector)

        # print("variance computed: {0}\n\nvariance by python {1}".format(variance, varPython))
        # if variance == varPython :
        #     print("equal!!")
        #
        # print("vector shape is {0}\n\nvariance is{1}".format( vector.shape, variance))
        return variance

    variance_matrix = np.array([
        computer_variance_vector(matrix[i]) for i in range(matrix.shape[0])
    ])
    # print("Custom variance matrix shape: {0}\n\n\n{1}".format( variance_matrix.shape, variance_matrix))

    return variance_matrix

# this is code for question 3.5, displaying the necessary data
def part35():
    X, y, w = load_data_and_init_params()
    # Load data and randomly initialise weights
    kmean_gradient = k_batch(X, y, w, k = 500 )

    true_gradient = lin_reg_gradient(X, y, w)

    print("kmean_gradient is: {}, \n\ntrue_gradient is: {}".format(kmean_gradient, true_gradient))
    print("kmean_gradient shape is: {}, true_gradient shape is: {}".format(kmean_gradient.shape, true_gradient.shape))
    distance = l2( true_gradient.reshape(1,13),  kmean_gradient.reshape(1, 13))
    cos_similarity = cosine_similarity(true_gradient, kmean_gradient)

    print("distance is {0}\ncos_similarity is {1}".format( distance, cos_similarity ))


# this is code for question 3.6, displaying the plot
def part36():
    # draw_all gets all the feature weight in a large subplot
    def draw_all( weight_logvarM ):
        plt.figure(figsize=(10, 12))
        feature_count = weight_logvarM.shape[1]

        log_m = np.log(np.arange(1,401))

        # i: index
        for j in range(feature_count):
            plt.subplot(5, 3, j + 1)
            # ith row meaning the ith feature
            plt.plot(log_m, weight_logvarM[:, j], 'b--')
            plt.ylabel("log variance")
            plt.xlabel("log m, for m in [1:400]")
            plt.title("{0}th log variance vs log m ".format(j + 1))

        plt.tight_layout()
        plt.show()

    X, y, w = load_data_and_init_params()

    weight_varianceMatrix = np.array([
        km_batch_variance(X, y, w, 500, m) for m in np.arange(1,401)
    ])

    # weight_sigmaMatrix is 401 x 13 matrix, with jth column representing a particular jth signma
    weight_logvarM = np.log(weight_varianceMatrix)

    # # plot first weight sigma against logm
    # plt.plot(np.log(np.arange(1,401)), weight_logvarM[:,5], 'b--')
    # plt.ylabel("log variance")
    # plt.xlabel("log m, for m in [1:400]")
    # plt.title("log variance vs log m, for j = 5")
    # plt.show()

    draw_all(weight_logvarM)


def main():
    part35()

    part36()



if __name__ == '__main__':
    main()