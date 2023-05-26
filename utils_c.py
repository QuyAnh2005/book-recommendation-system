import numpy as np
import tensorflow as tf


def normalize(Y, R):
    """
    Preprocess data by subtracting mean rating for every book (every row).
    Only include real ratings R(i,j)=1.

    [Y_norm, Y_mean] = normalize(Y, R) normalized Y so that each book
    has a rating of 0 on average. Unrated moves then have a mean rating (0)

    Returns the mean rating in Y_mean.
    """
    Y_mean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Y_norm = Y - np.multiply(Y_mean, R)

    return Y_norm, Y_mean

def cost_function(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the collaborative filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.

    Args:
      X (ndarray (num_books,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_books,num_users)    : matrix of user ratings of books
      R (ndarray (num_books,num_users)    : matrix, where R(i, j) = 1 if the i-th books was rated by the j-th user
      lambda_ (float): regularization parameter

    Returns:
      J (float) : Cost
    """

    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))

    return J
