import numpy as np
import tensorflow as tf

from tensorflow import keras
from pathlib import Path
from jsonargparse import ArgumentParser
from utils_c import normalize, cost_function


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, default="./processed")
    parser.add_argument("--out_dir", type=str, required=True, default="./weight")
    parser.add_argument("--num_features", type=int, required=True, default=10)
    parser.add_argument("--num_iterators", type=int, required=True, default=200)
    parser.add_argument("--learning_rate", type=float, required=True, default=1e-1)
    parser.add_argument("--lambda_", type=float, required=True, default=2.0)
    parser.add_argument("--seed", type=int, required=True, default=1234)
    parser.add_argument("--freq", type=int, required=True, default=20)

    return vars(parser.parse_args())

def main(
    data_dir,
    out_dir,
    num_features,
    num_iterators,
    learning_rate,
    lambda_,
    seed,
    freq
):
    # Load R matrix from file
    R = np.load(f'{data_dir}/R.npy', allow_pickle=True)
    # Load Y matrix from file
    Y = np.load(f'{data_dir}/Y.npy', allow_pickle=True)
    # Normalize the Dataset
    Y_norm, Y_mean = normalize(Y, R)

    num_books, num_users = Y.shape
    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(seed) # for consistent results

    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
    X = tf.Variable(tf.random.normal((num_books, num_features), dtype=tf.float64), name='X')
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    for iter in range(num_iterators):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost
        with tf.GradientTape() as tape:
            # Compute the cost (forward pass included in cost)
            cost_value = cost_function(X, W, b, Y_norm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient(cost_value, [X, W, b])

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        # Log periodically.
        if iter % freq == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    predict = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
    predict = predict + Y_mean

    # Save weight
    out_dir = Path(out_dir)
    if out_dir.exists():
        assert out_dir.is_dir()
    else:
        out_dir.mkdir(parents=True)
    np.save(f'{out_dir}/W.npy', W)
    np.save(f'{out_dir}/X.npy', X)
    np.save(f'{out_dir}/b.npy', b)
    np.save(f'{out_dir}/predicted.npy', predict)


if __name__ == "__main__":
    main(**parse_args())
