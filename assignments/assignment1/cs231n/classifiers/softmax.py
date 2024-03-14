from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (D, C) = W.shape
    (N, _) = X.shape

    # ** FORWARD PASS ** #
    scores = X @ W  # (N, C)
    cross_entropies = np.zeros((N))

    for sample_index in range(N):
        y_true = y[sample_index]
        sample_scores = scores[sample_index]
        score_true = sample_scores[y_true]

        cross_entropy = -score_true + np.log(np.sum(np.exp(sample_scores)))
        cross_entropies[sample_index] = cross_entropy

    loss = np.mean(cross_entropies) + reg * np.sum(W ** 2)

    # ** BACKWARD PASS ** #
    d_loss = 1  # (1, )

    # Backprop: loss = np.mean(cross_entropies) + ...
    d_cross_entropies = d_loss * np.full_like(cross_entropies, 1 / N)  # (N, )
    # Backprop: loss = ... + reg * np.sum(W ** 2)
    dW += d_loss * reg * 2 * W  # (D, C)

    d_scores = np.zeros_like(scores)  # (N, C)

    for sample_index in range(N):
        y_true = y[sample_index]

        # Backprop: cross_entropies[sample_index] = cross_entropy
        d_cross_entropy = d_cross_entropies[sample_index]  # (1, )

        d_sample_scores = np.zeros_like(sample_scores)  # (C, )
        # Backprop: cross_entropy = -score_true + ...
        d_score_true = -d_cross_entropy  # (1, )
        d_sample_scores[y_true] += d_score_true

        # Backprop: cross_entropy = ... + np.log(np.sum(np.exp(sample_scores)))
        sample_scores = scores[sample_index]
        sample_scores_exp = np.exp(sample_scores)
        d_sample_scores += d_cross_entropy * sample_scores_exp / np.sum(sample_scores_exp)

        d_scores[sample_index] = d_sample_scores

    # Backprop: scores = X @ W
    dW += X.T @ d_scores

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (D, C) = W.shape
    (N, _) = X.shape

    # ** FORWARD PASS ** #

    scores = X @ W  # (N, C)
    scores_true = scores[range(N), y]  # (N, )
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1)
    LSE = np.log(scores_exp_sum)  # (N, )
    cross_entropies = -scores_true + LSE  # (N, )
    loss = np.mean(cross_entropies)

    # ** BACKWARD PASS ** #

    # Backprop: loss = np.mean(cross_entropies)
    d_cross_entropies = np.full_like(cross_entropies, 1 / N)  # (N, )

    # Backprop: cross_entropies = -scores_true + LSE
    d_scores_true = -d_cross_entropies  # (N, )
    d_LSE = d_cross_entropies  # (N, )

    d_scores = np.zeros_like(scores)  # (N, C)
    # Backprop: LSE = np.log(np.sum(np.exp(scores), axis=1))
    d_scores += d_LSE[:, np.newaxis] * (scores_exp / scores_exp_sum[:, np.newaxis])
    # Backprop: scores_true = scores[range(N), y]
    d_scores[range(N), y] += d_scores_true

    # Backprop: scores = X @ W
    dW = X.T @ d_scores

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
