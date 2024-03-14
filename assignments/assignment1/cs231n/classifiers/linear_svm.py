from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :] / num_train
                dW[:, y[i]] -= X[i, :] / num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dloss = 1
    dW += dloss * reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X @ W  # (N, C)

    scores_true = scores[range(num_train), y][:, np.newaxis]
    margins = scores - scores_true + 1
    max_margins = np.fmax(0, margins)
    # max_margins[range(num_train), y] = 0 -> Same as substracting 1 from the loss

    loss += np.sum(max_margins) / num_train - 1
    loss += reg * np.sum(W ** 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Backprop: loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W

    # Backprop: loss += np.sum(max_margins) / num_train - 1
    d_max_margins = np.ones_like(max_margins) / num_train

    # Backprop: max_margins = np.fmax(0, margins)
    d_margins = d_max_margins * np.where(margins > 0, 1, 0)  # (N, C)

    # Backprop: margins = scores - scores_true (broadcasted along axis = 1) + 1
    d_scores = d_margins
    d_scores_true = -np.sum(d_margins, axis=1)

    # Backprop: scores_true = scores[range(num_train), y][:, np.newaxis]
    # d_scores (N, C) d_scores_true (N, 1)
    d_scores[range(num_train), y] += d_scores_true * 1

    # Backprop: scores = X @ W
    # dW (D, C) = X^T (D, N) @ d_scores (N, C)
    dW += X.T @ d_scores

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized_2(W1, W2, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    - W1: A numpy array of shape (D, H) containing weights.
    - W2: A numpy array of shape (H, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    loss = 0.0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    hidden_layer = np.maximum(0, X @ W1)  # (N, H)
    scores = hidden_layer @ W2  # (N, C)

    scores_true = scores[range(num_train), y][:, np.newaxis]
    margins = scores - scores_true + 1
    max_margins = np.fmax(0, margins)

    loss += np.sum(max_margins) / num_train - 1
    loss += reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW1 = np.zeros_like(W1)
    dW2 = np.zeros_like(W2)

    # Backprop: loss += reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    dW1 += reg * 2 * W1
    dW2 += reg * 2 * W2

    # Backprop: loss += np.sum(max_margins) / num_train - 1
    d_max_margins = np.ones_like(max_margins) / num_train

    # Backprop: max_margins = np.fmax(0, margins)
    d_margins = d_max_margins * np.where(margins > 0, 1, 0)  # (N, C)

    # Backprop: margins = scores - scores_true (broadcasted along axis = 1) + 1
    d_scores = d_margins
    d_scores_true = -np.sum(d_margins, axis=1)

    # Backprop: scores_true = scores[range(num_train), y][:, np.newaxis]
    # d_scores (N, C) d_scores_true (N, 1)
    d_scores[range(num_train), y] += d_scores_true * 1

    # Backprop: hidden_layer @ W2  # (N, C)
    # dW2 (H, C) = hidden_layer^T (H, N) @ d_scores (N, C)
    dW2 += hidden_layer.T @ d_scores
    # d_hidden_layer (N, H) = d_scores (N, C) @ W2^T (C, H)
    d_hidden_layer = d_scores @ W2.T

    # Backprop: np.maximum(0, X @ W1)  # (N, H)
    # dW1 (D, H) = X^T (D, N) @ d_hidden_layer (N, H)
    dW1 += X.T @ np.where(hidden_layer > 0, d_hidden_layer, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW1, dW2
