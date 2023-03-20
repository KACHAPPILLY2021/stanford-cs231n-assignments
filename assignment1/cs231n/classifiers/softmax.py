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

    # Calculating scores
    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = X.dot(W)
    for i in range(num_train):
      # Subtracting maximum score for each image to avoid numerical unstability
      scores[i] -= np.max(scores[i]) 
      # normalized probabilities for each class of a given image
      softmax = np.exp(scores[i]) / np.sum(np.exp(scores[i]))
      # Taking softmax of correct class
      loss += -np.log(softmax[ y[i] ])
      # dW is updated for all classes of current image
      for j in range(num_class):
        dW[: , j] += X[i]*softmax[j]
      # dW updated for correct class of current image
      dW[: , y[i]] -= X[i]

    # Adding regularization of weights to loss function
    loss /= num_train
    loss = loss + reg*np.sum(W*W)
    # adding derivative of regularization
    dW /= num_train
    dW = dW + 2*reg*W
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

    num_train = X.shape[0]
    # scores calculated of size(N,C)
    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1 , keepdims=True)

    # Calculating denominator to normalize probability
    sum_for_normalization =np.sum(np.exp(scores) , axis =1 , keepdims=True)

    # Softmax of size (N,C) and loss calcukation
    softmax = np.exp(scores) / sum_for_normalization
    loss = -np.log(softmax[np.arange(num_train),y])

    # For correct classes, softmax scores updated and then dW computed
    softmax[np.arange(num_train),y] -= 1
    dW += np.matmul(X.T , softmax )

    # loss determination by summing all values, followed by averaging and regularization
    loss = np.sum(loss)
    loss /= num_train
    loss += reg*np.sum(W*W) 

    # gradient of weights computed and adding derivative of regularization
 
    dW /= num_train
    dW += 2*reg*W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
