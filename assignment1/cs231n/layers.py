from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Reshaping input matrix into 2d array, so that it can be matrix multiplied with
    # weights matrix  
    x_2d = x.reshape( x.shape[0] , -1)
    x_product_w = np.matmul(x_2d,w)
    # Calculating output as linear function of input
    out = x_product_w + b
  
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Reshape input matrix
    x_2d = x.reshape( x.shape[0] , -1)
    # Gradient with respect to a feature, will have same shape as of feature
    db = np.sum(dout , axis =0)
    # The gradient wrt to a feature, is calculated by multiplying upstream 
    # gradient with the local gradient at that node(transpose if required to 
    # satisfy shape parameters for multiplication)
    dx = np.matmul(dout , w.T).reshape(x.shape)
    dw = np.matmul(x_2d.T , dout)
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # The output stores only the positive values

    out = np.where(x > 0 , x , 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # As the Relu function has slope of '1'. Its gradient will be 1*dout
    d_inter = np.where(x > 0 , 1 , 0)
    dx = d_inter*dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Initialise number of classes and training sets
    num_classes = x.shape[1]
    num_train = x.shape[0]
    loss = 0.0

    # Correct class scores are stored for each image of size (N , 1)
    correct_class_scores = x[np.arange(num_train), y].reshape(-1, 1)
    
    # margin calculation by broadcasting the difference between correct class 
    # from original scores 
    margin = np.maximum(0, x - correct_class_scores + 1) # note delta = 1

    # correct class score does not contribute to loss calculation 
    margin[np.arange(num_train), y] = 0 

    loss = np.sum(margin)
    loss /= num_train

    # Assigning gradient values for all images which satisfy margin criteria
    margin[margin>0] = 1
    count_per_image = np.sum(margin , axis =1 )
    # For all correct classes calculating the gradient
    margin[np.arange(num_train) , y] = margin[np.arange(num_train) , y] - count_per_image
    dx = margin/num_train 



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    # To avoid numerical instability when taking e^x
    x = x - np.max(x, axis=1 , keepdims=True)

    # Calculating denominator to normalize probability
    sum_for_normalization =np.sum(np.exp(x) , axis =1 , keepdims=True)

    # Softmax of size (N,C) and loss calcukation
    softmax = np.exp(x) / sum_for_normalization
    loss = -np.log(softmax[np.arange(num_train),y])

    # loss determination by summing all values, followed by averaging and regularization
    loss = np.sum(loss)
    loss /= num_train

    # For all incorrect classes, gradient of softmax is softmax itself and for
    # correct classes its (softmax-1)
    softmax[np.arange(num_train) , y] -=1
    dx = softmax/num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
