from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # As the Relu function has slope of '1'. Its gradient will be 1*dout
    d_inter = np.where(x > 0 , 1 , 0)

    # print(f"d_inter {d_inter.shape}")
    # print(f"d_out {dout.shape}")
    dx = d_inter*dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Value obtained to check whether its batch or layer norm
        axis = layernorm = bn_param.get("layernorm_value" , 0)

        # Mean and variance of the batch determined 
        mean_batch = np.mean(x , axis=0)
        var_batch = np.var(x , axis =0 )
        # Adding 'eps' to avoid division by zero 
        std_batch = np.sqrt(var_batch + eps)
        # normalised input
        x_cap = (x-mean_batch)/std_batch

        out = x_cap*gamma + beta

        # Only gets executed for batch normalizaton
        if layernorm == 0:
          running_mean = momentum * running_mean + (1 - momentum) * mean_batch
          running_var = momentum * running_var + (1 - momentum) * var_batch

        cache = (x, x_cap, gamma, mean_batch, var_batch, std_batch , axis)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Using runninng mean and variance to normalize the test data
        x_cap = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_cap * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_cap, gamma, mean_batch, var_batch, std_batch ,axis= cache

    N, D = dout.shape
    #partial derivative wrt beta and gamma
    dbeta = np.sum(dout , axis = 0)
    dgamma = np.sum(dout*x_cap , axis =0)

    # pd w.r.t. x_cap
    d_xcap = dout*gamma

    # pd w.r.t. std
    d_std_batch = -np.sum( (x-mean_batch)*d_xcap , axis=0)/(std_batch**2)
    # pd w.r.t. var 
    d_var = 0.5*d_std_batch/std_batch

    d_square = (1/N)*np.ones(dout.shape)*d_var

    # gradient for 'x^2', first to (-) node
    d_xmean2 = 2*(x-mean_batch)*d_square

    # the other gradient to (-) node
    d_x1 = d_xcap/std_batch

    # upstream gradient provided by (x-x_mean) gradients
    d_minus_node = d_x1 + d_xmean2

    #local gradient for subtraction followed by mean(i.e sum of all input values)
    d_x2 = (-1/N)*np.ones(x.shape)*np.sum(d_minus_node , axis=0)

    # gradient at the input level
    dx = d_minus_node+ d_x2 



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_cap, gamma, mean, var, std ,Axis= cache

    N, D = dout.shape
    #partial derivative wrt beta and gamma, and getting 'Axis' value to check whether
    # its batch or layer norm
    dbeta = np.sum(dout , axis = Axis)
    dgamma = np.sum(dout*x_cap , axis =Axis)

    df_dout = dout*gamma

    # evaluating the final expression for dx, found by simplifying the derivatives
    df_dout_sum = np.sum(df_dout,axis=0)                                       #[1xD]
    dx = df_dout - df_dout_sum/N - np.sum(df_dout * x_cap,axis=0) * x_cap/N    #[NxD]
    dx /= std


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Setting layer norm to true and only implementing in training mode
    ln_param['layernorm_value'] = 1
    ln_param['mode'] = 'train'
    # The shapes are transformed so that the matrices can be passed to the batch_norm function
    out , cache = batchnorm_forward(x.T, gamma.reshape(-1 , 1), beta.reshape(-1 , 1), ln_param)
    # Transposing output to the required form
    out = out.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Transpose dout, similar to forward call
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)
    # transpose gradients w.r.t. input, x, to their original dims
    dx = dx.T
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Mask created with random numbers with values less than p set to zero 
        # for inverted dropout
        mask = (np.random.rand(*x.shape) < p )/p

        out = x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # NO dropout during testing phase
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # non zero values will get the dout
        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Getting all the parameters
    pad = conv_param["pad"]
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # Padding the image for considering effects at image edges during convolution
    x_pad = np.pad(x, pad_width=( (0, 0), (0, 0) , (pad,pad) , (pad,pad) ) , mode ='constant' )

    H_pad = x_pad.shape[2]
    W_pad = x_pad.shape[3]

    # Check to see if width and height are integers
    assert ( H - HH + 2 * pad )%stride == 0, 'FAILURE in Height of conv'
    assert ( W - WW + 2 * pad )%stride == 0, 'FAILURE in width of conv layer'

    # Height and width of the output layer after convolution
    H_out = ( H - HH + 2 * pad )//stride + 1
    W_out = ( W - WW + 2 * pad )//stride + 1 

    out = np.zeros((N , F , H_out , W_out))

    # create weight_row matrix (to apply all filters simaltaneously over a given
    # region)
    w_row = w.reshape(F, C*HH*WW)

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C*HH*WW, H_out*W_out))
    for image in range(N):
        window = 0
        # Looping through the padded image
        for i in range(0, H_pad-HH+1, stride):
            for j in range(0, W_pad-WW+1, stride):
                x_col[:, window] = x_pad[image, :, i:i+HH, j:j+WW].reshape(C*HH*WW)
                window += 1

        # For each image the entire output layer determined
        out[image] = (np.dot(w_row, x_col) + b.reshape(F, 1)).reshape(F, H_out, W_out)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Getting all the parameters
    x, w, b, conv_param = cache
    pad = conv_param["pad"]
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # Padding the image for considering effects at image edges during convolution
    x_pad = np.pad(x, pad_width=( (0, 0), (0, 0) , (pad,pad) , (pad,pad) ) , mode ='constant' )   
    # Initializing gradient shape for each parameter
    dx = np.zeros_like(x) 
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx_pad = np.zeros_like(x_pad)

    dout_H = dout.shape[2]
    dout_W = dout.shape[3]
    
    # For each image
    for img in range(N):
      # For each filter
        for fil in range(F):
            # db summed over to maintain the dimensions as of 'b'
            db[fil] += np.sum(dout[img, fil])

            # The backward pass of a convolution op. is also convolution between
            # upstream gradient and local gradient
            for i in range(dout_H):
                for j in range(dout_W):
                    # The effective gradients of weights = Conv(x_pad,dout)
                    # Necessary precaution taken to maintain size of 'w'
                    dw[fil] += x_pad[img , :, i*stride:i*stride+HH, j*stride:j*stride+WW] * dout[img, fil, i, j]
                    # The effective gradients of img = Conv(w[filter],dout)                    
                    dx_pad[img, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[fil] * dout[img, fil, i, j]
    # dx have same dimension as 'x'
    dx = dx_pad[:, :, pad:-pad, pad:-pad]        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Getting all the parameters
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    step = pool_param['stride']
    N, C, H, W = x.shape

    # Getting dimension of output layer after maxpool
    H_out = 1 + (H - pool_h)//step
    W_out = 1 + (W - pool_w)//step

    out = np.zeros((N,C,H_out,W_out))

    # For each image
    for n in range(N):
      # For each channel
      for c in range(C):
        # Traversing through each element of output layer
        for h in range(H_out):
          for w in range(W_out):
            # An window is created and corresponding max value is stored in output layer
            out[n , c , h , w] = np.max(x[n , c , h*step:h*step+pool_h , w*step:w*step+pool_w ])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Getting all the parameters
    x, pool_param = cache
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    step = pool_param['stride']

    N, C, H, W = x.shape
    H_dout= dout.shape[2]
    W_dout = dout.shape[3]

    # Initializing gradient shape for parameter
    dx = np.zeros_like(x)

    # For each image
    for n in range(N):
      # For each channel
      for c in range(C):
        # Traversing through each element of dout layer
        for h in range(H_dout):
          for w in range(W_dout):
            # For maxpooling ,it doesn't have gradients.
            # So the upstream gradients are stored at the maximum value location at the 
            # corresponding window of given index of dout and rest of the elements of the
            # window are zero
            max_index = np.argmax(x[ n , c , h*step:h*step+pool_h , w*step:w*step+pool_w])
            # To get as indices
            ind = np.unravel_index(max_index , (pool_h , pool_w))
            dx[n , c , h*step:h*step+pool_h , w*step:w*step+pool_w][ind] = dout[n,c,h,w]    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Getting all the parameters
    N, C, H, W = x.shape
    # Moving the C axis to the end and converting into 2D array
    x = np.moveaxis(x, 1,-1).reshape(-1 , C)
    # Passing through regular batchnorm
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    # Shifting back to original dimensions of 'out'
    out = np.moveaxis(out.reshape(N,H,W,C) , -1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    # Shifting and reshaping the dout, so it can passed through the existing function
    dout = np.moveaxis(dout, 1,-1).reshape(-1 , C)

    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    # Putting back into the fundamental dimension of x
    dx = np.moveaxis(dx.reshape(N,H,W,C) , -1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
     
    # Getting all the parameters
    N, C, H, W = x.shape
    # Grouping and converting into 2D array form for ease
    size_2d = (N*G, C//G *H*W)
    x = x.reshape(size_2d).T
    # Mean and variance of the batch determined 
    x_mean = x.mean(axis=0)
    # Adding 'eps' to avoid division by zero 
    x_var = x.var(axis=0) + eps
    std = np.sqrt(x_var)
    # normalised input
    x_cap = (x - x_mean)/std
    x_cap = x_cap.T.reshape(N, C, H, W)
    out = gamma * x_cap + beta
    # For backwardpass
    cache={'std':std, 'gamma':gamma, 'x_cap':x_cap, 'size_2d':size_2d}

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Getting all the parameters
    x_std = cache['std']
    size_2d = cache['size_2d']
    gamma = cache['gamma']
    x_cap =cache['x_cap']

    N, C, H, W = dout.shape
    # dbeta and dgamma found as before, with only change in dimension management
    dbeta = dout.sum(axis=(0,2,3), keepdims=True)
    dgamma = np.sum(dout * x_cap, axis=(0,2,3), keepdims=True)

    # Reshaped into 2d array
    x_cap = x_cap.reshape(size_2d).T
    M = x_cap.shape[0]
    df_dout = dout * gamma
    df_dout = df_dout.reshape(size_2d).T
    # evaluating the final expression for dx, found by simplifying the derivatives
    df_dout_sum = np.sum(df_dout,axis=0)
    dx = df_dout - df_dout_sum/M - np.sum(df_dout * x_cap, axis=0) * x_cap/M
    dx /= cache['std']
    # Back to required dimension
    dx = dx.T.reshape(N, C, H, W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
