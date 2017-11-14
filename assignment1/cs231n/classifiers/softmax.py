import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
      f = X[i].dot(W)
      f -= np.max(f)
      #inner_sum = 0.0
      #for j in xrange(num_class):
      #  inner_sum += np.exp(f[j])
      #  dW[:, j] += (X[i] * np.exp(f[j]))/np.sum(np.exp(f))
      dW += X[i][:,np.newaxis].dot(np.exp(f)[np.newaxis, :])/np.sum(np.exp(f))
      dW[:,y[i]] += -X[i]
      loss += -np.log(np.exp(f[y[i]])/np.sum(np.exp(f)))
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  f = X.dot(W)  # N x C
  f -= np.max(f, axis=1, keepdims=True)
  
  loss = - np.sum(np.log(np.exp(f[np.arange(num_train), y])/np.sum(np.exp(f), axis=1))) / num_train + 0.5 * reg * np.sum(W * W)
  
  to_subtract = np.zeros((num_train, num_class))  # N x C
  to_subtract[np.arange(num_train), y] = -1
  dW = (X.T).dot(np.exp(f)/np.sum(np.exp(f), axis=1)[:, np.newaxis] + to_subtract) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

