"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = [Linear(a,b,weight_init_fn,bias_init_fn) for a,b in zip([input_size] + hiddens, hiddens + [output_size])] 

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i]) for i in range(num_bn_layers)] 


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        for i, (linear_layer, activation) in enumerate(zip(self.linear_layers, self.activations)):
            x = linear_layer(x)
            if self.bn and i < len(self.bn_layers):
                x = self.bn_layers[i](x, eval=not self.train_mode)
            x = activation(x)
        self.output = x
        return self.output


    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for linear_layer in self.linear_layers:
            linear_layer.dW.fill(0.0)
            linear_layer.db.fill(0.0)
        
        if self.bn:
            for bn_layer in self.bn_layers:
                bn_layer.dbeta.fill(0.0)
                bn_layer.dgamma.fill(0.0)
        

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for linear_layer in self.linear_layers:
            # Update weights and biases here
            dw = linear_layer.momentum_W = self.momentum * linear_layer.momentum_W - self.lr * linear_layer.dW
            db = linear_layer.momentum_b = self.momentum * linear_layer.momentum_b - self.lr * linear_layer.db
            linear_layer.W += dw
            linear_layer.b += db

        # Do the same for batchnorm layers
        if self.bn:
            for bn_layer in self.bn_layers:
                # Update weights and biases here
                bn_layer.beta -= self.lr * bn_layer.dbeta
                bn_layer.gamma -= self.lr * bn_layer.dgamma


    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.

        self.loss = self.criterion(self.output, labels)

        gradYlin = self.criterion.derivative()
        gradY = self.linear_layers[-1].backward(gradYlin)

        for i in range(len(self.linear_layers)-1)[::-1]:

            if not self.bn or i >= len(self.bn_layers):
                gradYlin = self.activations[i].derivative() * gradY # this is wihtin layer
            else:
                gradYlin = self.bn_layers[i].backward(gradY * self.activations[i].derivative())

            gradY = self.linear_layers[i].backward(gradYlin) # this is across layers (output from activation layer)
        
        # return self.loss


    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)


    # Setup ...

    print(f"Number of epochs={nepochs}")
    print(f"Batch Size={batch_size}")
    for e in range(nepochs):


        # Per epoch setup ...
        print(f"epoch={e}")
        np.random.shuffle(idxs)
        mlp.train()
        for b in range(0, len(trainx), batch_size):
            mlp.zero_grads()

            batch_idx = idxs[b:b+batch_size]
            batch_x, batch_y = trainx[batch_idx], trainy[batch_idx]

            mlp.forward(batch_x)
            mlp.backward(batch_y)

            training_losses[e] += mlp.total_loss(batch_y)
            training_errors[e] += mlp.error(batch_y)
            
            mlp.step()

        num_batches = int(len(trainx) + batch_size - 1 / batch_size)
        training_losses[e] /= num_batches 
        training_errors[e] /= len(trainx)
        print(f"train loss: {training_losses[e]}")
        print(f"train error: {training_errors[e]}")

        mlp.eval()
        for b in range(0, len(valx), batch_size):
            batch_x, batch_y = valx[b:b+batch_size], valy[b:b+batch_size]
            mlp.forward(batch_x)

            validation_losses[e] += mlp.total_loss(batch_y)
            validation_errors[e] += mlp.error(batch_y)

        num_batches = int(len(valx) + batch_size - 1 / batch_size)
        validation_losses[e] /= num_batches 
        validation_errors[e] /= len(valx)
        print(f"validation loss: {validation_losses[e]}")
        print(f"validation error: {validation_errors[e]}")

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

