import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_prime: (batch_size, hidden_size)
            hidden state at the current time step and current layer

        """
        # h_prime =
        # return h_prime
        a = np.dot(x, self.W_ih.T) #(bs, d) (d, h)
        b = np.dot(h, self.W_hh.T)
        return self.activation(a + b + self.b_ih + self.b_hh)


    def backward(self, delta, h, h_prev_l, h_prev_t):
        """RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]

        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = self.activation.derivative(state=h) * delta
#        import pdb
#        pdb.set_trace()
        # 3 is batch_size, 20 is hidden size, 10 is input size
#        print(batch_size) # 3
#        print(dz.shape) # (3, 20)
#        print(self.dW_ih.shape) # (20, 10)
#        print(h_prev_l.shape)  # (3, 10)
#        print(h_prev_t.shape) # (3, 20)
#        print(self.dW_hh.shape) # (20, 20)
#        print(self.db_ih.shape) #(20, )
#        print(self.db_hh.shape) #(20, )

        # 1) Compute the averaged gradients of the weights and biases
        # self.dW_ih +=
        # self.dW_hh +=
        # self.db_ih +=
        # self.db_hh +=

        # 2) Compute dx, dh
        # dx =
        # dh =

        # 3) Return dx, dh
        # return dx, dh

        # (h, d)
        # 
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size
        self.db_ih += np.sum(dz.T, axis=1) / batch_size
        self.db_hh += np.sum(dz.T, axis=1) / batch_size

        dx = np.dot(dz, self.W_ih)
        dh = np.dot(dz, self.W_hh)
        return dx, dh


















