import numpy as np
from activation import *

class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """

        self.x = x # (10,)
        self.hidden = h # (20, )

        # self.Wrx (20, 10)
        # self.Wrh (20, 20) 

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)


        # return h_t
        rt = self.r_act(np.dot(self.Wrx, x) + self.bir + np.dot(self.Wrh, h) + self.bhr)
        zt = self.z_act(np.dot(self.Wzx, x) + self.biz + np.dot(self.Wzh, h) + self.bhz)
        nt = self.h_act(np.dot(self.Wnx, x) + self.bin + rt * (np.dot(self.Wnh, h) + self.bhn))
        ht = (1 - zt) * nt + zt * h
        self.r = rt
        self.z = zt
        self.n = nt

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert ht.shape == (self.h,) # h_t is the final output of you GRU cell.

        return ht


    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly


        rt = self.r
        zt = self.z
        nt = self.n
        xt = self.x
        ht = self.hidden

        dx = np.zeros((1, self.d)) # (1, 5)
        dh = np.zeros((1, self.h)) # (1, 2)

        # return dx, dh

        import pdb

        # ht+1
        dnt = (1 - zt).reshape((1,-1)) * delta # (1,2) 
        dh += zt.reshape((1,-1)) * delta # (1,2)
        dzt = (-nt.reshape((1,-1)) + ht.reshape((1,-1))) * delta# (1,2)

        # nt
        dn_in = self.h_act.derivative(state=nt).reshape((1,-1)) * dnt # (1,2)
        self.dWnx += np.dot(xt.reshape((-1,1)), dn_in).T # (2,5)
        dx += np.dot(dn_in, self.Wnx) # (1, 5)
        self.dbin += dn_in.T.squeeze(axis=1)  # (2,)
        drt = (np.dot(self.Wnh, self.hidden) + self.bhn).reshape((1,-1)) * dn_in # (1,2)
        self.dWnh += np.dot(ht.reshape((-1,1)), dn_in * rt.reshape((1,-1))).T # (2, 2)
        self.dbhn += dn_in.reshape((-1,1)).squeeze(axis=1) * rt # (2,)
        dh += np.dot(dn_in * rt.reshape((1,-1)), self.Wnh) # (1,2)

        # zt
        dz_in = (self.z_act.derivative() * dzt).reshape((1,-1)) # (1,2)
        self.dWzx += np.outer(xt, dz_in).T # (2, 5)
        dx += np.dot(dz_in, self.Wzx) # (1,5)
        self.dbiz += dz_in.T.squeeze(axis=1) # (2,)
        self.dWzh += np.dot(ht.reshape((-1,1)), dz_in).T # (2,2)
        dh += np.dot(dz_in, self.Wzh) # (1,2)
        self.dbhz += dz_in.T.squeeze(axis=1) # (2,)

        # rt
        dr_in = (self.r_act.derivative() * drt).reshape((1,-1)) # (1,2)
        self.dWrx += np.outer(xt, dr_in).T # (2, 5)
        dx += np.dot(dr_in, self.Wrx) # (1,5)
        self.dbir += dr_in.T.squeeze(axis=1) # (2,)
        self.dWrh += np.dot(ht.reshape((-1,1)), dr_in).T # (2,2)
        dh += np.dot(dr_in, self.Wrh) # (1,2)
        self.dbhr += dr_in.T.squeeze(axis=1) # (2,)



        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)


        return dx, dh














