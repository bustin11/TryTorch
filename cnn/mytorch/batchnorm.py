# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """


        self.x = x


        mean = var = None
        if eval: # inference mode
            mean = self.running_mean
            var = self.running_var
        else:# train mode
            mean = self.mean = np.mean(x, axis=0)
            var = self.var = np.var(x, axis=0)
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        self.norm = (x - mean) / np.sqrt(var + self.eps)
        self.out = self.gamma * self.norm + self.beta



        return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        mean = self.mean
        var = self.var

        bs = self.x.shape[0] # batch size

        sqrt_var_eps = np.sqrt(var + self.eps)

        self.dgamma += np.sum(self.norm * delta, axis=0, keepdims=True) 
        self.dbeta += np.sum(delta, axis=0, keepdims=True)

        dNorm = self.gamma * delta
        dVar = -.5*(np.sum((dNorm * (self.x-mean)) / sqrt_var_eps**3, axis=0))
        first_term_dmu = -(np.sum(dNorm/sqrt_var_eps, axis=0))
        second_term_dmu = -(2/bs)*dVar*(np.sum(self.x-mean, axis=0))
        dMu = first_term_dmu + second_term_dmu

        first_term_dx = dNorm / sqrt_var_eps
        second_term_dx = dVar * (2/bs) * (self.x-mean)
        third_term_dx = dMu * (1/bs)

        dx = first_term_dx + second_term_dx + third_term_dx
        return dx
