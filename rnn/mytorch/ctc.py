import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]
        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]

        """
        extSymbols = []
        skipConnect = []

        # -------------------------------------------->

        # Your Code goes here
        blanks = [self.BLANK] * (1 + len(target))
        extSymbols = np.append(blanks, target)
        extSymbols[::2] = blanks
        extSymbols[1::2] = target
        skipConnect = [0] * len(extSymbols)

        j = 0
        for i in range(len(target)):
            j += 1
            if (i > 0 and target[i] != target[i-1]):
                skipConnect[j] = 1
            j += 1
        # <---------------------------------------------

        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        y = logits
        Sext = extSymbols
        N = S
        alpha[0,0] = y[0, Sext[0]]
        alpha[0,1] = y[0, Sext[1]] 
        alpha[0,2:N] = 0
        for t in range(1, T): #through time
            alpha[t,0] = alpha[t-1,0] * y[t,Sext[0]]
            for i in range(1,N): #through symbol length
                alpha[t,i] = alpha[t-1,i-1] + alpha[t-1,i]
                if (skipConnect[i]):
                    alpha[t,i] += alpha[t-1,i-2]
                alpha[t,i] *= y[t,Sext[i]]
        

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))
        N = S

        y = logits
        Sext = extSymbols
        beta[T-1,N-1] = 1
        beta[T-1,N-2] = 1
        beta[T-1,0:N-2] = 0
        for t in range(T-2,-1,-1):
            beta[t,N-1] = beta[t+1,N-1] * y[t+1,Sext[N-1]]
            for i in range(N-2,-1,-1):
                beta[t,i] = beta[t+1,i] * y[t+1,Sext[i]] + beta[t+1,i+1] * y[t+1,Sext[i+1]]
                if (i<N-3 and skipConnect[i+2]):
                    beta[t,i] += beta[t+1,i+2] * y[t+1,Sext[i+2]]

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros(T)
        N = S

        for t in range(T):
            sumgamma[t] = 0
            for i in range(N):
                gamma[t,i] = alpha[t,i] * beta[t,i]
                sumgamma[t] += gamma[t,i]
            for i in range(N):
                gamma[t,i] = gamma[t,i] / sumgamma[t]


        return gamma








