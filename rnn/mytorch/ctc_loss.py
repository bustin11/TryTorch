import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        import pdb
        self.logits = logits # (15, 12, 8)
        self.target = target # (12, 4)
        self.input_lengths = input_lengths # (12, )
        self.target_lengths = target_lengths #(12, )
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------
        ctc = CTC()

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            trunc_target = target[b, :target_lengths[b]] # (2,)
            #     Truncate the logits to input length
            trunc_logit = logits[:input_lengths[b],b,:] #(12,8) or (12,7)
            #     Extend target sequence with blank
            extSymbols, skipConnect = ctc.targetWithBlank(trunc_target) # (5,) #(5,)
            #     Compute forward probabilities
            alpha = ctc.forwardProb(trunc_logit, extSymbols, skipConnect) # (12,5)
            #     Compute backward probabilities
            beta = ctc.backwardProb(trunc_logit, extSymbols, skipConnect) # (12,5)
            #     Compute posteriors using total probability function
            gamma = ctc.postProb(alpha, beta) # (12,5) = (input length, symbols with blanks)
            #     Compute expected divergence for each batch and store it in totalLoss
            T = gamma.shape[0]
            for t in range(T):
                for i, r in enumerate(extSymbols):
                    totalLoss[b] += gamma[t,i] * np.log(trunc_logit[t,r])
            
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->

            # Your Code goes here
            # <---------------------------------------------

        return -np.sum(totalLoss) / B

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        ctc = CTC()
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            trunc_target = self.target[b, :self.target_lengths[b]]
            #     Truncate the logits to input length
            trunc_logit = self.logits[:self.input_lengths[b],b,:]
            #     Extend target sequence with blank
            extSymbols, skipConnect = ctc.targetWithBlank(trunc_target)
            #     Compute forward probabilities
            alpha = ctc.forwardProb(trunc_logit, extSymbols, skipConnect) # (12,5)
            #     Compute backward probabilities
            beta = ctc.backwardProb(trunc_logit, extSymbols, skipConnect) # (12,5)
            #     Compute posteriors using total probability function
            gamma = ctc.postProb(alpha, beta) # (12,5) = (input length, symbols with blanks)
            #     Compute derivative of divergence and store them in dY
            for t in range(self.input_lengths[b]):
                for i in range(C):
                    for j, r in enumerate(extSymbols):
                        if i == r:
                            dY[t,b,i] += -gamma[t,j] / trunc_logit[t,r]
            # <---------------------------------------------

            # -------------------------------------------->

            # Your Code goes here
            # <----------------------------------------------
        

        return dY
















