# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import pdb
import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        self.W            (out_channel, in_channel, kernel_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        batch_size = x.shape[0]
        input_size = x.shape[2]
        output_size = (input_size - self.kernel_size)//self.stride + 1
        ans = []
        for i in range(output_size):
            offset = i * self.stride
            x_section = x[:,:,offset:offset+self.kernel_size]
            no_bias = np.tensordot(x_section, self.W, axes=((1,2), (1,2))) # (batch_size, out_channel)
            ans.append(no_bias + self.b.reshape(1, len(self.b)))

        
        ans = np.transpose(np.array(ans), axes = (1, 2, 0))
        return ans


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        bs = delta.shape[0]
        oc = delta.shape[1]
        ds = delta.shape[2]
        dilated_delta = np.zeros((bs, oc, ds + (ds-1) * (self.stride-1)))
        filtered_idx = np.array([i for i in range(ds + (ds-1)*(self.stride-1)) if i % self.stride == 0])
        dilated_delta[:,:,filtered_idx] = delta

        # calculate wrt to filter
        for i in range(self.dW.shape[2]):
            x_section = self.x[:,:,i:i+dilated_delta.shape[2]]
            dw_per_batch = np.tensordot(x_section, dilated_delta, axes=((0,2), (0,2))).T
            self.dW[:,:,i] += dw_per_batch

        # calculate wrt to bias
        self.db = np.sum(np.sum(delta, axis=2), axis=0)

        # caluclate wrt to input
        ks = self.kernel_size
        padded_delta = np.pad(dilated_delta, [(0,0), (0,0), (ks-1, ks-1 + (self.x.shape[2] - ks) % self.stride)], mode='constant', constant_values=0)
        flipped_filter = np.flip(self.W, axis=2)
        dx = []
        for i in range(padded_delta.shape[2]-ks+1):
            d_section = padded_delta[:,:,i:i+ks]
            dx_per_batch = np.tensordot(d_section, flipped_filter, axes=((1,2), (0,2)))
            dx.append(dx_per_batch)
       
        dx = np.transpose(np.array(dx), axes = (1, 2, 0))
        return dx
        


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        bs = x.shape[0]
        ks = self.kernel_size
        input_width, input_height = x.shape[2], x.shape[3]

        output_width = (input_width - ks)//self.stride + 1
        output_height = (input_height - ks)//self.stride + 1

        ans = [[0] * output_width for _ in range(output_height)]
        for r in range(output_height):
            for c in range(output_width):
                offset_r, offset_c = r * self.stride, c * self.stride
                x_section = x[:,:,offset_r:offset_r+ks, offset_c:offset_c+ks]
                no_bias = np.tensordot(x_section, self.W, axes=((1,2,3), (1,2,3))) # (batch_size, out_channel)
                ans[r][c] = no_bias + self.b.reshape(1, len(self.b))

        
        
        ans = np.transpose(np.array(ans), axes = (2, 3, 0, 1))
        return ans

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        bs = delta.shape[0]
        oc = delta.shape[1]
        ow = delta.shape[2]
        oh = delta.shape[3]

        dilated_delta = np.zeros((bs, oc, ow + (ow-1) * (self.stride-1), oh + (oh-1) * (self.stride-1)))
        filtered_idx_r = np.array([i for i in range(ow + (ow-1)*(self.stride-1)) if i % self.stride == 0])
        filtered_idx_c = np.array([i for i in range(oh + (oh-1)*(self.stride-1)) if i % self.stride == 0])
        x, y = np.ix_(filtered_idx_r, filtered_idx_c)
        dilated_delta[:,:,x,y] = delta

        # calculate wrt to filter
        for r in range(self.dW.shape[2]):
            for c in range(self.dW.shape[3]):
                x_section = self.x[:,:,r:r+dilated_delta.shape[2],c:c+dilated_delta.shape[3]]
                dw_per_batch = np.tensordot(x_section, dilated_delta, axes=((0,2,3), (0,2,3))).T
                self.dW[:,:,r,c] += dw_per_batch

        # calculate wrt to bias
        self.db = delta.sum(axis=(0,2,3))

        # caluclate wrt to input
        ks = self.kernel_size
        padded_delta = np.pad(dilated_delta, [(0,0), (0,0), 
            (ks-1, ks-1 + (self.x.shape[2] - ks) % self.stride),
            (ks-1, ks-1 + (self.x.shape[3] - ks) % self.stride)],
            mode='constant', constant_values=0)

        flipped_filter = np.flip(np.flip(self.W, axis=2), axis=3)
        width_iters = padded_delta.shape[2]-ks+1
        height_iters = padded_delta.shape[3]-ks+1
        dx = [[0] * width_iters for _ in range(height_iters)]
        for r in range(width_iters):
            for c in range(height_iters):
                d_section = padded_delta[:,:,r:r+ks,c:c+ks]
                dx_per_batch = np.tensordot(d_section, flipped_filter, axes=((1,2,3), (0,2,3)))
                dx[r][c] = dx_per_batch
       
        dx = np.transpose(np.array(dx), axes = (2, 3, 0, 1))
        return dx
         


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        ks = self.kernel_size
        new_k = (ks-1) * (dilation-1) + ks
        self.kernel_dilated = new_k

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """

        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated

        # TODO: regular forward, just like Conv2d().forward()

        # padding x with self.padding parameter (HINT: use np.pad())
        self.x = x
        pad = self.padding
        x = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], constant_values=0, mode='constant')
        self.padded_x = x

        bs = x.shape[0]
        ic = self.in_channel
        oc = self.out_channel
        ks = self.kernel_dilated

        dilated_filter = np.zeros((oc, ic, ks, ks))
        filtered_idx_r = np.array([i for i in range(self.W.shape[2] + (self.W.shape[2]-1)*(self.dilation-1)) if i % self.dilation == 0])
        filtered_idx_c = np.array([i for i in range(self.W.shape[3] + (self.W.shape[3]-1)*(self.dilation-1)) if i % self.dilation == 0])
        a, b = np.ix_(filtered_idx_r, filtered_idx_c)
        dilated_filter[:,:,a,b] = self.W
        self.W_dilated = dilated_filter
        input_width, input_height = x.shape[2], x.shape[3]

        output_width = (input_width - ks)//self.stride + 1
        output_height = (input_height - ks)//self.stride + 1

        ans = [[0] * output_width for _ in range(output_height)]
        for r in range(output_height):
            for c in range(output_width):
                offset_r, offset_c = r * self.stride, c * self.stride
                x_section = x[:,:,offset_r:offset_r+ks, offset_c:offset_c+ks]
                no_bias = np.tensordot(x_section, dilated_filter, axes=((1,2,3), (1,2,3))) # (batch_size, out_channel)
                ans[r][c] = no_bias + self.b.reshape(1, len(self.b))
        
        
        ans = np.transpose(np.array(ans), axes = (2, 3, 0, 1))
        return ans


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        bs = delta.shape[0]
        oc = delta.shape[1]
        ow = delta.shape[2]
        oh = delta.shape[3]

        dilated_delta = np.zeros((bs, oc, ow + (ow-1) * (self.stride-1), oh + (oh-1) * (self.stride-1)))
        filtered_idx_r = np.array([i for i in range(ow + (ow-1)*(self.stride-1)) if i % self.stride == 0])
        filtered_idx_c = np.array([i for i in range(oh + (oh-1)*(self.stride-1)) if i % self.stride == 0])
        x, y = np.ix_(filtered_idx_r, filtered_idx_c)
        dilated_delta[:,:,x,y] = delta

        # calculate wrt to filter
        for r in range(0, self.W_dilated.shape[2], self.dilation):
            for c in range(0, self.W_dilated.shape[3], self.dilation):
                x_section = self.padded_x[:,:,r:r+dilated_delta.shape[2],c:c+dilated_delta.shape[3]]
                dw_per_batch = np.tensordot(x_section, dilated_delta, axes=((0,2,3), (0,2,3))).T
                self.dW[:,:,r//self.dilation,c//self.dilation] += dw_per_batch

        # calculate wrt to bias
        self.db = delta.sum(axis=(0,2,3))

        # caluclate wrt to input
        ks = self.kernel_dilated
        padded_delta = np.pad(dilated_delta, [(0,0), (0,0), 
            (ks-1, ks-1 + (self.padded_x.shape[2] - ks) % self.stride),
            (ks-1, ks-1 + (self.padded_x.shape[3] - ks) % self.stride)],
            mode='constant', constant_values=0)

        flipped_filter = np.flip(np.flip(self.W_dilated, axis=2), axis=3)
        width_iters = padded_delta.shape[2]-flipped_filter.shape[2]+1
        height_iters = padded_delta.shape[3]-flipped_filter.shape[3]+1
        dx = [[0] * self.x.shape[2] for _ in range(self.x.shape[3])]
        pad = self.padding
        for r in range(width_iters):
            for c in range(height_iters):
                d_section = padded_delta[:,:,r:r+ks,c:c+ks]
                dx_per_batch = np.tensordot(d_section, flipped_filter, axes=((1,2,3), (0,2,3)))
                if pad <= r < width_iters - pad and pad <= c < height_iters - pad:
                    dx[r-pad][c-pad] = dx_per_batch

        dx = np.array(dx)
        dx = np.transpose(dx, axes = (2, 3, 0, 1))
        return dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        self.flatten_x = x.reshape(self.b, self.c * self.w)
        return self.flatten_x

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        self.twod_x = delta.reshape(self.b, self.c, self.w)
        return self.twod_x




