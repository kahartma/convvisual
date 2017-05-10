import lasagne
import numpy as np
import theano.tensor as T
import theano
import layers
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

def pad_tensor4D(tensor,paddings,fill_value=0):
    """Pad Tensor with different paddings at start and end of each dimension

    tensor: Tensor to be padded
    paddings: Size of paddings at start and end of each dimension (4x2)
    fill_value: Value to be used for padding (default: 0)

    Returns: Padded tensor
    """
    shape = tensor.shape
    for i,dim in enumerate(paddings):
        for j,num in enumerate(dim):
            if num>0:
                tmp_shape = T.set_subtensor(shape[i],num)

                fill_vals = T.alloc(fill_value, tmp_shape[0],tmp_shape[1],
                                                tmp_shape[2],tmp_shape[3])
                fill_vals = T.cast(fill_vals, theano.config.floatX)

                if j==0:
                    tensor = T.concatenate((fill_vals,tensor), axis=i)
                if j==1:
                    tensor = T.concatenate((tensor,fill_vals), axis=i)

                shape = tensor.shape

    return tensor

def crop_tensor4D(tensor,crops):
    """Crop Tensor with different crop sizes at start and end of each dimension

    tensor: Tensor to be cropped
    crops: Size of crops at start and end of each dimension (4x2)

    Returns: Cropped Tensor
    """
    shape = tensor.shape
    ind = [None]*len(crops)

    tensor = tensor[crops[0][0]:T.sub(shape[0],crops[0][1]),
                    crops[1][0]:T.sub(shape[1],crops[1][1]),
                    crops[2][0]:T.sub(shape[2],crops[2][1]),
                    crops[3][0]:T.sub(shape[3],crops[3][1])]
    return tensor



class DilationSeparate2DLayer(lasagne.layers.Layer):
    """Separation Layer for dilated convolution.
    Dilates the 3rd and 4th dimension into 1st dimension.

    incoming: incoming Layer
    block_shape: Size of dilation
    paddings: Paddings to be added to the incoming Layer (default: None).
                If None it's automatically padded to correct size.
    fill_value: Value for padding (default: 0)

    See: https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html#space_to_batch_nd
    """
    def __init__(self, incoming, block_shape, paddings=None, fill_value=0, **kwargs):
        # Implementation of
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html#space_to_batch_nd
        self.block_shape = block_shape
        self.paddings = paddings
        self.fill_value = fill_value
        
        #If no padding is given automatically create valid padding at end
        #of dimensions
        if self.paddings == None:
            self.paddings = ((0,incoming.output_shape[2]%block_shape[0]),
                             (0,incoming.output_shape[3]%block_shape[1]))

        super(DilationSeparate2DLayer,self).__init__(incoming, **kwargs)


    def get_output_for(self, input, **kwargs):
        input_shape = input.shape
        #B,F,R,C
        B = input_shape[0]
        F = input_shape[1]
        R = input_shape[2]
        C = input_shape[3]

        X = self.block_shape[0]
        Y = self.block_shape[1]

        PR = sum(self.paddings[0])
        PC = sum(self.paddings[1])

        RPR = R+PR
        CPC = C+PC

        input = input.dimshuffle(0,2,3,1)

        paddings = ((0,0),self.paddings[0],self.paddings[1],(0,0))
        input = pad_tensor4D(input,paddings,fill_value=self.fill_value)

        input = input.reshape((B,RPR/X,X,CPC/Y,Y,F))
        input = input.dimshuffle(2,4,0,1,3,5)
        input = input.reshape((X*Y*B,RPR/X,CPC/Y,F))
        input = input.dimshuffle(0,3,1,2)

        return input


    def get_output_shape_for(self, input_shape):
        B = input_shape[0]
        F = input_shape[1]
        R = input_shape[2]
        C = input_shape[3]

        X = self.block_shape[0]
        Y = self.block_shape[1]

        PR = sum(self.paddings[0])
        PC = sum(self.paddings[1])

        shape_B = None
        if B is not None:
            shape_B = B*X*Y
        output_shape = (shape_B,F,(R+PR)/X,(C+PC)/Y)
        return output_shape


class DilationMerge2DLayer(lasagne.layers.Layer):
    """Merge Layer for dilated convolution.
    Merges dilated values from the 1st into 3rd and 4th dimension.

    incoming: incoming Layer
    block_shape: Original size of dilation.
    crops: Cropping of padding that was added in the separation.

    See: https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html#batch_to_space_nd
    """
    def __init__(self, incoming, block_shape, crops, **kwargs):
        # Implementation of
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html#batch_to_space_nd
        self.block_shape = block_shape
        self.crops = crops

        super(DilationMerge2DLayer,self).__init__(incoming, **kwargs)


    def get_output_for(self, input, **kwargs):
        input_shape = input.shape
        #B,F,R,C
        B = input_shape[0]
        F = input_shape[1]
        R = input_shape[2]
        C = input_shape[3]

        X = self.block_shape[0]
        Y = self.block_shape[1]

        PR = sum(self.crops[0])
        PC = sum(self.crops[1])

        RPR = R-PR
        CPC = C-PC

        input = input.dimshuffle(0,2,3,1)

        input = input.reshape((X,Y,B/(X*Y),R,C,F))
        input = input.dimshuffle(2,3,0,4,1,5)
        input = input.reshape((B/(X*Y),R*X,C*Y,F))
        input = input.dimshuffle(0,3,1,2)

        crops = ((0,0),(0,0),self.crops[0],self.crops[1])
        input = crop_tensor4D(input,crops)

        return input


    def get_output_shape_for(self, input_shape):
        B = input_shape[0]
        F = input_shape[1]
        R = input_shape[2]
        C = input_shape[3]
        
        X = self.block_shape[0]
        Y = self.block_shape[1]

        PR = sum(self.crops[0])
        PC = sum(self.crops[1])

        shape_B = None
        if B is not None:
            shape_B = B/X/Y
        output_shape = (shape_B,F,R*X-PR,C*Y-PC)
            
        return output_shape


class Dummy2DLayer(lasagne.layers.Pool2DLayer):
    """Layer that does nothing.
    """
    def __init__(self, incoming, **kwargs):
        pool_size = 1

        super(Dummy2DLayer, self).__init__(incoming, pool_size, **kwargs)


class ReversedPool2DLayer(lasagne.layers.TransposedConv2DLayer):
    """Used for reversing 2D Pooling Layer
    Currently does not reverse mode used by original pooling layer (e.g. can't reverse max).
    Simply upscales original output to the size of the original input.

    CAREFUL: Currently uses TransposedConv2DLayer for the implementation. Because of that
    each filter is upscaled into all filters. Only use this if another convolutional layer
    follows. Else you will receive incorrect values! Use with caution.

    For Arguments see lasagne.layers.Pool2DLayer
    """
    def __init__(self, incoming, num_filters, pool_size,
                                stride, pad, ignore_border, output_size,
                                W=lasagne.init.Constant(1), mode='max',
                                flip_filters=False,
                                **kwargs):
        self.output_size = output_size
        self.ignore_border = ignore_border

        tmp_output_size = output_size
        if not ignore_border:
            tmp_output_size = None
        super(ReversedPool2DLayer, self).__init__(incoming, num_filters,
                pool_size, stride=stride,
                W=W, crop=pad, output_size=tmp_output_size,
                flip_filters=flip_filters, **kwargs)
 
        input_shape = incoming.output_shape
        output_shape = super(ReversedPool2DLayer, self).get_output_shape_for(input_shape)
        self.conv_output_shape = output_shape
        self.pool_output_size = output_size


    def get_output_for(self, input, **kwargs):
        output = super(ReversedPool2DLayer, self).get_output_for(input, **kwargs)
        print input.shape,self.stride,self.crop
        output_shape = super(ReversedPool2DLayer, self).get_output_shape_for(input.shape)
        conv_output_size = output_shape[2:]


        if np.array_equal(conv_output_size,self.pool_output_size):
            return output

        if not self.ignore_border:
            output = output[:,:,:self.pool_output_size[0],
                                :self.pool_output_size[1]]

        return output

    def get_output_shape_for(self, input_shape):
        output_shape = super(ReversedPool2DLayer, self).get_output_shape_for(input_shape)

        return_shape = (output_shape[0],output_shape[1],
                        self.pool_output_size[0],self.pool_output_size[1])

        return return_shape
    
    def convolve(self, input, **kwargs):
        # Adaptation from lasagne.layers.TransposedConv2DLayer.convolve
        # had to change the way the output shape is determined
        border_mode = 'half' if self.crop == 'same' else self.crop

        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.conv_output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.conv_output_shape[2:]
        conved = op(self.W, input, output_size)
        return conved

    def convolve2(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        output_size = self.conv_output_shape[2:]

        kshp = self.get_W_shape()
        imshp = self.conv_output_shape
        kshp[1] = 1
        imshp[1] = 1
        
        conved = T.alloc(0, self.output_shape[0],self.output_shape[1],
                                self.conv_output_shape[2],self.conv_output_shape[3])
        for i in self.output_shape[1]:
            op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=imshp,
            kshp=kshp,
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
            conved[:,i,:,:] = op(self.W[:,i,:,:], input[:,i,:,:], output_size)

        return conved



class ReversedConv2DLayer(lasagne.layers.TransposedConv2DLayer):
    """Used for reversing 2D Convolutional Layer
    Wrapper for lasagne.layers.TransposedConv2DLayer

    For Arguments see lasagne.layers.Conv2DLayer
    """
    def __init__(self, incoming, num_filters, filter_size,
                                stride, pad, output_size,
                                W=lasagne.init.Constant(1),
                                flip_filters=False, mode='max',
                                **kwargs):
        self.filter_size = filter_size
        super(ReversedConv2DLayer, self).__init__(incoming, num_filters,
                filter_size, stride=stride,
                W=W, crop=pad, output_size=output_size,
                flip_filters=flip_filters, **kwargs)

    
class ReversedFinalReshapeLayer(DilationSeparate2DLayer):
    """Used for reversing raindecode.veganlasagne.layers.FinalReshapeLayer

    Because FinalReshapeLayer automatically removes all padding from the dilation,
    for the reversion the block_shape and paddings from all previous (not merged) dilations are needed.

    incoming: Incoming Layer
    block_shape: block_shape from all previous, unmerged dilation layers
    paddings: paddings from all previous, unmerged dilation layers
    remove_invalids: If invalids were removed in FinalReshapeLayer (False not supported yet)
    flatten: If output was flattened in FinalReshapeLayer (True not supported yet)
    fill_value: Fill value used for padding (default: 0)
    """
    def __init__(self, incoming, block_shape, paddings,
                 remove_invalids, flatten, fill_value=0,
                 **kwargs):
        self.remove_invalids = remove_invalids
        self.flatten = flatten

        if not remove_invalids:
            raise Exception("Currently does not support inverse without removal of invalids")
        if flatten:
            raise Exception("No inverse for flattening supported. Set flattening to false and add another ReshapeLayer.")
        
        super(ReversedFinalReshapeLayer, self).__init__(incoming, block_shape, paddings, fill_value=fill_value, **kwargs)
            
    def get_output_for(self, input, **kwargs):
        return super(ReversedFinalReshapeLayer, self).get_output_for(input)


    def get_output_shape_for(self, input_shape): 
            
        return super(ReversedFinalReshapeLayer, self).get_output_shape_for(input_shape)         
