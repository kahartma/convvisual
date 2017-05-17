import numpy as np
import lasagne
import braindecode
from braindecode.veganlasagne import batch_norm
import convvisual.receptive_field.layers as layers
from theano import tensor as T, function, printing

def handle_switcher():
    return {
        lasagne.layers.conv.Conv2DLayer: handle_conv2d,
        braindecode.veganlasagne.layers.Conv2DAllColsLayer: handle_conv2d, #should work because AllColls simply calls Conv2DLayer
        lasagne.layers.pool.Pool2DLayer: handle_pool2d,
        lasagne.layers.DimshuffleLayer: handle_dimshuffle,
        lasagne.layers.ReshapeLayer: handle_reshape,
        lasagne.layers.BatchNormLayer: handle_dummy,
        lasagne.layers.DropoutLayer: handle_dummy,
        lasagne.layers.NonlinearityLayer: handle_dummy,
        braindecode.veganlasagne.layers.FinalReshapeLayer: handle_final_reshape,
        braindecode.veganlasagne.layers.StrideReshapeLayer: handle_stride_reshape,
        batch_norm.BatchNormLayer: handle_dummy
    }

def handle_conv2d(l_in, l_original, use_learned_W=True, n_filters=None, **kwargs):
    W = l_original.W if use_learned_W else lasagne.init.Constant(1)
    #b = l_original.b if use_learned_W else lasagne.init.Constant(0)

    if n_filters is None:
        n_filters = l_original.input_shape[1]

    l_deconv = layers.ReversedConv2DLayer(l_in, n_filters,
                                    l_original.filter_size, l_original.stride,
                                    l_original.pad,
                                    (l_original.input_shape[2],l_original.input_shape[3]),
                                    W=W, flip_filters=not l_original.flip_filters)

    return l_deconv

def handle_pool2d(l_in, l_original, n_filters=None, **kwargs):
    if n_filters is None:
        n_filters = l_original.input_shape[1]

    l_upscale = layers.ReversedPool2DLayer(l_in,n_filters,
                                        l_original.pool_size, l_original.stride,
                                        l_original.pad, l_original.ignore_border, (l_original.input_shape[2],l_original.input_shape[3]))

    return l_upscale

def handle_dimshuffle(l_in, l_original, **kwargs):
    pattern = l_original.pattern
    assert(len(pattern)==len(l_original.input_shape))

    reverse_pattern = np.zeros(len(pattern))
    for p in np.arange(len(pattern)):
        ind = np.where(pattern==p)
        assert(len(ind)==1)
        reverse_pattern[p] = ind[0]

    reverse_pattern = list(reverse_pattern.astype(int))
    l_reverse = lasagne.layers.DimshuffleLayer(l_in, reverse_pattern)
    return l_reverse

def handle_reshape(l_in, l_original, n_filters=None, **kwargs):
    if n_filters is None:
        n_filters = l_original.input_shape[1]

    shape = l_original.input_shape
    shape[1] = n_filters

    shape_new = list()
    for i in range(len(shape)):
        if shape[i]==None:
            shape_new.append([i])
        else:
            shape_new.append(shape[i])

    l_reshape = lasagne.layers.ReshapeLayer(l_in, shape_new)

    return l_reshape

def handle_dummy(l_in, l_original, **kwargs):
    #dummy function for layers that do not change size or
    #have any other effect on potential receptive field
    l_dummy = layers.Dummy2DLayer(l_in)

    return l_dummy

def handle_final_reshape(l_in, l_original, X_reshape=None, fill_value=0, n_filters=None, **kwargs):
    if l_original.flatten:
        raise Exception("No inverse for flattening supported. Set flattening to false and add another ReshapeLayer.")
        
    if n_filters is None:
        n_filters = l_original.input_shape[1]

    orig_in_shape = l_original.input_shape
    orig_out_shape = l_original.output_shape
    B = orig_in_shape[0]
    F = n_filters
    R = orig_in_shape[2]
    C = orig_in_shape[3]
    
    RX = orig_out_shape[2]
        
    X = np.ceil(RX/float(R))
    
    if X_reshape is not None:
        X = X_reshape
        
    block_shape = [int(X),1]
    paddings = [[0,int(R*X-RX)],[0,0]]
    
    l = layers.ReversedFinalReshapeLayer(l_in, block_shape, paddings,
                                         l_original.remove_invalids, l_original.flatten,
                                         fill_value=fill_value)
    return l

def handle_stride_reshape(l_in, l_original, n_filters=None, **kwargs):
    X = l_original.n_stride
        
    if n_filters is None:
        n_filters = l_original.input_shape[1]

    orig_in_shape = l_original.input_shape
    orig_out_shape = l_original.output_shape
    B = orig_in_shape[0]
    F = n_filters
    R = orig_in_shape[2]
    C = orig_in_shape[3]
    
    RP = orig_out_shape[2]*float(X)
    P = RP-R
            
    l = layers.DilationMerge2DLayer(l_in,(X,1),((0,int(P)),(0,0)))
    
    return l