import lasagne
import theano
import theano.tensor as T
import numpy as np
from braindecode.veganlasagne.layers import StrideReshapeLayer, FinalReshapeLayer
from braindecode.test.util import to_4d_time_array, equal_without_nans,\
    allclose_without_nans
    
from lasagne.nonlinearities import softmax, identity
from numpy.random import RandomState
from braindecode.veganlasagne.pool import SumPool2dLayer
from braindecode.veganlasagne.nonlinearities import safe_log

from layers import DilationSeparate2DLayer,DilationMerge2DLayer

def test_stride_reshape_layer():
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,15,1], input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    #network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = DilationSeparate2DLayer(network,(2,1),((0,1),(0,0)),fill_value=np.nan)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[2, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    #network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = DilationSeparate2DLayer(network,(2,1),((0,0),(0,0)),fill_value=np.nan)
    network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=[2, 1],
                                         W=to_4d_time_array([[1,1], [-1,-1], [0.1,0.1], [-0.1,-0.1]]), stride=(1,1),
                                        nonlinearity=lasagne.nonlinearities.identity)
    #network = FinalReshapeLayer(network, remove_invalids=False)
    network = DilationMerge2DLayer(network,(4,1),((0,0),(0,0)),flatten=True)
    
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network)[1:])
    pred_cnt_func = theano.function([input_var], preds_cnt)
    layer_activations = pred_cnt_func(to_4d_time_array([range(1,16), range(16,31)]))
    assert equal_without_nans(np.array([[[[  6.], [  9.], [ 12.], [ 15.], [ 18.], [ 21.], 
                           [ 24.], [ 27.], [ 30.], [ 33.], [ 36.], [ 39.], 
                           [ 42.]]],
        [[[ 51.], [ 54.], [ 57.], [ 60.], [ 63.], [ 66.], 
          [ 69.], [ 72.], [75.], [ 78.], [ 81.], [ 84.], 
          [ 87.]]]], 
            dtype=np.float32),
        layer_activations[0])

    assert equal_without_nans(np.array(
        [[[[  6.], [ 12.], [ 18.], [ 24.], [ 30.], [ 36.], [ 42.]]],
       [[[ 51.], [ 57.], [ 63.], [ 69.], [ 75.], [ 81.], [ 87.]]],
       [[[  9.], [ 15.], [ 21.], [ 27.], [ 33.], [ 39.], [ np.nan]]],
       [[[ 54.], [ 60.], [ 66.], [ 72.], [ 78.], [ 84.], [ np.nan]]]],
       dtype=np.float32),
       layer_activations[1])
    
    assert equal_without_nans(np.array([[[[  18.], [  30.], [  42.], [  54.], [  66.], [  78.]]],
       [[[ 108.], [ 120.], [ 132.], [ 144.], [ 156.], [ 168.]]],
       [[[  24.], [  36.], [  48.], [  60.], [  72.], [  np.nan]]],
       [[[ 114.], [ 126.], [ 138.], [ 150.], [ 162.], [  np.nan]]]],
       dtype=np.float32),
       layer_activations[2])
    
    assert allclose_without_nans(np.array(
        [[[[  60.        ], [ 108.        ]], [[ -60.        ], [-108.        ]],
        [[   6.00000048], [  10.80000019]], [[  -6.00000048], [ -10.80000019]]],
        [[[ 240.        ], [ 288.        ]], [[-240.        ], [-288.        ]],
        [[  24.        ], [  28.80000114]], [[ -24.        ], [ -28.80000114]]],
        [[[  72.        ], [ 120.        ]], [[ -72.        ], [-120.        ]],
        [[   7.20000029], [  12.        ]], [[  -7.20000029], [ -12.        ]]],
        [[[ 252.        ], [ 300.        ]], [[-252.        ], [-300.        ]],
        [[  25.20000076], [  30.00000191]], [[ -25.20000076], [ -30.00000191]]],
        [[[  84.        ], [ 132.        ]], [[ -84.        ], [-132.        ]],
        [[   8.40000057], [  13.19999981]], [[  -8.40000057], [ -13.19999981]]],
        [[[ 264.        ], [ 312.        ]], [[-264.        ], [-312.        ]],
        [[  26.40000153], [  31.20000076]], [[ -26.40000153], [ -31.20000076]]],
        [[[  96.        ], [          np.nan]], [[ -96.        ], [          np.nan]],
        [[   9.60000038], [          np.nan]], [[  -9.60000038], [          np.nan]]],
        [[[ 276.        ], [          np.nan]], [[-276.        ], [          np.nan]],
        [[  27.60000038], [          np.nan]], [[ -27.60000038], [          np.nan]]]],
        dtype=np.float32),
        layer_activations[4])
    
    assert allclose_without_nans(np.array(
        [[  60.        ,  -60.        ,    6.0,   -6.],
        [  72.        ,  -72.        ,    7.2,   -7.2],
        [  84.        ,  -84.        ,    8.4,   -8.4],
        [  96.        ,  -96.        ,    9.6,   -9.6],
        [ 108.        , -108.        ,   10.8,  -10.8],
        [ 120.        , -120.        ,   12. ,  -12.        ],
        [ 132.        , -132.        ,   13.2,  -13.2],
        [          np.nan,           np.nan,           np.nan,           np.nan],
        [ 240.        , -240.        ,   24.        ,  -24.        ],
        [ 252.        , -252.        ,   25.2,  -25.2],
        [ 264.        , -264.        ,   26.4,  -26.4],
        [ 276.        , -276.        ,   27.6,  -27.6],
        [ 288.        , -288.        ,   28.8,  -28.8],
        [ 300.        , -300.        ,   30.0,  -30.0],
        [ 312.        , -312.        ,   31.2,  -31.2],
        [          np.nan,           np.nan,           np.nan,           np.nan]],
        dtype=np.float32),
        layer_activations[5])
    
def test_dilation_layer():
    from convvisual import receptive_field_build_deconv_layers
    from braindecode.veganlasagne.layer_util import print_layers
    
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,15,1], input_var=input_var)
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = FinalReshapeLayer(network, remove_invalids=True, flatten=False)
    print_layers(network)

    network2 = lasagne.layers.InputLayer(shape=[None,1,15,1], input_var=input_var)
    network2 = DilationSeparate2DLayer(network,(2,1),((0,1),(0,0)),fill_value=np.nan)
    network2 = DilationSeparate2DLayer(network2,(2,1),((0,0),(0,0)),fill_value=np.nan)
    network2 = DilationMerge2DLayer(network2,(4,1),((0,1),(0,0)))
    print_layers(network2)
    
    input = to_4d_time_array([range(1,16), range(16,31)])
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network))
    pred_cnt_func = theano.function([input_var], preds_cnt)
    layer_activations = pred_cnt_func(input)
    print layer_activations
    
    
    input = to_4d_time_array([range(1,16), range(16,31)])
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network2))
    pred_cnt_func = theano.function([input_var], preds_cnt)
    layer_activations2 = pred_cnt_func(input)
    print layer_activations2

    assert(np.array_equal(layer_activations2[3],layer_activations[3]))

def test_inverse_dilation_layer():
    from convvisual import receptive_field_build_deconv_layers
    from braindecode.veganlasagne.layer_util import print_layers
    
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network1 = lasagne.layers.InputLayer(shape=[None,1,15,1], input_var=input_var)
    network2 = StrideReshapeLayer(network1, n_stride=2, invalid_fill_value=np.nan)
    network3 = StrideReshapeLayer(network2, n_stride=2, invalid_fill_value=np.nan)
    network = FinalReshapeLayer(network3, remove_invalids=True, flatten=False)
    print_layers(network)

    #network = DilationSeparate2DLayer(network,(4,1),((0,1),(0,0)),fill_value=np.nan)
    #network = DilationMerge2DLayer(network,(2,1),((0,0),(0,0)))
    #network = DilationMerge2DLayer(network,(2,1),((0,1),(0,0)))
    deconv_network = receptive_field_build_deconv_layers(network,network2)
    print_layers(deconv_network)
    
    input = to_4d_time_array([range(1,16), range(16,31)])
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network))
    pred_cnt_func = theano.function([input_var], preds_cnt)
    layer_activations = pred_cnt_func(input)
    print layer_activations
    
    
    layer_activations2 = lasagne.layers.get_output(lasagne.layers.get_all_layers(deconv_network),layer_activations[-1])
    print layer_activations[-1].shape
    print layer_activations2[3].eval()

    assert(np.array_equal(layer_activations2[3].eval(),layer_activations[0]))