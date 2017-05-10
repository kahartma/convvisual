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

def test_inv_conv():
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,10,5], input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 2],
                                         W=lasagne.init.Constant(1), stride=(2,5))
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    
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

    