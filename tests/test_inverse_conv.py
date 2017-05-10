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

from convvisual.receptive_field.layers import ReversedConv2DLayer

def test_inv_conv():
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,7,7], input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1),pad='same')
    conv1 = network
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 3],
                                         W=lasagne.init.Constant(1), stride=(2,2),pad='same')
    conv2 = network
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1),pad='same')
    conv3 = network
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1),pad='same')
    conv4 = network

    network = ReversedConv2DLayer(network,1, (3,1),
                                (1,1),'same',conv4.input_shape[2:4])
    network = ReversedConv2DLayer(network,1, (3,1),
                                (1,1),'same',conv3.input_shape[2:4])
    network = ReversedConv2DLayer(network,1, (3,3),
                                (2,2),'same',conv2.input_shape[2:4])
    network = ReversedConv2DLayer(network,1, (3,1),
                                (1,1),'same',conv1.input_shape[2:4])
    
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network))
    pred_cnt_func = theano.function([input_var], preds_cnt)
    inputs = ((np.diag(range(7))+1)[np.newaxis,:,:], (np.diag(range(7))+5)[np.newaxis,:,:])
    layer_activations = pred_cnt_func(inputs)
    print preds_cnt
    print 0,layer_activations[0]
    print 1,layer_activations[1]
    print 2,layer_activations[2]
    print 3,layer_activations[3]
    print 4,layer_activations[4]
    print 5,layer_activations[5]
    print 6,layer_activations[6]
    print 7,layer_activations[7]
    print 8,layer_activations[8]


    assert equal_without_nans(np.array([[[[  6.], [  9.], [ 12.], [ 15.], [ 18.], [ 21.], 
                           [ 24.], [ 27.], [ 30.], [ 33.], [ 36.], [ 39.], 
                           [ 42.]]],
        [[[ 51.], [ 54.], [ 57.], [ 60.], [ 63.], [ 66.], 
          [ 69.], [ 72.], [75.], [ 78.], [ 81.], [ 84.], 
          [ 87.]]]], 
            dtype=np.float32),
        layer_activations[0])

    