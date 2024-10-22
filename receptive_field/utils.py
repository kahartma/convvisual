import convvisual.receptive_field.layers as layers
from convvisual.receptive_field.layer_handlers import handle_switcher

import numpy as np
import theano
import theano.tensor as T
import lasagne
import braindecode
from braindecode.veganlasagne import batch_norm
from braindecode.veganlasagne.layers import create_pred_fn





def receptive_field_build_deconv_layers(top_layer,last_layer,n_filters=None,  **kwargs):
    """Create network structure for calculating receptive fields of units in a certain layer.

    Parameters
    ----------
    top_layer: Layer that contains the units of interest
    last_layer: Layer for which the RF should be computed (e.g. input layer)
        
    use_learned_W: Use learned weights from a trained network to create RF (default=False)

    Returns
    -------
    RF_layer: Output layer in which the RF lies (e.g. instead of original input layer)

    """
    #backwards operations for different types of layers
    switcher = handle_switcher()
    
    #Get all layers from the output layer we want RF from and exclude original input layer
    lower_layers = lasagne.layers.get_all_layers(top_layer,treat_as_input=[last_layer])

    shape = list(top_layer.output_shape)
    if n_filters is not None:
        shape[1] = n_filters

    l_in = lasagne.layers.InputLayer(shape) #Replace output layer for RF with new input layer
    RF_layer = l_in
    #Go backwards through all layers and apply corresponding backwards operation
    for curr_layer in lower_layers[::-1]:
        layer_type =  type(curr_layer)
        func = switcher.get(layer_type)
        RF_layer = func(RF_layer,curr_layer, n_filters=n_filters, **kwargs)
        #assert np.array_equal(RF_layer.output_shape,curr_layer.input_shape),"Something went wrong here"
        
    return RF_layer

def get_receptive_field_mask(Units,l_RF,combined_units=False):
    """Get receptive field of the specified units.
    Only works for 4D layers

    Units: Active units in input layer of l_RF (Nx4) Input|Filter|X|Y
    l_RF: Receptive field layer (original input layer)
    combined_units: Separate receptive fields for each unit or combined for all (default: False)

    Returns:
    RF_output: Activation of units in the receptive field layer.
                Activation > 0: unit lies in receptive field
    """
    bot_layer = l_RF
    top_layer = lasagne.layers.get_all_layers(l_RF)[0]

    assert (len(top_layer.shape)==4 and len(bot_layer.output_shape)==4),"Currently only works for 4D tensors"

    if Units.shape[1]==4:
        Units = Units[:,1:]

    n_Units = len(Units)

    in_indcs = np.arange(n_Units)
    if combined_units:
        in_indcs = 0

    In_Units = np.zeros((n_Units,
                    top_layer.shape[1],top_layer.shape[2],top_layer.shape[3]),
                    dtype=np.float32)
    In_Units[in_indcs,Units[:,0],Units[:,1],Units[:,2]] = 1

    assert np.array_equal(top_layer.shape[1:],In_Units.shape[1:]),"Weird"

    pred_fn = create_pred_fn(bot_layer)
    RF_output = pred_fn(In_Units)

    return RF_output

def get_receptive_field_masked_inputs(Inputs,Units,l_RF,fill_value=np.nan):
    """Get input that activated a unit and only keep the values in its receptive field.

    Inputs: Inputs that were used to calculated unit activations
    Units: Active units in input layer of l_RF (Nx4) Input|Filter|X|Y
    l_RF: Receptive field layer (original input layer)
    fill_value: Value that is used for values not lying in the receptive field (default: NaN)

    Returns:
    X_RF_complete: Outputs size of the receptive field layer
    mask: Mask for the receptive fields of the units in receptive field layer
    """
    mask = get_receptive_field_mask(Units,l_RF)
    mask = mask>0

    Inputs = Inputs[Units[:,0],:,:,:]
    X_RF_complete = np.empty(Input.shape)
    X_RF_complete[:] = fill_value
    X_RF = Input[mask]
    X_RF_complete[mask] = X_RF.flatten()

    return X_RF_complete,mask

def get_most_active_units_in_layer(Inputs,layers,layer_ind,filter,n_units=None,unique=True,abs_act=False):
    """Get most active (sorted) units in specific layer for inputs.

    Inputs: Inputs to calculate unit activations with
    layers: Complete list of network layers
    layer_ind: Index in layers in which unit activation should be computed
    filter: Specific filter(s) in which the units are. If None: All filters in layer
    n_Units: Number of most active units. If None: all units sorted (default: None)
    unique: If only the most active Unit for each input should be kept. (default: True)
    abs_act: If absolute unit activity should be used. (default: False)

    Returns:
    Most active units (Nx4) Input|Filter|X|Y
    """

    if layers is not list:
        layers = lasagne.layers.get_all_layers(layers)
    if filter==None:
        filter = np.arange(layers[layer_ind].output_shape[1])

    pred_fn = create_pred_fn(layers)
    output = pred_fn(Inputs)

    return get_most_active_units_in_layer_from_output(output,layer_ind,filter,n_units=n_units,unique=unique,abs_act=abs_act)


def get_most_active_units_in_layer_from_output(Outputs,layer_ind,filter,n_units=None,unique=True,abs_act=False):
    """Get most active (sorted) units in specific layer from already computed outputs.

    Outputs: Outputs to get the most active units from. Can be list of outputs (for all layers)
    layers: Complete list of network layers
    layer_ind: Index in layers in which unit activation should be computed
    filter: Specific filter(s) in which the units are. If None: All filters in layer
    n_Units: Number of most active units. If None: all units sorted (default: None)
    unique: If only the most active Unit for each input should be kept. (default: True)
    abs_act: If absolute unit activity should be used. (default: False)

    Returns:
    Most active units (Nx4) Input|Filter|X|Y
    """

    if type(Outputs) is list:
        Outputs = np.copy(Outputs[layer_ind])

    assert (len(Outputs.shape)==4)

    if abs_act:
        Outputs = abs(Outputs)

    mask = np.ones(Outputs.shape)
    mask[:,filter,:,:] = 0
    Outputs[mask==1] = 0
    del mask

    output_sorted = Outputs.argsort(axis=None)[::-1]
    output_sorted_ind = np.unravel_index(output_sorted,Outputs.shape)
    unique_ind = np.arange(len(output_sorted_ind[0]))
    if unique:
        a,unique_ind = np.unique(output_sorted_ind[0],return_index=True)
        unique_ind = sorted(unique_ind)

    if n_units==None:
        n_units = len(max_act)
    output_sorted_ind = np.asarray(output_sorted_ind).T
    Units = output_sorted_ind[unique_ind[:n_units],:]

    return Units

def check_if_finalreshape_is_needed(model,layer_ind):
    """Checks if there is an unmerged occurence of braindecode.veganlasagne.layers.StrideReshapeLayer
    If so adds an instance of braindecode.veganlasagne.layers.FinalReshapeLayer at the end of the network.

    model: Network model
    layer_ind: Until which layer of the mode should be checked (new output layer)

    Returns:
    model: Model with FinalReshapeLayer if needed. New output layer is layer_ind or layer_ind+1 (if added layer)
    layer_ind: New output layer index
    """
    if model is not list:
        model = lasagne.layers.get_all_layers(model)
        
    model = model[:layer_ind+1]
    need_reshape = False
    for layer in model:
        layer_type =  type(layer)
        if layer_type is braindecode.veganlasagne.layers.StrideReshapeLayer:
            need_reshape = True
        if layer_type is braindecode.veganlasagne.layers.FinalReshapeLayer:
            need_reshape = False
        
    model = model[-1]
    if need_reshape:
        model = braindecode.veganlasagne.layers.FinalReshapeLayer(model,flatten=False)
        layer_ind += 1
        
        
    model = lasagne.layers.get_all_layers(model)
    return model,layer_ind

        

def get_balanced_batches(n_trials, rng, shuffle, n_batches=None, batch_size=None):
    """Create indices for batches balanced in size (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng :
        
    shuffle :
        Whether to shuffle indices before splitting set.
    n_batches :
         (Default value = None)
    batch_size :
         (Default value = None)

    Returns
    -------

    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))
    
    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial =  n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_trial = 0
    end_trial = 0
    batches = []
    for i_batch in xrange(n_batches):
        end_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            end_trial += 1
        batch_inds = all_inds[range(i_trial, end_trial)]
        batches.append(batch_inds)
        i_trial = end_trial
    assert i_trial == n_trials
    return batches


def log_loss_acc(X_flat,y,f_val,log,rng):
    batches_inds_sorted = get_balanced_batches(len(X_flat),rng,batch_size=1000, shuffle=False)
    loss_sum = 0
    acc_sum = 0
    for batch_inds in batches_inds_sorted:
        X_batch = X_flat[batch_inds]
        y_batch = y[batch_inds]
        loss, acc = f_val(X_batch, y_batch)
        loss_sum += (loss * len(batch_inds))
        acc_sum += (acc * len(batch_inds))
    loss = loss_sum / float(len(X_flat))
    acc= acc_sum / float(len(X_flat))
    log.info("loss: {:.4f}".format(loss))
    log.info("acc: {:.2f}".format(acc))
