import numpy as np
import theano
import theano.tensor as T
import lasagne
from numpy.random import RandomState

def get_mnist_data(mnist_file):
    import gzip
    import pickle

    train, val, test = pickle.load(gzip.open(mnist_file))

    return (train, val, test)


def create_two_digit_data(X_flat,y,rng=RandomState(1)):
    X_topo = X_flat.reshape(X_flat.shape[0], 28,28)

    random_inds = range(len(X_flat))
    rng.shuffle(random_inds)

    X_topo_both = np.concatenate([X_topo, X_topo[random_inds]], axis=2)
    X_flat_both = X_topo_both.reshape(X_topo_both.shape[0],-1)
    y = y.astype(np.int32)

    return (X_topo_both,X_flat_both,y)

def create_mnist_net_twodigit():
    l_in = lasagne.layers.InputLayer((None, 28*56),name="Input")

    l_shape = lasagne.layers.ReshapeLayer(l_in, (-1, 1, 28, 56),name="Input reshape")

    l_conv = lasagne.layers.Conv2DLayer(l_shape, num_filters=3, filter_size=3,
                                    nonlinearity=lasagne.nonlinearities.elu,
                                    name="L1")
    l_pool = lasagne.layers.Pool2DLayer(l_conv, pool_size=2, stride=2,
                                    name="P1")
    l_conv2 = lasagne.layers.Conv2DLayer(l_pool, num_filters=3, filter_size=2,
                                    nonlinearity=lasagne.nonlinearities.elu,
                                    name="L2")
    l_pool2 = lasagne.layers.Pool2DLayer(l_conv2, pool_size=2, stride=2,
                                    name="P2")
    l_conv3 = lasagne.layers.Conv2DLayer(l_pool2, num_filters=3, filter_size=(3,2),
                                    nonlinearity=lasagne.nonlinearities.elu,
                                    name="L3")
    l_pool3 = lasagne.layers.Pool2DLayer(l_conv3, pool_size=2, stride=2,
                                    name="P3")

    l_out = lasagne.layers.Conv2DLayer(l_pool3, num_filters=10, filter_size=(2,6),name="Dense Output")

    l_out_shape = lasagne.layers.FlattenLayer(l_out, 2,name="Flat Output")
    l_out_final = lasagne.layers.NonlinearityLayer(l_out_shape,
                                               nonlinearity=lasagne.nonlinearities.softmax,
                                               name="Softmax Output")

    return l_out_final


def train_mnist(l_out, X_train, y_train, n_epochs=20, learning_rate=0.005, batch_size=1000,
                log=None,X_val=None,y_val=None, rng=RandomState(1)):
    from convvisual import get_balanced_batches, log_loss_acc

    # Compile and train the network.
    X_sym = T.matrix()
    y_sym = T.ivector()

    # not strictly necessary in this case but just to encourage good practice
    # I use deterministic=True for test preds
    # and false for train preds, this is necessary for dropout etc.
    train_output = lasagne.layers.get_output(l_out, X_sym, deterministic=False)
    test_output = lasagne.layers.get_output(l_out, X_sym, deterministic=True)
    train_pred = train_output.argmax(-1)
    test_pred = test_output.argmax(-1)

    train_loss = T.mean(lasagne.objectives.categorical_crossentropy(train_output, y_sym))
    test_loss = T.mean(lasagne.objectives.categorical_crossentropy(test_output, y_sym))

    test_acc = T.mean(T.eq(test_pred, y_sym))

    params = lasagne.layers.get_all_params(l_out)

    grad = T.grad(train_loss, params)
    updates = lasagne.updates.adam(grad, params, learning_rate=learning_rate)

    f_train = theano.function([X_sym, y_sym], updates=updates)
    f_val = theano.function([X_sym, y_sym], [test_loss, test_acc])
    f_output = theano.function([X_sym], test_output)



    if log!=None:
        log.info("Epoch: {:d}".format(0))
        log.info("Train:")
        log_loss_acc(X_train,y_train,f_val,log,rng)

        if X_val!=None:
            log.info("Validation:")
            log_loss_acc(X_val,y_val,f_val,log,rng)

    for i_epoch in range(n_epochs):
        train_batches_inds = get_balanced_batches(len(X_train), rng,batch_size=batch_size, shuffle=True)
        for batch_inds in train_batches_inds:
            X_batch = X_train[batch_inds]
            y_batch = y_train[batch_inds]
            f_train(X_batch, y_batch)

        if log!=None:
            log.info("Epoch: {:d}".format(i_epoch+1))
            log.info("Train:")
            log_loss_acc(X_train,y_train,f_val,log,rng)

            if X_val!=None:
                log.info("Validation:")
                log_loss_acc(X_val,y_val,f_val,log,rng)
