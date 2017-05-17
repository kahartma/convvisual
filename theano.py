def init_theano(switch_cpu=False,debug=False):
    import os
    assert 'THEANO_FLAGS' in os.environ
    # in case you want to switch to cpu, to be able to use more than one notebook
    if switch_cpu:
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu,nvcc.fastmath=True'
    if debug:
    	os.environ['THEANO_FLAGS'] += ',optimizer=fast_compile,exception_verbosity=high'