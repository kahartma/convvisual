import pickle
import numpy as np
from numpy.random import RandomState
import analysis
from braindecode.experiments.load import load_exp_and_model
import scipy
import convvisual.analysis.kuiper as kuiper


def create_flattened_featurearr(featurearrs,shape=(-1)):
    """Flattens list of multiple features into specific shape (mostly one long list of feature values)

    featurearrs: List of features
    shape: Shape into which each entry in featurearrs should be reshaped into (has to be applicable to all entries in featurearrs)

    Returns:
    Array of reshaped and concatenated feature entries
    """
    featurearr = np.array([])
    for featarr in featurearrs:
        featurearr = np.hstack([featurearr, np.reshape(featarr,shape)]) if featurearr.size else np.reshape(featarr,shape)

    return featurearr


def create_feature_index_labels(label,shape):
    """From labels and shapes create Arrays with label and entry index
    Only works for 1D and 2D

    label: Name of the feature
    shape: Shape of the feature entry

    Returns
    indeces: Array with index label for each entry in an array of shape
    indeces[i]=[label,[i]] or indeces[i,j]=[label,[i,j]]
    """
    indeces = np.empty(shape,dtype=object)
    for i in range(shape[0]):
        if len(shape)==2:
            for j in range(shape[1]):
                indeces[i,j] = [label,[i,j]]
        else:
            indeces[i] = [label,[i]]

    return indeces


def make_labels_from_index_labels(indexlabels,indexstrings):
    """Takes Indexlabel array (create_feature_index_labels) and replaces index values
    with corresponding names from indexstrings
    1D and 2D

    indexlabels: List of indexlabel arrays (flattened)
    indexstrings: Dict with names for indeces in indexlabels ([[i]] or [[i],[j]] for each indexlabel entry)

    Returns:
    List with label strings

    Example:
    indexlabels = [['Color',[0]],['Color',[1]],['Position',[0,1]],['Position',[1,1]]]
    indexstrings = ['Color':[['red','blue']],'Position':[['A','B'],['1','2']]]

    make_labels_from_index_labels(indexlabels,indexstrings)
    ['Color red','Color blue','Position A 2','Position B 2']
    """
    labels = list()
    for entry in indexlabels:
        if len(entry[1])>1:
            label = '%s %s %s'%(entry[0],indexstrings[entry[0]][0][entry[1][0]],indexstrings[entry[0]][1][entry[1][1]])
        else:
            label = '%s %s'%(entry[0],indexstrings[entry[0]][0][entry[1][0]])
        labels.append(label)

    return labels



def get_cropped_RF(RF_data,shape):
    """Take receptive field output and crop only the values lying in the RF from it

    RF_data: receptive field outputs
    shape: Shape used to reshape values
            E.g. [[0],128,-1,1]

    Returns:
    Cropped RF
    """
    X_RF = RF_data
    X_RF_cropped = X_RF[np.isnan(X_RF)==False]

    l_shape = list(shape)
    for i,entry in enumerate(l_shape):
        if type(entry) is list:
            assert len(entry)==1
            l_shape[i] = X_RF.shape[entry[0]]

    X_RF_cropped = X_RF_cropped.reshape(l_shape)

    return X_RF_cropped


def get_feature_vals(inputs,functions,**kwargs):
    """
    """
    features = list()
    for func in functions:
        features_entries = func(inputs,**kwargs)
            
        features.append(features_entries)

    return features


def cut_rand_windows(inputs,win_size,wins_per_input,rng=RandomState(1)):
    border_size = inputs.shape[2]-win_size
    assert border_size-wins_per_input>0

    win_indeces = range(inputs.shape[2]-win_size)
    if wins_per_input > len(win_indeces):
        wins_per_input = win_indeces
        print "To much wins_per_input for data. Changed it to max number of indeces (%d)"%wins_per_input
    
    ind_size = [inputs.shape[0],wins_per_input]
    ind_start = np.zeros(ind_size).astype(int)
    for i in range(inputs.shape[0]):
        ind_start[i] = rng.choice(win_indeces,size=(wins_per_input),replace=False)
    ind_end = ind_start + win_size
    print 'Ind_start',ind_start[[0,100,200,500,1000,1200]]
    
    ret = np.zeros((inputs.shape[0]*wins_per_input,inputs.shape[1],win_size,1))
    for i in range(inputs.shape[0]):
        for j in range(wins_per_input):
            tmp = inputs[i,:,ind_start[i,j]:ind_end[i,j]]
            ret[i*wins_per_input+j] = tmp
            
    return ret

def cut_ind_windows(inputs,win_size,win_indeces,wins_per_input=None,rng=RandomState(1)):
    border_size = inputs.shape[2]-win_size

    win_indeces = np.asarray(win_indeces)
    if wins_per_input is None:
        wins_per_input = len(win_indeces)
        
    assert np.all(win_indeces+win_size<=inputs.shape[2])
    
    ind_size = [inputs.shape[0],wins_per_input]
    ind_start = np.zeros(ind_size).astype(int)
    for i in range(inputs.shape[0]):
        ind_start[i] = rng.choice(win_indeces,size=(wins_per_input),replace=False)
    ind_end = ind_start + win_size
    
    ret = np.zeros((inputs.shape[0]*wins_per_input,inputs.shape[1],win_size,1))
    for i in range(inputs.shape[0]):
        for j in range(wins_per_input):
            tmp = inputs[i,:,ind_start[i,j]:ind_end[i,j]]
            ret[i*wins_per_input+j] = tmp
            
    return ret


def cut_all_windows(inputs,win_size):
    border_size = inputs.shape[2]-win_size

    win_indeces = np.arange(border_size)
    ind_start = win_indeces
    ind_end = ind_start+win_size

    ret = np.zeros((inputs.shape[0]*len(win_indeces),inputs.shape[1],win_size,1))
    for k in range(len(win_indeces)):
        tmp = inputs[:,:,ind_start[k]:ind_end[k]]
        ret[k*inputs.shape[0]:(k+1)*inputs.shape[0]] = tmp

    return ret

def get_dataset(modelpath):
    exp, model = load_exp_and_model(modelpath, set_invalid_to_NaN=False)

    datasets = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)
    
    return exp,model,datasets

def get_dataset_batches(exp,dataset,batch_size=999999,shuffle=False):
    exp.iterator.batch_size = batch_size
    batches = list(exp.iterator.get_batches(dataset, shuffle=shuffle))

    return batches


def extract_features_and_diff(X_class,X_baseline,sampling_rate,bonferroni_correction=True):
    features_class,feature_labels = extract_features(X_class,sampling_rate)
    features_base,feature_labels = extract_features(X_baseline,sampling_rate)
    
    feat_diff,feat_p = feat_diff_KS(features_class,features_base)
    if bonferroni_correction:
        feat_p *= len(feat_p)
    return feat_diff,feat_p,feature_labels,features_class,features_base


def feat_diff_KS(features_class,features_base):
    feat_diff = np.zeros((features_class.shape[1]))
    feat_p = np.zeros((features_class.shape[1]))
    for i in range(len(feat_diff)):
        feat_diff[i],feat_p[i] = scipy.stats.ks_2samp(features_class[:,i],features_base[:,i])

    return feat_diff,feat_p


def feat_diff_KS_kuiper(features_class,features_base):
    feat_diff = np.zeros((features_class.shape[1]))
    feat_p = np.zeros((features_class.shape[1]))
    for i in range(len(feat_diff)):
        feat_diff[i],feat_p[i] = kuiper.kuiper_two(features_class[:,i],features_base[:,i])

    return feat_diff,feat_p


def extract_features(X,sampling_rate,return_flattened_arr=True):
    feature_funcs = (analysis.get_frequency,analysis.get_frequency_change,analysis.get_offset,analysis.get_offset_change,analysis.get_phase_change,analysis.get_bandpower)
    feature_names = ('FFT','FFTc','Phase','Phasec','Mean','Meanc','Power')

    FFT,FFTc,mean,meanc,phasec,bandpower = get_feature_vals(X,feature_funcs,sampling_rate=sampling_rate)
    
    FFT = FFT[:,:,1:]
    FFTc = FFTc[:,:,1:]

    phase = np.angle(FFT)#[:,:,1:] #cut off phase 0Hz
    
    FFT = analysis.real_frequency(FFT)
    
    label_FFT = create_feature_index_labels(feature_names[0],FFT.shape[1:])
    label_FFTc = create_feature_index_labels(feature_names[1],FFTc.shape[1:])
    label_phase = create_feature_index_labels(feature_names[2],phase.shape[1:])
    label_phasec = create_feature_index_labels(feature_names[3],phasec.shape[1:])
    label_mean = create_feature_index_labels(feature_names[4],mean.shape[1:])
    label_meanc = create_feature_index_labels(feature_names[5],meanc.shape[1:])
    label_bandpower = create_feature_index_labels(feature_names[6],bandpower.shape[1:])
    
    features = [FFT,FFTc,phase,phasec,mean,meanc,bandpower]
    feature_labels = [label_FFT,label_FFTc,label_phase,label_phasec,label_mean,label_meanc,label_bandpower]
    
    if return_flattened_arr:
        features = create_flattened_featurearr(features,shape=(FFT.shape[0],-1))
        feature_labels = create_flattened_featurearr(feature_labels)
    
    return features,feature_labels,feature_names


def get_RF(RF_result,filt,reshape_channels,remove_shifted_trials=False):
    #max_units_in_filters = np.asarray(RF_result.max_units_in_filters)
    #filt_input_indeces = max_units_in_filters[:,1]==filt
    #max_units_in_filters = max_units_in_filters[filt_input_indeces]
    
    #X_RF_cropped = get_cropped_RF(RF_result.X_RF_complete[filt_input_indeces].squeeze(),([0],reshape_channels,-1))

    valid_indeces = np.arange(len(RF_result.X_RF_complete))
    s = 0
    if remove_shifted_trials:
        max_units_in_filters = np.asarray(RF_result.max_units_in_filters)
        remove_unit_inds = list()

        even_trial_inds = np.where(np.mod(max_units_in_filters[:,0],2)==0)[0]

        for even_ind in even_trial_inds:
            shifted_ind = np.where(max_units_in_filters[:,0]==(max_units_in_filters[even_ind,0]+1))[0]
            if len(shifted_ind)>0:
                s += 1
                if even_ind<shifted_ind[0]:
                    remove_unit_inds.append(shifted_ind[0])
                else:
                    remove_unit_inds.append(even_ind)
        valid_indeces = np.setdiff1d(np.arange(len(max_units_in_filters)),np.asarray(remove_unit_inds))

    X_RF_cropped = get_cropped_RF(RF_result.X_RF_complete[valid_indeces].squeeze(),([0],reshape_channels,-1))
    return X_RF_cropped


def get_RF_old(RF_result,filt,reshape_channels):
    
    max_units_in_filters = np.asarray(RF_result.max_units_in_filters)
    filt_input_indeces = max_units_in_filters[:,1]==filt
    max_units_in_filters = max_units_in_filters[filt_input_indeces]
    
    X_RF_cropped = get_cropped_RF(RF_result.RF_X[filt_input_indeces].squeeze(),([0],reshape_channels,-1))
    return X_RF_cropped


def cut_baseline(Inputs_baseline,wins_per_input,X_RF_shape,rng=RandomState(1)):
    X_baseline = cut_rand_windows(Inputs_baseline,X_RF_shape[2],wins_per_input).squeeze()
    print 'X_Shape',X_baseline.shape,X_RF_shape
    X_baseline = X_baseline.reshape((-1,X_RF_shape[1],X_RF_shape[2]))
    perm = rng.permutation(range(X_baseline.shape[0]))
    print 'Permutation1',perm.shape
    X_baseline = X_baseline[perm]
    return X_baseline


def get_FFT_Phase_band_means_KS_diff(signals_base,signals_class,sampling_rate,frequencies,freq_bands,median=True):
    FFT_base_i = analysis.get_frequency(signals_base,sampling_rate=sampling_rate)

    FFT_base = analysis.real_frequency(FFT_base_i)
    Phase_base = np.angle(FFT_base_i)[:,:,1:]

    if not median:
        FFT_base_mean = np.mean(FFT_base,axis=0)
    else:
        FFT_base_mean = np.median(FFT_base,axis=0)
    Phase_base_var = scipy.stats.circvar(Phase_base,axis=0)

    FFT_bands_base = analysis.get_band_means(FFT_base_mean,frequencies,freq_bands,median)
    Phase_bands_base = analysis.get_band_means(Phase_base_var,frequencies[1:],freq_bands,median)


    FFT_class_i = analysis.get_frequency(signals_class,sampling_rate=sampling_rate)

    FFT_class = analysis.real_frequency(FFT_class_i)
    Phase_class = np.angle(FFT_class_i)[:,:,1:]

    if not median:
        FFT_class_mean = np.mean(FFT_class,axis=0)
    else:
        FFT_class_mean = np.median(FFT_class,axis=0)
    Phase_class_var = scipy.stats.circvar(Phase_class,axis=0)

    FFT_bands_class = analysis.get_band_means(FFT_class_mean,frequencies,freq_bands,median)
    Phase_bands_class = analysis.get_band_means(Phase_class_var,frequencies[1:],freq_bands,median)

    FFT_KS = feat_diff_KS(FFT_base.reshape((FFT_base.shape[0],-1)),FFT_class.reshape((FFT_class.shape[0],-1)))[0].reshape(FFT_base.shape[1:])
    Phase_KS = feat_diff_KS(Phase_base.reshape((Phase_base.shape[0],-1)),Phase_class.reshape((Phase_class.shape[0],-1)))[0].reshape(Phase_base.shape[1:])

    FFT_KS_bands = analysis.get_band_means(FFT_KS,frequencies,freq_bands,median)
    Phase_KS_bands = analysis.get_band_means(Phase_KS,frequencies,freq_bands,median)

    FFT_mean_bands_ratio = np.log2(FFT_bands_class/FFT_bands_base)
    Phase_var_bands_ratio = np.log2(Phase_bands_class/Phase_bands_base)

    return FFT_mean_bands_ratio,Phase_var_bands_ratio,(FFT_KS_bands,Phase_KS_bands)


def calc_filt_variances(RF_Result,n_chans,n_class_inputs,median=True):
    filt_variances = np.zeros((len(RF_Result.max_filters)))
    for filt in RF_Result.max_filters:
        X_RF_cropped = get_RF(RF_Result,filt,n_chans)
        
        max_units_in_filters = np.asarray(RF_Result.max_units_in_filters)
        filt_input_indeces = max_units_in_filters[:,1]==filt
        max_units_in_filters = max_units_in_filters[filt_input_indeces]
        
        round_numbers = max_units_in_filters[np.mod(max_units_in_filters[:,0],2)==0,0]
        round_numbers += 1
        X_RF_cropped = X_RF_cropped[np.in1d(max_units_in_filters[:,0],round_numbers)==False]

        X_RF_cropped = X_RF_cropped[:n_class_inputs]
            
        if median:
            filt_variances[filt] = np.sum(np.square(np.median(X_RF_cropped,axis=0)))
        else:
            filt_variances[filt] = np.sum(np.square(np.mean(X_RF_cropped,axis=0)))
        
    filt_variances = filt_variances
    return filt_variances


def percentile_deviation(data,upper_percentile=68,axis=None):
    lower_percentile = 100-upper_percentile
    s1 = np.percentile(data,lower_percentile,axis=axis)
    s2 = np.percentile(data,upper_percentile,axis=axis)
    m = np.median(data,axis=axis)

    d_s1 = np.abs(m-s1)
    d_s2 = np.abs(m-s2)

    return (d_s1+d_s2)/2