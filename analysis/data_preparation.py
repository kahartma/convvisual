import numpy as np
import joblib
import os
import scipy

import convvisual.analysis.utils
import convvisual.analysis as can
import convvisual.receptive_field as crf


def create_baseline_data(savepath,layer_ind,wins_per_input,n_max_baseline,shape_RF,RF_Data=None,RF_Layer=None):
	if RF_Data is None:
		RF_Data = crf.receptive_field.load_ExtractorData(savepath)
	if RF_Layer is None:
		RF_Layer = crf.receptive_field.load_LayerData(savepath,layer_ind)

	data = {}
	data['X_baseline'] = can.utils.cut_baseline(RF_Data.inputs,wins_per_input,shape_RF)[:n_max_baseline]

	return data

def save_baseline_data(savepath,layer_ind,data):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'X_Baseline.Xdata')
	joblib.dump(data,fname,compress=True)

def load_baseline_data(savepath,layer_ind):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'X_Baseline.Xdata')
	return joblib.load(fname)


def create_RF_data(savepath,layer_ind,cl,filter_ind,reshape_chans,n_inputs):
	RF_Filter = crf.receptive_field.load_FilterData(savepath,layer_ind,cl,filter_ind)

	data = {}
	data['X_RF_cropped'] = can.utils.get_RF(RF_Filter,filter_ind,reshape_chans,remove_shifted_trials=True)[:n_inputs]
	data['max_units_in_filters'] = RF_Filter.max_units_in_filters

	return data

def save_RF_data(savepath,layer_ind,cl,filter_ind,data):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'Class_%s'%str(cl),'Filter_%d'%filter_ind,'X_RF.Xdata')
	joblib.dump(data,fname,compress=True)

def load_RF_data(savepath,layer_ind,cl,filter_ind):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'Class_%s'%str(cl),'Filter_%d'%filter_ind,'X_RF.Xdata')
	return joblib.load(fname)


def create_feature_data(X,sampling_rate):
	data = {}
	data['features'],data['feature_labels'],data['feature_names'] = can.utils.extract_features(X,sampling_rate,return_flattened_arr=False)

	return data

def save_baseline_feature_data(savepath,layer_ind,data):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'F_Baseline.Fdata')
	joblib.dump(data,fname,compress=True)

def load_baseline_feature_data(savepath,layer_ind):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'F_Baseline.Fdata')
	return joblib.load(fname)

def save_RF_feature_data(savepath,layer_ind,cl,filter_ind,data):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'Class_%s'%str(cl),'Filter_%d'%filter_ind,'F_RF.Fdata')
	joblib.dump(data,fname,compress=True)

def load_RF_feature_data(savepath,layer_ind,cl,filter_ind):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'Class_%s'%str(cl),'Filter_%d'%filter_ind,'F_RF.Fdata')
	return joblib.load(fname)


def create_KS_score_data(data_F_baseline,data_F_RF):
	data = {}
	for i,feat_name in enumerate(data_F_RF['feature_names']):
		feat_baseline = data_F_baseline['features'][i]
		feat_RF = data_F_RF['features'][i]
		print feat_baseline.shape,feat_RF.shape

		feat_baseline = can.utils.create_flattened_featurearr([feat_baseline],shape=(feat_baseline.shape[0],-1))
		feat_RF = can.utils.create_flattened_featurearr([feat_RF],shape=(feat_RF.shape[0],-1))

		print feat_baseline.shape,feat_RF.shape
		feat_diff,feat_p = can.utils.feat_diff_KS(feat_baseline,feat_RF)
		feat_diff_kui,feat_p_kui = can.utils.feat_diff_KS_kuiper(feat_baseline,feat_RF)

		data[feat_name] = {'KS':feat_diff,'p':feat_p,'KS_kuiper':feat_diff_kui,'p_kuiper':feat_p_kui}

	return data

def save_KS_score_data(savepath,layer_ind,cl,filter_ind,data):
	fname = fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'Class_%s'%str(cl),'Filter_%d'%filter_ind,'Score.KSdata')
	joblib.dump(data,fname,compress=True)

def load_KS_score_data(savepath,layer_ind,cl,filter_ind):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'Class_%s'%str(cl),'Filter_%d'%filter_ind,'Score.KSdata')
	return joblib.load(fname)


def run_preparation(datapath,perc_top_inputs,min_top_inputs,max_baseline_inputs,wins_per_input):
	print 'Start loading data'
	RF_Data = crf.receptive_field.load_ExtractorData(datapath)

	layer_indeces = RF_Data.layer_indeces
	classes = RF_Data.classes
	sampling_rate = RF_Data.sampling_rate
	n_chans = RF_Data.n_chans

	misc_data = {}
	misc_data['layer_indeces'] = layer_indeces
	misc_data['classes'] = classes
	misc_data['sampling_rate'] = sampling_rate
	misc_data['n_chans'] = n_chans
	misc_data['targets'] = RF_Data.targets
	misc_data['model'] = RF_Data.model
	misc_data['sensor_names'] = RF_Data.sensor_names
	misc_data['nUnits'] = RF_Data.nUnits
	misc_data['use_mean_filter_diff'] = RF_Data.use_mean_filter_diff

	misc_data['perc_top_inputs'] = perc_top_inputs
	misc_data['min_top_inputs'] = min_top_inputs
	misc_data['max_baseline_inputs'] = max_baseline_inputs
	misc_data['wins_per_input'] = wins_per_input
	save_misc_data(datapath,misc_data)

	for layer_ind in layer_indeces:
	    print 'Layer: %d'%layer_ind
	    
	    data_X_baseline = None
	             
	    for cl in classes:
	        print 'Class: %s'%str(cl)
	        RF_cl = crf.receptive_field.load_ClassData(datapath,layer_ind,cl)
	        max_filters = RF_cl.max_filters
	        n_input_indeces = len(RF_cl.input_indeces)
	        
	        n_inputs = int(n_input_indeces*perc_top_inputs)+1
	        if n_inputs < min_top_inputs:
	            n_inputs = min_top_inputs
	        del RF_cl
	           
	        for filter_ind in max_filters:
	            print 'Filter: %d'%filter_ind 
	            chans = n_chans
	            if layer_ind == 3:
	                chans = 1
	                
	            print 'X'
	            data_X_RF = create_RF_data(datapath,layer_ind,cl,filter_ind,chans,n_inputs)
	            save_RF_data(datapath,layer_ind,cl,filter_ind,data_X_RF)
	            
	            if data_X_baseline is None:
	                print 'Data baseline'
	                data_X_baseline = create_baseline_data(datapath,layer_ind,wins_per_input,max_baseline_inputs,data_X_RF['X_RF_cropped'].shape,RF_Data=RF_Data)
	                data_F_baseline = create_feature_data(data_X_baseline['X_baseline'],sampling_rate)

	                print 'Save baseline'
	                save_baseline_data(datapath,layer_ind,data_X_baseline)
	                save_baseline_feature_data(datapath,layer_ind,data_F_baseline)

	                frequencies = scipy.fftpack.fftfreq(data_X_baseline['X_baseline'].shape[2], 1./sampling_rate)
	                frequencies = frequencies[:frequencies.shape[0]/2].astype(str)[1:]

	                frequenciesc = scipy.fftpack.fftfreq(data_X_baseline['X_baseline'].shape[2]/2, 1./sampling_rate)
	                frequenciesc = frequencies[:frequenciesc.shape[0]/2].astype(str)[1:]

	                sensor_names = misc_data['sensor_names']
	                if chans == 1:
	                	sensor_names = ['N/A' for entry in sensor_names]

	                data_KS_labels = {}
	                data_KS_labels['feature_names'] = data_F_baseline['feature_names']
	                data_KS_labels['labels'] = list()
	                for i,feature_name in enumerate(data_KS_labels['feature_names']):
	                	tmp_list = can.utils.create_flattened_featurearr([data_F_baseline['feature_labels'][i]])
	                	labels = can.utils.make_labels_from_index_labels(tmp_list,
	                			{'FFT':[sensor_names,frequencies],
	                			'FFTc':[sensor_names,frequenciesc],
	                			'Phase':[sensor_names,frequencies],
	                			'Phasec':[sensor_names,frequenciesc],
	                			'Mean':[sensor_names],
	                			'Meanc':[sensor_names],
	                			'Power':[sensor_names]})
	                	data_KS_labels['labels'].append(labels)
	                save_labels_data(datapath,layer_ind,data_KS_labels)
	                
	            
	            print 'F'
	            data_F_RF = create_feature_data(data_X_RF['X_RF_cropped'],sampling_rate)
	            del data_X_RF
	            save_RF_feature_data(datapath,layer_ind,cl,filter_ind,data_F_RF)
	            
	            print 'KS'
	            data_KS = create_KS_score_data(data_F_baseline,data_F_RF)
	            del data_F_RF
	            
	            save_KS_score_data(datapath,layer_ind,cl,filter_ind,data_KS)
	            del data_KS

def save_misc_data(savepath,data):
	fname = os.path.join(savepath,'misc.IData')
	joblib.dump(data,fname,compress=True)
def save_labels_data(savepath,layer_ind,data):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'KS_labels.IData')
	joblib.dump(data,fname,compress=True)

def load_misc_data(savepath):
	fname = os.path.join(savepath,'misc.IData')
	return joblib.load(fname)
def load_labels_data(savepath,layer_ind):
	fname = os.path.join(savepath,'Layer_%02d'%layer_ind,'KS_labels.IData')
	return joblib.load(fname)