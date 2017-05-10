import lasagne
import pickle
import joblib
import os
from braindecode.experiments.load import load_exp_and_model
from braindecode.veganlasagne.layers import create_pred_fn
from convvisual import *
import sys
import gc


def make_and_save_RF_data(modelpath,savepath,classes,layer_indeces,
				finalReshapeAux,filter_batch_sizes=None,unit_batch_sizes=None,
				nUnits=50,nFilters=None,use_mean_filter_diff=False):
	sys.setrecursionlimit(10000)
	RFEData = ReceptiveFieldExtractorData(modelpath,savepath,classes,layer_indeces,
				nUnits=nUnits,nFilters=nFilters,use_mean_filter_diff=use_mean_filter_diff)

	if not os.path.isdir(savepath):
		os.makedirs(savepath)
	joblib.dump(RFEData,os.path.join(savepath,'RF_data.data'),compress=True)

	for i,layer_ind in enumerate(layer_indeces):
		fsplit = 1
		if filter_batch_sizes is not None:
			fsplit = filter_batch_sizes[i]
		RFELayer = ReceptiveFieldExtractorLayer(RFEData,layer_ind,finalReshapeAux[i],filter_batch_size=fsplit)

		layerpath = os.path.join(savepath,'Layer_%02d'%layer_ind)
		if not os.path.isdir(tmppath):
			os.makedirs (tmppath)
		joblib.dump(RFELayer,os.path.join(tmppath,'RF_layer.data'),compress=True)
		for cl in classes:
			usplit = 1
			if unit_batch_sizes is not None:
				usplit = unit_batch_sizes[i]
			RFEClass = ReceptiveFieldExtractorClass(RFEData,RFELayer,cl,unit_batch_size=usplit)

			classpath = os.path.join(layerpath,'Class_%s'%str(cl))
			if not os.path.isdir(classpath):
				os.makedirs (classpath)
			joblib.dump(RFEClass,os.path.join(classpath,'RF_class.data'),compress=True)

			del RFEClass
			gc.collect()

		del RFELayer
		gc.collect()


def load_ExtractorData(savepath):
	return joblib.load(os.path.join(savepath,'RF_data.data'))

def load_LayerData(savepath,layer_ind):
	return joblib.load(os.path.join(savepath,'Layer%02d'%layer_ind,'RF_layer.data'))

def load_ClassData(savepath,layer_ind,cl):
	return joblib.load(os.path.join(savepath,'Layer%02d'%layer_ind,'RF_class_%s.data'%str(cl)))


class ReceptiveFieldExtractorData:
	"""Helps extracting RF from Inputs in a network
	Main function: calc_max_RF_input

	modelpath: Path to the experiment model and load iputs from
	classes: Classes the network was trained on. Will create RF for class specific Inputs
				If None: Use Inputs for all classes
	n_Units: Number of units to be calculated for each filter (unique for each input, sorted by activity) (default: 50)
	"""

	def __init__(self,modelpath,savepath,classes,layer_indeces,
					nUnits=50,nFilters=None,use_mean_filter_diff=False):
		print 'Init extractor'
		exp, model = load_exp_and_model(modelpath, set_invalid_to_NaN=False)
		self.exp = exp
		self.model = model
		self.update_output = True

		exp.dataset.load()

		datasets = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)
		exp.iterator.batch_size = 999999 # dirty hack for simply getting all inputs
		test_batches = list(exp.iterator.get_batches(datasets['train'], shuffle=False))
		inputs,targets = test_batches[0]

		targets = targets.reshape((len(inputs),-1,4))
		targets = targets.sum(axis=1).argmax(axis=1)

		self.inputs = inputs.astype(np.float32)
		self.targets = targets.astype(np.int8)

		self.n_chans = lasagne.layers.get_all_layers(model)[0].shape[1]
		self.sensor_names = exp.dataset.test_set.sensor_names
		self.sampling_rate = exp.dataset.test_set.signal_processor.cnt_preprocessors[1][1]['newfs']

		self.classes = classes
		self.layer_indeces = layer_indeces
		self.nUnits = nUnits
		self.nFilters = nFilters
		self.use_mean_filter_diff = use_mean_filter_diff

		self.savepath = savepath


class ReceptiveFieldExtractorLayer:
	"""
	FinalReshapeAux: If there is a FinalReshapeLayer in the model, this is the dilation size
							If multiple dilations: multiplication of the dilation sizes (e.g. 3*3*3)
	"""

	def __init__(self,ExtractorData,layer_ind,finalReshapeAux,filter_batch_size=None):
		self.layer_ind = layer_ind

		self.model_tmp,self.layer_tmp = check_if_finalreshape_is_needed(ExtractorData.model,layer_ind)
		self.model_RF = receptive_field_build_deconv_layers(self.model_tmp[self.layer_tmp],
																self.model_tmp[1],
																use_learned_W=False,
																X_reshape=finalReshapeAux)
		self.pred_fn = create_pred_fn(self.model_tmp[self.layer_tmp])

		all_outputs = list()

		if filter_batch_size is None:
			filter_splits = 1
		else:
			filter_splits = len(ExtractorData.inputs)/filter_batch_size+1
		input_batches = np.array_split(np.arange(len(ExtractorData.inputs)),filter_splits)
		for batch in input_batches:
			if len(batch)==0:
				break
			tmp = self.pred_fn(list(ExtractorData.inputs[batch])).astype(np.float16)
			tmp[tmp<0]=0
			all_outputs.extend(tmp)
		self.outputs = np.asarray(all_outputs,dtype=np.float16)



class ReceptiveFieldExtractorClass:

	def __init__(self,ExtractorData,ExtractorLayer,cl,unit_batch_size=1):
		filt_means = ExtractorLayer.outputs.mean(axis=3).mean(axis=2).mean(axis=0)

		self.cl = cl
		self.input_indeces = np.arange(len(ExtractorData.targets))
		if cl is not None:
			self.input_indeces = np.where(ExtractorData.targets==cl)[0]
			outputs = ExtractorLayer.outputs[self.input_indeces]
			inputs = ExtractorData.inputs[self.input_indeces]
		else:
			outputs = ExtractorLayer.outputs
			inputs = ExtractorData.inputs

		self.targets = ExtractorData.targets[self.input_indeces]

		if ExtractorData.nFilters is None:
			n_filters = outputs.shape[1]
		else:
			n_filters = ExtractorData.nFilters

		self.filters_means = outputs.mean(axis=3).mean(axis=2).mean(axis=0)
		if ExtractorData.use_mean_filter_diff:
			self.filters_means = self.filters_means - filt_means
			self.filters_std = np.sqrt(outputs.var(axis=0).mean(axis=2).mean(axis=1))
			self.filters_means = np.divide(self.filters_means,self.filters_std)
			self.max_filters = self.filters_means.argsort(axis=0)[::-1][:n_filters]
			outputs = outputs[:,self.max_filters,:,:].astype(np.float16)
		else:
			self.max_filters = range(n_filters)

		self.max_units_in_filters = list()
		print 'Outputs',outputs.shape
		for i,filt in enumerate(self.max_filters):
			print 'Filter'
			tmp_units = get_most_active_units_in_layer_from_output(outputs[:,i:i+1,:,:],
										ExtractorLayer.model_RF,0,
										n_units=ExtractorData.nUnits,
										abs_act=False).astype(np.uint16)
			tmp_units[:,1] += i
			self.max_units_in_filters.extend(tmp_units)

		if unit_batch_size is None:
			unit_splits = 1
		else:
			unit_splits = len(self.max_units_in_filters)/unit_batch_size+1

		max_unit_batches = np.array_split(np.arange(len(self.max_units_in_filters)),unit_splits)

		self.X_RF_complete = np.array([],dtype=np.float32)
		self.max_units_in_filters = np.asarray(self.max_units_in_filters,dtype=np.uint16)

		print 'max units'
		print max_unit_batches
		for batch in max_unit_batches:
			print 'Batch'
			if len(batch)==0:
				break

			max_ind_ = self.max_units_in_filters[batch]
			max_ind_shape = max_ind_.shape

			X_RF_complete_,mask = get_receptive_field_masked_inputs(inputs,max_ind_,ExtractorLayer.model_RF)

			X_RF_complete_ = X_RF_complete_.astype(np.float32)
			X_RF_shape = X_RF_complete_.shape
			X_RF_complete_ = X_RF_complete_.reshape(max_ind_shape[0],-1,X_RF_shape[1],X_RF_shape[2],X_RF_shape[3])
			X_RF_complete_.shape

			self.X_RF_complete = np.vstack([self.X_RF_complete, X_RF_complete_]) if self.X_RF_complete.size else X_RF_complete_


class ReceptiveFieldExtractorFilter:
	def __init__(self,ExtractorData,ExtractorLayer,ExtractorClass,filter,unit_batch_size=1):
		filt_means = ExtractorLayer.outputs.mean(axis=3).mean(axis=2).mean(axis=0)