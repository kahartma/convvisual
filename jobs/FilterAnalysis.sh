cd /home/hartmank/braindecode
source vienv/bin/activate
cd convvisual/notebooks
export THEANO_FLAGS=device=cpu,floatX=float64,unpickle_function=False,experimental.unpickle_gpu_on_cpu=True
export EEGCLASS=$SGE_TASK_ID
runipy 27_RF_Feature_Extraction_Rewrite_KSTest_Final-Copy1.ipynb out_notebooks/02_20_KSTest_Final_CL$SGE_TASK_ID.ipynb