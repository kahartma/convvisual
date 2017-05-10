import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_max_filters(RF_Result):
    print 'Max filters: ',RF_Result.max_filters
    sns.heatmap(np.reshape(RF_Result.filters_means,(-1,1)).T,
                linewidths=.5)
    plt.xlabel('Filter #')
    plt.ylabel('Activation Diff')
    plt.title('Mean activation for feature maps over all inputs')
    plt.show()
    
    
def print_features(score,p,labels,indeces):
    for idx in indeces:
        print 'Score %f  p %f  : %s'%(score[idx],p[idx],labels[idx])
        
        
def plot_RF_starts(RF_Data,RF_results,win_size,filt):
    max_units_in_filters = np.asarray(RF_results.max_units_in_filters)
    max_units_in_filters = max_units_in_filters[max_units_in_filters[:,1]==filt]
    sns.distplot(max_units_in_filters[:,2],bins=np.arange(0,RF_Data.inputs.shape[2],20),kde=False)
    plt.xlim([1,RF_Data.inputs.shape[2]-win_size])
    plt.xlabel('Sample #')
    plt.ylabel('Number of receptive fields starting at sample')
    plt.title('Starting of the RF windows in complete input signal')
    plt.show()
    
    
def plot_channels(RF_results,sensor_names,filt,n_chans):
    max_units_in_filters = np.asarray(RF_results.max_units_in_filters)
    max_units_in_filters = max_units_in_filters[max_units_in_filters[:,1]==filt]
    chan_counts = np.bincount(max_units_in_filters[:,3])
    max_chan = chan_counts.argmax()
    max_name = sensor_names[max_chan]
    plt.figure(figsize=(10, 10))
    sns.distplot(max_units_in_filters[:,3],bins=np.arange(n_chans),kde=False)
    plt.xlim([0,n_chans])
    plt.ylabel('Number of times signal window originated in channel')
    plt.xlabel('Channel')
    plt.title('Channels that provided maximized input %s: %d'%(max_name,chan_counts[max_chan]))
    plt.xticks(range(1,len(sensor_names)+1),sensor_names,rotation=90,fontsize=5)
    plt.show()
        
        
def plot_channel_avg(X_RF_cropped,channel,title=''):
    sns.tsplot(data=X_RF_cropped[:,channel],err_style="unit_points")
    m = X_RF_cropped[:,channel].mean(axis=0)
    s = X_RF_cropped[:,channel].std(axis=0)
    plt.fill_between(np.arange(X_RF_cropped.shape[2]),m-s,m+s,color='r',zorder=100,alpha=0.3) 
    plt.plot(np.arange(X_RF_cropped.shape[2]),m,color='r',zorder=101) 
    plt.xlabel('Sample # (250Hz)')
    plt.ylabel('Amplitude')
    plt.title(title)
    

def plot_channel_avg_comparison(X1,X2,channel):
    plt.figure()
    #sns.tsplot(data=X_RF_cropped[:,channel],err_style="unit_traces")
    sns.tsplot(data=X1[:,channel],err_style="unit_traces")
    sns.tsplot(data=X2[:,channel],err_style="unit_traces")
    plt.show()
    
    
def plot_dist_comparison(features,features_base,labels,idx):
    sns.distplot(features[:,idx],label='Input')
    sns.distplot(features_base[:,idx],label='Baseline')
    plt.legend()
    plt.xlabel(labels[idx])
    plt.show()
    
    
def subplots_4_features(features,features_base,labels,indeces):
    plt.figure()
    plt.subplot(221)
    plot_dist_comparison(features,features_base,labels,indeces[0])
    plt.subplot(222)
    plot_dist_comparison(features,features_base,labels,indeces[1])
    plt.subplot(223)
    plot_dist_comparison(features,features_base,labels,indeces[2])
    plt.subplot(224)
    plot_dist_comparison(features,features_base,labels,indeces[3])
    
    
def plot_phaselocks(phaselocks,sensor_names):
    plt.figure(figsize=(30, 30))
    sns.heatmap(phaselocks,xticklabels=sensor_names,yticklabels=sensor_names)
    plt.plot([0, len(sensor_names)],[len(sensor_names), 0])
    plt.show()


def plot_scalp_grid_bands_mean(values,sensor_names,freq_bands,title='',fontsize=10,figsize=(10,30),scale_per_row=True,vmin=None,vmax=None):
    import braindecode.paper.plot
    values[np.isnan(values)]=0
    values = np.expand_dims(values.T,axis=0)
    braindecode.paper.plot.plot_scalp_grid(values,sensor_names,col_names=freq_bands,fontsize=fontsize,figsize=figsize,scale_per_row=scale_per_row,vmin=vmin,vmax=vmax)
    plt.show()