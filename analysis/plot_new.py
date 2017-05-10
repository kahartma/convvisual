import numpy as np
from matplotlib import pyplot as plt
import convvisual.analysis.utils as utils
import seaborn as sns
import os

def plot_max_filters(RF_Result):
    print 'Max filters: ',RF_Result.max_filters
    sns.heatmap(np.reshape(RF_Result.filters_means,(-1,1)).T,
                linewidths=.5)
    plt.xlabel('Filter #')
    plt.ylabel('Activation Diff')
    plt.title('Mean activation for feature maps over all inputs')
    plt.show()
    
    
def print_features(path,score,p,labels,indeces):
    f=open(os.path.join(path,'KS scores'),'w')
    for idx in indeces:
        print >> f, 'Score %f  p %f  : %s'%(score[idx],p[idx],labels[idx])
    f.close()
        
        
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
    m = np.median(X_RF_cropped[:,channel],axis=0)
    s25 = np.percentile(X_RF_cropped[:,channel],25,axis=0)#utils.percentile_deviation(X_RF_cropped[:,channel],axis=0)
    s75 = np.percentile(X_RF_cropped[:,channel],75,axis=0)
    plt.fill_between(np.arange(X_RF_cropped.shape[2]),s25,s75,color='r',zorder=100,alpha=0.3) 
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
    
    
def plot_dist_comparison(features,features_base,labels,idx,title=''):
    sns.distplot(features[:,idx],label='Input')
    sns.distplot(features_base[:,idx],label='Baseline')
    plt.title(title)
    plt.legend()
    plt.xlabel(labels[idx])
    
    
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


def KS_bar_plot(data,feature_names,KS_D_critical,fig_h):
    rows = feature_names
    columns = ['Filter %d' % x for x in range(data.shape[1])]

    plt.figure(figsize=(data.shape[1]/2,fig_h))

    colors = plt.cm.jet(np.linspace(0, 0.9, data.shape[0]))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x) for x in data[row]])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()
    rows = rows[::-1]

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel('KS Score')
    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('KS Score in Filter by Feature Type (Bonferroni corrected p<0.05, KS>%03f)'%KS_D_critical)


def grid_plot(data,sensor_names,frequencies,fig_h,xticks,yticks,labels,title,cmap):
    plt.figure(figsize=(fig_h,fig_h))
    plt.imshow(data,interpolation='nearest',aspect='equal',origin='lower',cmap=plt.get_cmap(cmap))
    plt.grid(False)
    plt.colorbar()
    plt.xticks(xticks[0],xticks[1],fontsize=7,rotation='vertical')
    plt.yticks(yticks[0],yticks[1])
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    plt.title(title)