import scipy
import scipy.fftpack
import scipy.optimize
import numpy as np
import braindecode.analysis.stats as stats
from scipy.signal import lfilter,firwin,hilbert

def real_frequency(FFT):
	FFT = abs(FFT)
	FFT = 20*scipy.log10(FFT)
	return FFT

def get_frequency(signals, sampling_rate=None, **kwargs):
	FFT = scipy.fft(signals,axis=2)
	FFT = FFT[:,:,:FFT.shape[2]/2]#+np.complex(np.finfo(float).eps,np.finfo(float).eps)

	return FFT


def get_frequency_change(signals, sampling_rate=None, **kwargs):
	if signals.shape[2]%2!=0:
		signals = signals[:,:,:-2]

	signals1 = signals[:,:,:signals.shape[2]/2]
	signals2 = signals[:,:,signals.shape[2]/2:]
	signals2 = signals2[:,:,:signals1.shape[2]]

	FFT1 = real_frequency(get_frequency(signals1,sampling_rate))
	FFT2 = real_frequency(get_frequency(signals2,sampling_rate))
	FFT = FFT2-FFT1

	return FFT


def get_phase_change(signals, sampling_rate=None, **kwargs):
	if signals.shape[2]%2!=0:
		signals = signals[:,:,:-2]

	signals1 = signals[:,:,:signals.shape[2]/2]
	signals2 = signals[:,:,signals.shape[2]/2:]
	signals2 = signals2[:,:,:signals1.shape[2]]
	phase1 = np.angle(get_frequency(signals1,sampling_rate)[:,:,1:])
	phase2 = np.angle(get_frequency(signals2,sampling_rate)[:,:,1:])
	phasec = phase2-phase1

	phasec[phasec<-np.pi] += 2*np.pi
	phasec[phasec>np.pi] -= 2*np.pi

	return phasec


def get_offset(signals, **kwargs):
	means = signals.mean(axis=2)+np.finfo(float).eps
	return means


def get_offset_change(signals, **kwargs):
	if signals.shape[2]%2!=0:
		signals = signals[:,:,:-2]

	signals1 = signals[:,:,:signals.shape[2]/2]
	signals2 = signals[:,:,signals.shape[2]/2:]

	means = get_offset(signals2)-get_offset(signals1)
	return means


def get_bandpower(signals, **kwargs):
	return np.log(signals.var(axis=2))


def phase_locking_value(signals, sampling_rate, filt_order, filt_range):
	filt = firwin(filt_order, filt_range, pass_zero=False, nyq=sampling_rate/2)
	signals = np.angle(hilbert(lfilter(filt, 1.0, signals, axis=2),axis=2))
    
	plv = np.zeros((signals.shape[1],signals.shape[1]))
	for ch2 in range(signals.shape[1]):
		for ch1 in range(ch2,signals.shape[1]):
			data1 = signals[:,ch1,:]
			data2 = signals[:,ch2,:]
			plv[ch1,ch2] = np.mean(np.abs(np.sum(np.exp(1j*(data1-data2)),axis=0))/signals.shape[0],axis=0)
            
	return plv


def get_band_means(FFT_vals,FFT_freqs,bands,circmean=False,median=False):
	"""Calculates mean amplitude of frequency bands

	FFT_vals: FFT NxFrequencies
	FFT_freqs: Frequencies
	bands: Mx2 with start and end frequency

	Returns:
	bands_means: NxM band means
	"""
	if median:
		assert(not circmean)

	bands = np.asarray(bands)
	band_means = np.zeros((len(FFT_vals),len(bands)))
	for i,band in enumerate(bands):
		band_start = np.argmax(FFT_freqs>=band[0])
		band_end = np.argmax(np.logical_and(FFT_freqs>=band[0],FFT_freqs>=band[1]))
		if band_end==len(FFT_freqs)-1 and FFT_freqs[-1]>=band[1]:
			band_end += 1
		if not circmean:
			if not median:
				band_means[:,i] = np.mean(FFT_vals[:,band_start:band_end],axis=1)
			else:
				band_means[:,i] = np.median(FFT_vals[:,band_start:band_end],axis=1)
		else:
			band_means[:,i] = scipy.stats.circmean(FFT_vals[:,band_start:band_end],axis=1)

	return band_means


def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample


def sinusoid(x, a1, a2, a3):
    return a1 * np.sin(a2 * x + a3)


def fit_sinusoid(signal,sampling_rate):
    t = 2*np.pi*np.arange(len(signal))/sampling_rate

    offset = signal.mean()
    signal = signal - offset

    FFT = scipy.fftpack.fft(signal)
    idx = (np.abs(FFT)**2).argmax()
    freqs = scipy.fftpack.fftfreq(len(signal),d=1./sampling_rate)
    frequency = freqs[idx]

    phase = np.angle(scipy.fftpack.fft(signal))[idx]

    amplitude = np.abs(signal).max()

    guess = [amplitude, frequency, phase]
    print guess
    (amplitude, frequency, phase), pcov = scipy.optimize.curve_fit(sinusoid,t,signal,guess)
    return (offset, amplitude, frequency, phase)