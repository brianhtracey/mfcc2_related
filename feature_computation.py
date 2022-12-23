
def compute_4kratio(fvec, Pxx, fmax = 12000):
    # get linear ratio of energy below 4 kHz to energy above

    idx_4k = (np.abs(fvec - 4000 )).argmin()  # 4k point
    
    idx_max = (np.abs(fvec - fmax )).argmin()
    
    low_result = np.nansum(Pxx[0:idx_4k])  
    hi_result = np.nansum(Pxx[idx_4k:idx_max])
    
    res = low_result /  hi_result

    return(res)

#####

def compute_features(y, sr):
    
    winlen_sec = 0.02
    nsamp = 2 ** math.ceil(math.log2(sr * winlen_sec))
    from scipy.signal import welch
    fvec, Pxx = welch(y, fs=sr, window='hann', nperseg=nsamp, noverlap=nsamp / 2)
    
    # get high-low energy ratio
    ratio4k = compute_4kratio(fvec,Pxx)

    # get mfcc2 for comparison
    import librosa
    nsamp = int(sr * 0.02)  # 20 msec frame length
    hop_samp = int(nsamp / 2)  # set pitch frames and spectrogram hoplength to be 2x as fast as spectrogram frames
    # compute spectrogram
    SGram = librosa.stft(y=y, n_fft=nsamp, hop_length=hop_samp)

    # fmax = 8000 Hz
    Sm = librosa.feature.melspectrogram(S=np.abs(SGram) ** 2, sr=sr, n_mels=40, fmax = 8000.0)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(Sm), n_mfcc=13)
    mfcc2_8k = np.mean(mfcc[1,:]) 
    
    # fmax = 12000 Hz
    Sm = librosa.feature.melspectrogram(S=np.abs(SGram) ** 2,
                                        sr=sr, n_mels=60, fmax = 12000.0)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(Sm), n_mfcc=13)
    mfcc2_12k = np.mean(mfcc[1,:])
    
    # fmax = 15000 Hz
    Sm = librosa.feature.melspectrogram(S=np.abs(SGram) ** 2,
                                        sr=sr, n_mels=60, fmax = 15000.0)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(Sm), n_mfcc=13)
    mfcc2_15k = np.mean(mfcc[1,:])
                

    return(fvec, Pxx, ratio4k, mfcc2_8k, mfcc2_12k, mfcc2_15k)

