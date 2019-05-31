# Third Party
import librosa
import numpy as np

# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train', sequence_number = 8):
    try:
        data = np.load(path+'_len:{}.npy'.format(spec_len))
        return data
    except Exception as ex:
        pass
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    randtime = 0 #np.random.randint(0, time-spec_len)
    # preprocessing, subtract mean, divided by time-wise var
    mu =  np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)
    spec_mag = (mag_T - mu) / (std + 1e-5)
    spec_mag = np.reshape(spec_mag[:,0:-1],(-1, 257, spec_len))#[:, randtime:randtime+spec_len]
    np.save(path+'_len:{}.npy'.format(spec_len),spec_mag)
    return spec_mag
