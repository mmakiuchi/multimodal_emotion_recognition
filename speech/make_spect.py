"""
    Name: make_spect.py
    Description: Generate mel spectrograms and save them as npy files.
"""

import os
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import argparse

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq # point where the gain drops by 3dB

    # butterworth order-th order filter. Returns the filter coefficients
    # b = numerator, a = denominator polynomials of the filter
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def compute_fft(x, config):
    """ Computes the FFT and returns the power spectrum """
    x = np.pad(x, int(config.fft_len//2), mode='reflect')

    noverlap = config.fft_len - config.hop_len
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//config.hop_len, config.fft_len)
    strides = x.strides[:-1]+(config.hop_len*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', config.fft_len, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=config.fft_len).T
    return np.abs(result)

def save_spect(file_list, dir_name, subdir, a, b, mel_basis, min_level, target_dir, config):
    """ Saves a spectrogram as an npy file """

    for file_name in sorted(file_list):
        print("Processing ", file_name)
        # Read audio file
        x, _ = sf.read(os.path.join(dir_name,subdir, file_name))
        
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)

        # Compute spectrogram
        D = compute_fft(y, config).T
        
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)
        
        # save spect as .npy  
        np.save(os.path.join(target_dir, subdir, file_name[:-4]),
                S.astype(np.float32), allow_pickle=False)

def spect_transform(config, target_dir):
    """ Spectrogram transformation for the IEMOCAP dataset """

    # get the transpose of the mel filter bank
    mel_basis = mel(config.freq, config.fft_len, fmin=config.min_f, fmax=config.max_f, n_mels=config.mel_bins).T

    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, config.freq, order=5)

    dir_name, subdirList, _ = next(os.walk(config.speech_dir))
    print('Found directory: %s' % dir_name)

    for subdir in sorted(subdirList):
        print("sub dir: ", subdir)
        if not os.path.exists(os.path.join(target_dir, subdir)):
            os.makedirs(os.path.join(target_dir, subdir))
        new_target_dir = os.path.join(target_dir, subdir)

        session_dir, spks_gender, _ = next(os.walk(os.path.join(dir_name,subdir)))
        
        for gender in spks_gender:
            _ , _, file_list = next(os.walk(os.path.join(session_dir,gender)))

            if not os.path.exists(os.path.join(new_target_dir, gender)):
                os.makedirs(os.path.join(new_target_dir, gender))

            save_spect(file_list=file_list,
                       dir_name=session_dir,
                       subdir=gender,
                       a=a,
                       b=b,
                       mel_basis=mel_basis,
                       min_level=min_level,
                       target_dir=new_target_dir,
                       config=config)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_dir", type=str, help="directory to the speech wave files")
    parser.add_argument("--spmel_dir", type=str, help="directory of the spmel files")
    parser.add_argument("--freq", type=int, default=16000, help="type of encoder architecture")
    parser.add_argument("--fft_len", type=int, default=400, help="size of a fft window")
    parser.add_argument("--hop_len", type=int, default=320, help="hop length used to compute the fft")
    parser.add_argument("--mel_bins", type=int, default=80, help="number of mel frequency bins")
    parser.add_argument("--min_f", type=int, default=90, help="min frequency to compute mel")
    parser.add_argument("--max_f", type=int, default=7600, help="max frequency to compute mel")

    config = parser.parse_args()
    print("Configuration: \n", config)

    # define the target directory (where we will store the spectrograms as npy files)
    target_dir = './' + config.spmel_dir

    spect_transform(config=config,
                    target_dir=target_dir)

if __name__ == '__main__':
    main()