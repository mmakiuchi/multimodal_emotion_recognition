import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from speech.make_spect import butter_highpass, compute_fft

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_spec(file_name):

    config = {}
    config["freq"]=16000
    config["fft_len"]=400
    config["hop_len"]=320
    config["mel_bins"]=80
    config["min_f"]=90
    config["max_f"]=7600

    mel_basis = mel(config["freq"], config["fft_len"], fmin=config["min_f"], fmax=config["max_f"], n_mels=config["mel_bins"]).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, config["freq"], order=5)

    # Read audio file
    x, _ = sf.read(file_name)
    
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)

    config = dotdict(config)

    # Compute spect
    D = compute_fft(y, config).T
    
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)

    return S