import pyworld
import numpy as np
import librosa

def load_wav(path, sr):
    wav, sr = librosa.load(path, sr)
    return wav

def get_logf0(wav, fs, frame_period=0.005):
    if isinstance(wav[0], np.float32):
        wav = wav.astype(np.double)

    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period)
    nonzero_indecies = f0 > 0
    logf0 = np.zeros_like(f0)
    logf0[nonzero_indecies] = np.log(f0[nonzero_indecies])
    return logf0