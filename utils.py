import pyworld
import numpy as np
import librosa

def load_wav(path, sr):
    wav, sr = librosa.load(path, sr)
    wav = librosa.effects.trim(wav, top_db=20, frame_length=1024, hop_length=256)[0]
    return wav

def get_logf0(wav, fs, frame_period=0.005):
    if isinstance(wav[0], np.float32):
        wav = wav.astype(np.double)

    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period)
    nonzero_indecies = f0 > 0
    logf0 = np.zeros_like(f0)
    logf0[nonzero_indecies] = np.log(f0[nonzero_indecies])
    return logf0

def speaker_norm(f0):
    nonzero = f0 > 0
    f0[nonzero] = (f0[nonzero] - f0[nonzero].mean()) / f0[nonzero].std() / 4
    return f0