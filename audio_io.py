# -*- coding: utf-8 -*-

import numpy as np
import soundfile
import librosa
import io

def wav_read(filename, tgt_fs=16000):
    #y, fs = soundfile.read(filename, dtype='float32')
    y, fs = librosa.load(filename, sr=None, mono=False)
    if tgt_fs is not None:
        if fs != tgt_fs:
            y = librosa.resample(y, orig_sr=fs, target_sr=tgt_fs)
            fs = tgt_fs
    return y, fs

def wav_write(data, fs, filename):
    max_value_int16 = (1 << 15) - 1
    data *= max_value_int16
    soundfile.write(filename, data.astype(np.int16), fs, subtype='PCM_16',
             format='WAV')
