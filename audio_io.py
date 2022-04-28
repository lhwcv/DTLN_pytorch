# -*- coding: utf-8 -*-

import numpy as np
import soundfile
import librosa
import io

def wav_read(filename, tgt_fs=None):
    y, fs = soundfile.read(filename, dtype='float32')
    if tgt_fs is not None:
        if fs != tgt_fs:
            if fs != 16000:
                y = librosa.resample(y, tgt_fs, 16000)
                fs = tgt_fs
    return y, fs

def wav_write(data, fs, filename):
    max_value_int16 = (1 << 15) - 1
    data *= max_value_int16
    soundfile.write(filename, data.astype(np.int16), fs, subtype='PCM_16',
             format='WAV')
