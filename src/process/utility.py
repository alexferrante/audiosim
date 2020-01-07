import sys
import os
import numpy as np

import librosa
import soundfile as sf
import pydub

from scipy.io import wavfile, lfilter, hamming
import matplotlib.pyplot as plt
import librosa.display

sys.path.append(os.getcwd())
from constants import RAW_AUDIO_PATH, PROCESS_AUDIO_PATH, WINLEN, WINSHIFT

class Utility:
    def mp3_to_wav(self, filename):
        data = pydub.AudioSegment.from_mp3(RAW_AUDIO_PATH + filename + '.mp3')
        data.export(filename + '.wav', format="wav")
        wav = wavfile.read(PROCESS_AUDIO_PATH + filename + '.wav')[1]
        samples = np.sum(wav, axis=-1)
        return samples
    
    def slice_samples(self, samples, w_len, w_shift):
        ls = []
        for i in range(0, len(samples)//w_len*w_len, w_shift):
            if i+w_len > len(samples):
                break
            ls.append(samples[i:i+w_len])
        return np.array(ls)
    
    def pre_window_filter(self, input, p=0.97):
        return lfilter([1, -p], [1], input)

    def apply_hamm_window(self, input):
        N, M = input.shape
        win = hamming(M, sym=False)
        return (input * win)

    def mfcc(self, samples, w_len=WINLEN, w_shift=WINSHIFT, p_emp_coeff=pre_data_constants.PREEMPCOEFF, nfft=NFFT, n_ceps=NCEPS, sample_rate=FS, lift_coef=22, w_lifter=False):
        frames = slice_samples(samples, w_len, w_shift)
        p_emp = pre_window_filter(frames, p_emp_coeff)
        h_win = apply_hamm_window(p_emp)
        pow_spec = power_spec(h_win, nfft)
        m_spec, m_weights = log_mel_spec(pow_spec, sample_rate)
        mel_cep = get_cep(m_spec, n_ceps)
        if w_lifter:
            return lift(mel_cep, lift_coef)
        else:
            return mel_cep, m_spec, m_weights
        

    
Utility.process_mp3('2AU96PBR4PzUJPVGIQTalF')
    
