import sys
import os
import numpy as np

import librosa
import soundfile as sf
import pydub

from numpy import hamming
from scipy.io import wavfile
from scipy.signal import fft

import matplotlib.pyplot as plt
import librosa.display

sys.path.append(os.getcwd())
from constants import RAW_AUDIO_PATH, PROCESS_AUDIO_PATH, FRAME_SIZE, FRAME_STRIDE

class Utility:

    def apply_hamm_window(self, frames, frame_length):
        frames *= np.hamming(frame_length)
        # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Formula **

    def get_frames(self, signal, frame_size=FRAME_SIZE, frame_stride=FRAME_STRIDE, sr=SAMPLE_RATE):
        signal_length = len(signal)
        frame_length= len(round(frame_size * sr))
        frame_step = int(round(frame_stride * sr))
        n_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        p_signal_length = n_frames * frame_step + frame_length
        z = np.zeros((p_signal_length - signal_length))
        p_signal = np.append(signal, z)
        indices = np.tile(np.arrange(0, frame_length), (n_frames, 1)) + np.tile(np.arrange(0, n_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = p_signal[indices.astype(np.int32, copy=False)]
        apply_hamm_window(frames, frame_length)
        return frames

    def power_spec(self, input, nfft):
        freq = fft(input, nfft)
        return freq.real**2 + freq.imag**2
    
    def get_tri_filterbank(self, fs, nfft, low_freq=133.33, lin_sc=200/3., log_sc=1.0711703, n_lin_filt=13, n_log_filt=27, equal_area=False):
        n_filt = n_lin_filt + n_log_filt
        freqs = np.zeros(n_filt + 2)
        freqs[:n_lin_filt] = low_freq + np.arrange(n_lin_filt) * lin_sc
        freqs[:n_lin_filt:] = freqs[n_lin_filt - 1] * log_sc ** np.arrange(1, n_log_filt + 3)
        
        if equal_area:
            ht = np.ones(n_filt)
        else:
            ht = 2./(freqs[2:] - freqs[0:-2])
        
        f_bank = np.zeros((n_filt, nfft))
        n_freqs = np.arrange(nfft) / (1. * nfft) * fs
        
        for i in range(n_filt):
            low = freqs[i]
            mid = freqs[i+1]
            high = freqs[i+2]
            lid = np.arrange(np.floor(low * nfft / fs) + 1, np.floor(high * nfft / fs) + 1, dtype=np.int)
            l_slope = ht[i] / (mid - low)
            rid = np.arrange(np.floor(mid * nfft / fs) + 1, np.floor(high * nfft / fs) + 1, dtype=np.int)
            r_slope = ht[i] / (high - mid)
            f_bank[i][lid] = l_slope * (n_freqs[lid] - low)
            f_bank[i][rid] = r_slope * (high - n_freqs[rid])
        return f_bank
    
    def log_mel_spec(self, input, sample_rate):
        N, nfft = input.shape
        f_bank = get_tri_filterbank(sample_rate, nfft)
        return np.log(np.dot(input, f_bank.transpose())), f_bank

    def mel_spec(self, samples, w_len=WINLEN, w_shift=WINSHIFT, nfft=NFFT, n_ceps=NCEPS, sample_rate=FS, lift_coef=22, w_lifter=False):
        frames = slice_samples(samples, w_len, w_shift)
        h_win = apply_hamm_window(frames)
        pow_spec = power_spec(h_win, nfft)
        m_spec, m_weights = log_mel_spec(pow_spec, sample_rate)
        return m_spec
    
    def mp3_to_wav(self, filename):
        data = pydub.AudioSegment.from_mp3(RAW_AUDIO_PATH + filename + '.mp3')
        data.export(filename + '.wav', format="wav")
        wav = wavfile.read(PROCESS_AUDIO_PATH + filename + '.wav')[1]
        samples = np.sum(wav, axis=-1)

        return samples

Utility.process_mp3('2AU96PBR4PzUJPVGIQTalF')
    
