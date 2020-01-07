import sys
import os
import numpy as np

import librosa
import soundfile as sf
import pydub

from scipy.io import wavfile

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from constants import RAW_AUDIO_PATH, PROCESS_AUDIO_PATH, FRAME_SIZE, FRAME_STRIDE, NFFT, NFILT, MEL_HZ_CONST_1, MEL_HZ_CONST_2

class Utility:
    def apply_hamm_window(self, frames, frame_length):
        frames *= np.hamming(frame_length) # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Formula **
        return frames

    def apply_pow_spec(self, fft_res, nfft=NFFT):
        pow_spec = ((1.0 / NFFT) * ((fft_res) ** 2))
        return pow_spec
        
    def apply_fft(self, frames, nfft=NFFT):
        fft_res = np.absolute(np.fft.rfft(frames, NFFT))
        return fft_res

    def apply_tri_filterbank(self, frames, sr=SAMPLE_RATE, nfilt=NFILT, nfft=NFFT):
        low_freq_mel = 0
        high_freq_mel = (MEL_HZ_CONST_1 * np.log10(1 + (sr/2) / MEL_HZ_CONST_2)) # m = 2595 * log10(1 + f / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (MEL_HZ_CONST_2 * (10**(mel_points / MEL_HZ_CONST_1) - 1))  # f = 700(10^( m / 2595) - 1)
        b = np.floor((nfft + 1) * hz_points / sr)
        fb = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
        for i in range(1, nfilt + 1):
            f_m_btm = int(b[i - 1])
            f_m_mid = int(b[i])
            f_m_top = int(bin[i + 1])
            for j in range(f_m_btm, f_m_mid):
                fb[i - 1, j] = (j - b[i - 1]) / (b[i] - b[i - 1])
            for j in range(f_m_mid, f_m_top):
                fb[i - 1, j] = (b[i + 1] - j) / (b[i + 1] - b[i])
        fb = np.dot(frames, fb.T)
        fb = np.where(fb == 0, np.finfo(float).eps, fb)
        fb = 20 * np.log10(fb)
        return fb

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

    def wav_to_mel_log(self, wav):
        frames = get_frames(wav)
        fft_signal = apply_fft(frames)
        pow_signal = apply_pow_spec(fft_signal)
        mel_spec = apply_tri_filterbank(pow_signal)
        # OPTIONAL : sinusoidal liftering for de-emphasizing 
        # (nframes, ncoeff) = mel_spec.shape
        # n = numpy.arange(ncoeff)
        # lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
        # mel_spec *= lift  

    def mp3_to_wav(self, filename):
        data = pydub.AudioSegment.from_mp3(RAW_AUDIO_PATH + filename + '.mp3')
        data.export(filename + '.wav', format="wav")
        wav = wavfile.read(PROCESS_AUDIO_PATH + filename + '.wav')[1]
        return wav
    
