import sys
import os
import numpy as np
import subprocess

import librosa
import librosa.display
import librosa.core
import soundfile as sf
import pydub

from scipy.io import wavfile
import scipy
from scipy import signal, fftpack
import numpy.fft as fft

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from constants import *

def mp3_to_wav(filename):
    data = pydub.AudioSegment.from_mp3(RAW_AUDIO_PATH + filename + '.mp3')
    data.export(PROCESS_AUDIO_PATH + filename + '.wav', format='wav')
    wav = wavfile.read(PROCESS_AUDIO_PATH + filename + '.wav')[1]
    return wav

def get_mel_filterbank():
    input_bins = (FRAME_LENGTH // 2) + 1
    fb = np.zeros((input_bins, NUM_BANDS))

    min_mel = MEL_HZ_CONST_1 * np.log1p(MEL_MIN_FREQ / MEL_HZ_CONST_2)
    max_mel = MEL_HZ_CONST_1 * np.log1p(MEL_MAX_FREQ / MEL_HZ_CONST_2)
    spacing = (max_mel - min_mel) / (NUM_BANDS + 1)
    peaks_mel = min_mel + np.arange(NUM_BANDS + 2) * spacing
    peaks_hz = MEL_HZ_CONST_2 * (np.exp(peaks_mel / MEL_HZ_CONST_1) - 1)
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    for b, filt in enumerate(fb.T):
        left_hz, top_hz, right_hz = peaks_hz[b:b+3] 
        left_bin, top_bin, right_bin = peaks_bin[b:b+3]
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) / (top_bin - left_bin))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) / (right_bin - top_bin))
        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()
    return fb
    
def spectrogram(samples, batch=50):
    if len(samples) < FRAME_LENGTH:
        return np.empty((0, FRAME_LENGTH // 2 + 1), dtype=samples.dtype)
    win = np.hanning(FRAME_LENGTH).astype(samples.dtype)
    num_frames = max(0, (len(samples) - FRAME_LENGTH) // HOPSIZE + 1)
    batch = min(batch, num_frames)
    if batch <= 1 or not samples.flags.c_contiguous:
        rfft = rfft_builder(samples[:FRAME_LENGTH], n=FRAME_LENGTH)
        spect = np.vstack(np.abs(rfft(samples[pos:pos + FRAME_LENGTH] * win))
                        for pos in range(0, len(samples) - FRAME_LENGTH + 1, int(HOPSIZE)))
    else:
        rfft = rfft_builder(np.empty((batch, FRAME_LENGTH), samples.dtype), n=FRAME_LENGTH, threads=1)
        frames = np.lib.stride_tricks.as_strided(
                        samples, shape=(num_frames, FRAME_LENGTH),
                        strides=(samples.strides[0] * HOPSIZE, samples.strides[0]))
        spect = [np.abs(rfft(frames[pos:pos + batch] * win))
                        for pos in range(0, num_frames - batch + 1, batch)]
        if num_frames % batch:
            spect.append(spectrogram(samples[(num_frames // batch * batch) * HOPSIZE:], batch=1))
        spect = np.vstack(spect)
    return spect
    
def get_samples_ffmpeg(filename, cmd='ffmpeg'):
    call = [cmd, "-v", "quiet", "-i", PROCESS_AUDIO_PATH + filename + '.wav', "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"]
    samples = subprocess.check_output(call)
    return np.frombuffer(samples, dtype=np.float32)

def extract_spect(filename):
    wav = mp3_to_wav(filename)
    try:
        samples = get_samples_ffmpeg(filename)
    except Exception:
        samples = get_samples_ffmpeg(filename, cmd='avconv')
    return spectrogram(samples)

def apply_filterbank(batches, filterbank):
    for spects, labels in batches:
        # transform all excerpts in a single dot product
        yield (np.dot(spects.reshape(-1, spects.shape[-1]), filterbank).reshape(
                (spects.shape[0], spects.shape[1], -1)), labels)

def apply_logarithm(batches, clip=1e-7):
    for spects, labels in batches:
        yield np.log(np.maximum(spects, clip)), labels

def rfft_builder(samples, *args, **kwargs):
    if samples.dtype == np.float32:
        return lambda *a, **kw: np.fft.rfft(*a, **kw).astype(np.complex64)
    else:
        return np.fft.rfft

def preemphasis(seq, coeff):
    return scipy.append(seq[0], seq[1:] - coeff * seq[:-1])

def test():
    spec = extract_spect('0uFnnurkT6KwgzfQ6KAvSi')
    plt.plot(spec)
    plt.show()
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # plt.subplot(4, 2, 1)
    # librosa.display.specshow(D, y_axis='linear')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Linear-frequency power spectrogram')

test()


