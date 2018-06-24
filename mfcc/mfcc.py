#coding:utf-8
'''
@author : chencheng
@company: topzero.cn
@date   : 20180617
@brief  :
    mfcc -> mel-frequency cepstral coefficients，梅尔频率倒谱系数
    filter banks 滤波bank
'''

import numpy as np
import scipy.io.wavfile
import scipy.fftpack

import matplotlib.pyplot as plt


def PlotWav(wav):
    plt.plot(range(len(wav)), wav)
    plt.show()


def mfcc(signal, sample_rate):
    ''' Pre-Emphasis, not needed in modern speech processing system '''
    emphasized_signal = signal    

    ''' Framing '''
    frame_size   = 0.025
    frame_stride = 0.01
    frame_length = frame_size * sample_rate
    frame_step   = frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples, 
    # without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z) 

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step),
                    (frame_length, 1) ).T
    # print(indices)
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    ''' window function '''
    print(('before windowing', frames.shape, np.hamming(frame_length).shape))
    frames *= np.hamming(frame_length)
    print(('after windowing', frames.shape))

    ''' Fourier-Transform and Power Spectrum '''
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    print(('power_frames', pow_frames.shape))

    ''' Filter Banks '''
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz   
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    print(('fbank', fbank.shape))

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB


    ''' MFCC '''
    num_ceps = 12
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    #(nframes, ncoeff) = mfcc.shape
    #n = np.arange(ncoeff)
    #lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    #mfcc *= lift  #*


    ''' mean normalization '''
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    print(('fiterbanks', filter_banks.shape))
    print(('mfcc', mfcc.shape))

    print('# mfcc vs filter-bank!')
    return filter_banks, mfcc    


def main():
    ''' setup '''
    wavpath = '/home/chen/myworkspace/gits/Matlab-toolbox-for-DNN-based-speech-separation/premix_data/clean_speech/S_01_01.wav'
    # wavpath = '/home/chen/dataset/speech_commands_v0.01/bed/0a7c2a8d_nohash_0.wav'

    sample_rate, signal = scipy.io.wavfile.read(wavpath)
    print(('rate, len, time', sample_rate, len(signal), len(signal)/float(sample_rate)))
    # PlotWav(signal)

    mfcc(signal, sample_rate)



if __name__ == "__main__":
    main()
