#coding:utf-8

import matplotlib.pyplot as plt

import scipy.io.wavfile
import scipy.signal

import numpy as np



def PlotWav(wav):
    plt.plot(range(len(wav)), wav)
    plt.show()


def SaveWav(filename, rate, data):
    wavData = np.int16(data)
    scipy.io.wavfile.write(filename, rate, wavData)
    print('# %s file saved!' %(filename))


def generate_mixture( wavpath, noisepath,
    first_time_flag = False, mix_db = 0.0, 
    noise_cut=0.5, repeat_times=1):

    n_fs, n_tmp = scipy.io.wavfile.read(noisepath)
    s_fs, s     = scipy.io.wavfile.read(wavpath)
    
    if(first_time_flag):
        n_tmp = scipy.signal.resample(n_tmp, len(n_tmp)*16000/n_fs)
        s     = scipy.signal.resample(s,     len(s)    *16000/s_fs)
        n_fs  = 16000
        s_fs  = 16000
    
    ''' for computaion conviniecne, convert to float32 '''
    n_tmp = np.float32(n_tmp)
    s     = np.float32(s)

    if(first_time_flag):
        SaveWav('noise.wav', n_fs, n_tmp)
        SaveWav('s.wav', s_fs, s)

    ''' 
    noise is cut to 2 part, 
    first part for training, second part for test 
    noise_cut = 0.5
    '''
    n_tmp        = n_tmp[:int(len(n_tmp)*noise_cut)]
    double_n_tmp = np.tile(n_tmp, 2)

    for var_ind in range(repeat_times):
        start_cut_point = np.random.randint(len(n_tmp))
        
        ''' noise '''
        n = double_n_tmp[start_cut_point:start_cut_point + len(s)]
        
        ''' snr '''
        snr = 10 * np.log10(np.sum(s**2) / np.sum(n**2))

        db = mix_db
        alpha = np.sqrt(np.sum(s**2) / np.sum(n**2) / (10**(db/10)))
        snr1 = 10*np.log10(np.sum(s**2) / np.sum((alpha*n)**2))

        ''' mix '''   
        mix = s + alpha*n
        SaveWav('mix.wav', s_fs, mix)
        
        ''' energy normalization '''
        '''
        constant = 5000000
        c   = np.sqrt(constant * len(mix)/sum(mix**2))
        mix = mix * c
        print(np.min(mix), np.max(mix))
        '''
    return mix, s, alpha*n


''' ERB(f) 
    erb=21.4*log10(4.37e-3*hz+1)
'''
def Hz_to_ERB(f):
    erb = 21.4 * np.log10(0.00437*f + 1)
    return erb


def gammatone(in_data, numChannel, fRange, fs): 
    in_data = np.float32(in_data)
    
    filterOrder = 4
    gL = 2048
    signalLength = len(in_data)
    
    phase = zeros(numChannel)
    erb_b = hz2erb(fRange)
    erb   = 


def ideal(mat_s, mat_n, mix_db):
    (line, column) = mat_s.shape
    mask = np.zeros((line, column))
    for i in range(line):
        for j in range(column):
            cur_snr = 10 * np.log10(mat_s[i,j] / mat_n[i,j])
            if (cur_snr > mix_db - 5):
                mask[i,j] = 1
            else:
                mask[i,j] = 0
    return mask


def ideal_binary_mask(mix, s, n, NUMBER_CHANNEL, id_wiener_mask, db):
    SAMPLING_FREQUENCY = 16000;
    window_time = 0.02 # use a window of 20 ms */
    stride_time = 0.01 # compute acf every 10 ms */
    WINDOW     = np.int32(SAMPLING_FREQUENCY * window_time);  
    OFFSET     = np.int32(SAMPLING_FREQUENCY * stride_time);               
    
    n_frame = floor( len(mix) / OFFSET ) 
    
    ''' ideal mask '''
    g_s    = gammatone(s, NUMBER_CHANNEL, [50 80000], SAMPLING_FREQUENCY)
    coch_s = cochleagram(g_s, 320)
    g_n    = gammatone(n, NUMBER_CHANNEL, [50 80000], SAMPLING_FREQUENCY)
    coch_n = cochleagram(g_n, 320)

    if (is_wiener_mask == False):
        ideal_mask = ideal(coch_s, coch_n, db)        
    
    ''' features '''
    ''' 
    very complicated, a concatenation of  
    AMS (Amplitude modulation spectrogram)
    RASTA-PLP (relative spectral transfromed perceptual linear prediction coneffiecnets)
    MFCC (mel-frequency cepstral coefficients)
    GF   (Gammaton filterbank power spectra)
    '''
    features = 0

    return features, ideal_mask


def main():
    ''' global define '''
    first_time_flag = False
    mix_db          = 0.0
    noise_cut       = 0.5
    repeat_times    = 1
    NUMBER_CHANNEL  = 64
    


    if(first_time_flag):
        wavpath   = '/home/chen/myworkspace/gits/Matlab-toolbox-for-DNN-based-speech-separation/premix_data/clean_speech/S_01_01.wav'
        # wavpath = '/home/chen/dataset/speech_commands_v0.01
        noisepath = '/home/chen/myworkspace/gits/Matlab-toolbox-for-DNN-based-speech-separation/premix_data/noise/factory.wav'
    else:
        wavpath = 's.wav'
        noisepath = 'noise.wav'

    ''' generate mixture signal '''
    mix, signal, noise = generate_mixture( wavpath, noisepath,
        first_time_flag, mix_db, 
        noise_cut, repeat_times)

    ''' generate features and IBM '''





    print('# generate training pair!')


if __name__ == "__main__":
    main()
