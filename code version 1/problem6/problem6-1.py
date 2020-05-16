from scipy.io import loadmat
from scipy.io import wavfile
from scipy.signal import fftconvolve, butter, convolve
from scipy import signal
#from IPython.display import Audio
import sounddevice as sd
import numpy as np 
from numpy import fft
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os
from time import *
import shutil
import re

def _u(t):
    if (t > 0):
        return 1
    else:
        return 0

def u(t):
    ufunc = np.zeros_like(t)
    for i,time in enumerate(t):
        ufunc[i] = _u(time)
    return ufunc

def delta(t):
    ufunc = np.zeros_like(t)
    for i,time in enumerate(t):
        ufunc[i] = 1 if (time == 0) else 0
    return ufunc

def readmat(filename, pt = True):
    x = loadmat(filename)
    data = x['H'][0]
    if pt:
        figure = plt.figure()
        plt.plot(data)
        print(data.shape)
        plt.show()
    return data

def fft_plot(signal, pt = True):
    sigfft = fft.fft(signal)
    if pt:
        plt.plot(np.abs(sigfft))
        plt.show()
    return sigfft

def fast_convolve(sig1, sig2):
    sigf1 = fft.fft(sig1)
    sigf2 = fft.fft(sig2)
    convftsign = sigf1 * sigf2
    return np.real(fft.ifft(convftsign))

#high freq pass 
def butter_hpf(t, freq = None, duration = 1):
    def highpass(data):
        base = 2 * freq * np.sinc(2 * freq * t)
        signal = delta(t) - base
        out = convolve(signal, data)/44100#np.sum(base)
        return out[data.shape[0]//2 : 3 * data.shape[0]//2]
    return highpass

# low freq pass 
def butter_lpf(t, freq = None, duration = 1):
    def lowpass(data):
        signal = 2 * freq * np.sinc(2 * freq * t)
        out = convolve(signal, data)/44100#(np.sum(signal))
        return out[data.shape[0]//2 : 3 * data.shape[0]//2]
    return lowpass

def butter_bpf(t, freq1 = None, freq2 = None, duration = 1):
    def bandpass(data):
        # signal = (wcl/np.pi) * np.sinc(wcl * t) - (wch/np.pi) * np.sinc(wch * t)
        signall = (2*freq1) * np.sinc((2*freq1) * t)
        signalh = (2*freq2) * np.sinc((2*freq2) * t) 
        out = convolve((signalh - signall), data)/44100#(np.sum(signall))
        return out[data.shape[0]//2 : 3 * data.shape[0]//2]
    return bandpass



def player_init(fs):
    def playsound(data):
        sd.play(data, fs)
        sleep(data.shape[0]/fs)
        sd.stop()
        pass
    return playsound


if __name__ == "__main__":
    fs = 44100
    spls = fs * 1
    player = player_init(fs)
    duration = 1

    t = np.linspace((0 - duration/2)*fs + 0.5, (duration - duration/2)*fs - 0.5, spls)/fs
    data = np.sin(2 * np.pi * 3 * t) + np.cos(2 * np.pi * 19 * t) 
    data2 = np.sin(2 * np.pi * 3 * t)
    data3 = np.cos(2 * np.pi * 19 * t) 
    print(t)

    # lpf = butter_lpf(t, freq = 3000)
    # bpf1 = butter_bpf(t, freq1 = 5000 - rounding/2, freq2 = 5000+rounding/2)
    # bpf2 = butter_bpf(t, freq1 = 12000 - rounding/2, freq2 = 12000+rounding/2)
    # bpf3 = butter_bpf(t, freq1 = 19000 - rounding/2, freq2 = 19000+rounding/2)
    ammod = np.cos(2 * np.pi * 19 * t)/np.sin(2 * np.pi * 19 * t)
    lpf = butter_lpf(t, freq = 5)
    bpf = butter_bpf(t, freq1 = 1, freq2= 5)

    flsig1 = lpf(data*ammod)
    # fft_plot(data2)
    # fft_plot(data3)
    # fft_plot(data)
    # fft_plot(flsig1)


    

    plt.plot(data, color = 'black')
    plt.plot(data2, color = 'blue')
    plt.plot(data3, color = 'green')
    plt.plot(flsig1, color = 'red')
    plt.show()