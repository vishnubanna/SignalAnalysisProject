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
def butter_hpf(t, R = None, C = None, m = None, freq = None, duration = 1):
    if m == None:
        if R == None or C == None: 
            rc = 1/(2 * np.pi * freq)
        else:
            rc = R * C
    def highpass(data):
        wc = 1/rc
        base = 2 * freq * np.sinc(2 * freq * t)
        signal = delta(t) - base
        out = convolve(signal, data)/np.sum(base)
        return out[data.shape[0]//2 : 3 * data.shape[0]//2]
    return highpass

# low freq pass 
def butter_lpf(t, R = None, C = None, m = None, freq = None, duration = 1):
    if m == None:
        if R == None or C == None: 
            rc = 1/(2 * np.pi * freq)
        else:
            rc = R * C
    def lowpass(data):
        wc = 1/rc
        signal = 2 * freq * np.sinc(2 * freq * t)
        out = convolve(signal, data)/(np.sum(signal))
        return out[data.shape[0]//2 : 3 * data.shape[0]//2]
    return lowpass

def butter_bpf(t, R1 = None, C1 = None, m1 = None, freq1 = None, R2 = None, C2 = None, m2 = None, freq2 = None, duration = 1):
    if m1 == None:
        if R1 == None or C1 == None: 
            rc1 = 1/(2 * np.pi * freq1)
        else:
            rc1 = R1 * C1
    if m2 == None:
        if R2 == None or C2 == None: 
            rc2 = 1/(2 * np.pi * freq2)
        else:
            rc2 = R2 * C2
    def bandpass(data):
        wch = 1/rc2
        wcl = 1/rc1
        # signal = (wcl/np.pi) * np.sinc(wcl * t) - (wch/np.pi) * np.sinc(wch * t)
        signall = (2*freq1) * np.sinc((2*freq1) * t)
        signalh = (2*freq2) * np.sinc((2*freq2) * t) 
        out = convolve((signalh - signall), data)/(np.sum(signall))
        return out[data.shape[0]//2 : 3 * data.shape[0]//2]
    return bandpass

def shift(signal, freq, time):
    sigfft = fft.fft(signal * np.exp(np.complex(0, 2 *np.pi *freq) * t))
    #sigfft = fft.fft(signal * np.cos(2 * np.pi *freq * t)/np.sin(2 * np.pi *freq * t))
    ret = fft.ifft(sigfft)
    return ret

def player_init(fs):
    def playsound(data):
        sd.play(data, fs)
        sleep(data.shape[0]/fs)
        sd.stop()
        pass
    return playsound


if __name__ == "__main__":
    fs = 44100
    player = player_init(fs)
    data = readmat("RFspectrum.mat")
    duration = data.shape[0]/fs
    datafft = fft_plot(data)
    t = np.linspace((0 - duration/2)*fs + 0.5, (duration - duration/2)*fs - 0.5, data.shape[0])/fs
    print(t)


    rounding = 6000
    rounding2 = 0
    lpf = butter_lpf(t, freq = 3000, duration=duration)
    bpf1 = butter_bpf(t, freq1 = 5000 - rounding/2, freq2 = 5000+rounding/2, duration=duration)
    bpf2 = butter_bpf(t, freq1 = 12000 - rounding/2, freq2 = 12000+rounding/2, duration=duration)
    bpf3 = butter_bpf(t, freq1 = 19000 - rounding/2, freq2 = 19000+rounding/2, duration=duration)
    
    bpf = butter_bpf(t, freq1 = 1, freq2 = 6000, duration=duration)
    lpf1 = butter_lpf(t, freq = 5000, duration=duration)
    lpf2 = butter_lpf(t, freq = 12000, duration=duration)
    lpf3 = butter_lpf(t, freq = 19000, duration=duration)

    w1 = np.cos(2 * np.pi * 5e3 * t)
    w2 = np.cos(2 * np.pi * 12e3 * t)
    w3 = np.cos(2 * np.pi * 19e3 * t)

    # flsig4 = bpf1(data)#/w1
    # flsig5 = bpf2(data)#/w2
    # flsig6 = bpf3(data)#/w3

    flsig4 = bpf1(data)#/w1
    flsig5 = bpf2(data)#/w2
    flsig6 = bpf3(data)#/w3

    # flsig4[flsig4 < 0] = 0
    # flsig5[flsig5 < 0] = 0
    # flsig6[flsig6 < 0] = 0

    flsig4_l = lpf(shift(flsig4, -2e3, t))
    flsig5_l = lpf(shift(flsig5, -9e3, t))
    flsig6_l = lpf(shift(flsig6, -16e3, t))

    # flsig4_l = shift(flsig4_l, 1e3, t)
    # flsig5_l = shift(flsig5_l, 1e3, t)
    # flsig6_l = shift(flsig6_l, 1e3, t)

    flsig4_h = lpf(np.real(shift(flsig4, 2e3, t)))
    flsig5_h = lpf(np.real(shift(flsig5, 9e3, t)))
    flsig6_h = lpf(np.real(shift(flsig6, 16e3, t)))

    # flsig4_h = shift(flsig4_h, -1e3, t)
    # flsig5_h = shift(flsig5_h, -1e3, t)
    # flsig6_h = shift(flsig6_h, -1e3, t)

    flsig4 = lpf(np.real(flsig4_l + flsig4_h))
    flsig5 = lpf(np.real(flsig5_l + flsig5_h))
    flsig6 = lpf(np.real(flsig6_l + flsig6_h))
    # flsig4 = lpf1(flsig4)
    # flsig5 = lpf2(flsig5)
    # flsig6 = lpf3(flsig6)

    # flsig4 = lpf(bpf1(data))
    # flsig5 = lpf(bpf2(data))
    # flsig6 = lpf(bpf3(data))

    plt.plot(flsig4, color = 'green')
    plt.plot(flsig5, color = 'blue')
    plt.plot(flsig6, color = 'red')
    plt.show()

    plt.plot(np.imag(flsig4), color = 'green')
    plt.plot(np.imag(flsig5), color = 'blue')
    plt.plot(np.imag(flsig6), color = 'red')
    plt.show()

    # datafft1 = fft_plot(flsig1)
    # datafft2 = fft_plot(flsig2)
    # datafft3 = fft_plot(flsig3)
    datafft4 = fft_plot(flsig4)
    datafft5 = fft_plot(flsig5)
    datafft6 = fft_plot(flsig6)
    # datafft4 = fft_plot(flsig4)
    # # datafft = fft_plot(data)

    # #plt.plot(datafft, color = 'black')
    # plt.plot(np.abs(datafft3), color = 'green')
    # plt.plot(np.abs(datafft2), color = 'blue')
    # plt.plot(np.abs(datafft1), color = 'red')
    plt.plot(np.abs(datafft), color = 'black')
    plt.plot(np.abs(datafft4), color = 'green')
    plt.plot(np.abs(datafft5), color = 'blue')
    plt.plot(np.abs(datafft6), color = 'red')
    plt.plot(np.abs(fft.fft([1])), color = 'purple')
    
    plt.show()


    player(flsig4)
    player(flsig5)
    player(flsig6)
    # player(np.abs(flsig4))
    # player(np.abs(flsig5))
    # player(np.abs(flsig6))
    # player(flsig4)