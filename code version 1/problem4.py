import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from time import *
import matplotlib
import scipy.fft as fft

def gauss_sig(a, t):
    #return np.sqrt(np.pi/a) * np.exp(-np.square(t)/(4 * a))
    return np.exp(-np.square(t) * a)


def fft_plot(signal, t, mult, a = 1):
    font = {'family' : 'normal',
                'size'   : 10, }
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 10))
    axe0 = fig.add_subplot(3,1,1)
    axe0.set_title(f"signal, a = {a}")
    axe0.set_ylabel("y")
    axe0.set_xlabel("t", x = 1, y = -2)
    axe2 = fig.add_subplot(3,1,2) 
    axe2.set_title("Frequency domain Amplitude")
    axe2.set_ylabel("Amplitude")
    axe2.set_xlabel("w", x = 1, y = -2)
    axe3 = fig.add_subplot(3,1,3) 
    axe3.set_title("Frequency domain Phase")
    axe3.set_ylabel("Phase")
    axe3.set_xlabel("w", x = 1, y = -2)
    
    sigfft = fft.fftshift(fft.fft(signal))
    freq = fft.fftshift(fft.fftfreq(t.shape[-1], 1/mult))
    axe0.plot(t, signal)
    axe2.plot(freq,np.abs(sigfft))#, use_line_collection=True)
    axe3.plot(freq,np.angle(sigfft))#, use_line_collection=True)
    plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.05, top = 0.9, wspace = 0.35, hspace = 0.35)
    plt.show()
    return

def prob(a, start, stop, mult):
    exp = 1
    time = np.linspace(start * exp, stop * exp, (stop - start) * exp * mult)
    sig = gauss_sig(a, time)
    fft_plot(sig, time, mult, a)

    a = 1
    exp = a
    time = np.linspace(start * exp, stop * exp, (stop - start) * exp * mult * 10)
    sig = gauss_sig(10, time)
    fft_plot(sig, time, mult * 100, 10)

    a = 1
    exp = a
    time = np.linspace(start * exp, stop * exp, (stop - start) * exp * mult * 1000)
    sig = gauss_sig(1000, time)
    fft_plot(sig, time, mult * 1000, 1000)
    # ftsig = fft.fftshift(fft.fft(sig))

    # fig = plt.figure() 
    # axe0 = fig.add_subplot(2,1,1)
    # axe1 = fig.add_subplot(2,1,2)
    # axe1 = fig.add_subplot(2,1,2)

    # axe0.plot(time,sig)
    # freq = fft.fftshift(fft.fftfreq(time.shape[-1], 1/mult))
    # axe1.plot(freq,np.abs(ftsig))
    # plt.show()
    return

if __name__ == "__main__":
    prob(1, -5, 5, 10)