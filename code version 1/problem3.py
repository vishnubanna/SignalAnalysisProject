import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib 
from time import *
import scipy.fft as fft
from fs import forierSeries

#import problem1
#import problem2
def u(t):
    ufunc = np.zeros_like(t)
    for i,time in enumerate(t):
        ufunc[i] = _u(time)
    return ufunc

def _u(t):
    if (t > 0):
        return 1
    else:
        return 0

def h(R, C, t):
    k = (1/(R*C))
    sig = k * np.exp(-k * t) * u(t)
    return sig

def x(t):
    frequency = period
    signal = []
    for time in t:
        signal.append(square(time, period=frequency))
    return np.array(signal)

def square_init(period):
    def square(i):
        t = i%period
        if (t > period/2):
            return 1
        else:
            return -1
    return square

def sin_init(period):
    def sinsig(i):
        return np.sin(2 * np.pi * i/period)
    return sinsig

def saw_init(period):
    def sawtooth(i):
        t = i%period
        return (1/period) * t
    return sawtooth



def fft_plot(signal, t, period = 1):
    font = {'family' : 'normal',
                'size'   : 10, }
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 10))
    axe0 = fig.add_subplot(3,1,1)
    axe0.set_title(f"signal")
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
    freq = fft.fftshift(fft.fftfreq(200 * 8, d = 1/200))
    axe0.plot(t, signal)
    axe2.plot(freq,np.abs(sigfft))#, use_line_collection=True)
    axe3.plot(freq,np.angle(sigfft))#, use_line_collection=True)
    plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.05, top = 0.9, wspace = 0.35, hspace = 0.35)
    plt.show()
    return

def main(): 

    mult = 200
    period = 1
    R = 10000
    C = 5e-6

    #cutoff frequ = 3.183098862
    n = 100
    #fs = forierSeries(period, sects = 8, points_per_period=mult, convolved=True)
    fs1 = forierSeries(period, sects = 8, points_per_period=mult)
    fs1.setsignal(square_init)
    fs1.forierserier(n)

    fs1.plot()

    fs = forierSeries(period, sects = 8, points_per_period=mult)
    fs.setsignal(square_init)
    time = fs.t
    start = time[0]
    stop = time[-1]

    print(start, stop)

    f = fs.insig
    g = h(R, C, time)
    
    fft_plot(g, time)
    
    conv1 = np.convolve(f, g)/np.sum(g)
    i = 0
    while conv1[i] == 0:
        i += 1
    i -= 1
    fs.insig = conv1[i : i + fs.sects * fs.ppp]
    fs.forierserier(n)
    fs.insig = f
    fs.a = np.array(fs.a) - np.array(fs1.a)
    fs.b = np.array(fs.b) - np.array(fs1.b)
    fs.a_min = np.array(fs.a_min) - np.array(fs1.a_min)
    fs.b_min = np.array(fs.b_min) - np.array(fs1.b_min)
    fs.plot()



    #plt.plot(fs.insig)
    return


if __name__ == "__main__":
    main()