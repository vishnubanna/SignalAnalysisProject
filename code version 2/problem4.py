import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from time import *
from fs import forierSeries


def gauss_sig(a, t):
    #return np.sqrt(np.pi/a) * np.exp(-np.square(t)/(4 * a))
    return np.exp(-np.square(t) * a)

def prob(a, start, stop, mult):
    time = np.linspace(start, stop, (stop - start) * mult)
    sig = gauss_sig(a, time)

    ftsig = np.fft.fft(sig)

    fig = plt.figure() 
    axe0 = fig.add_subplot(2,1,1)
    axe1 = fig.add_subplot(2,1,2)
    axe0.plot(time,sig)
    axe1.plot(time,np.abs(ftsig))
    plt.show()
    return

if __name__ == "__main__":
    prob(1000, -5, 5, 10)