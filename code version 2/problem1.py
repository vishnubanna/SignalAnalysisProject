import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os
from time import *
import shutil
import re

#plt.style.use('dark_background')

# start = -50
# stop = 50
# mult = 1

fig = plt.figure() 
axe0 = fig.add_subplot(3,1,1)
axe0.set_ylable('amplitude')
axe1 = fig.add_subplot(3,1,2)
axe3 = fig.add_subplot(3,1,2)
axe2 = fig.add_subplot(3,1,3)

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

def f(t, amplitude = 1, a = 0, b = 1):
    t = np.array(t)
    f_t = np.zeros_like(t)
    for i,time in enumerate(t):
        f_t[i] = amplitude * (_u(time - a) - _u(time - b))
    return f_t

def g(t, amplitude = 1, a = 0, b = 1):
    t = np.array(t)
    g_t = np.zeros_like(t)
    for i,time in enumerate(t):
        g_t[i] = amplitude * (_u(time - a) - _u(time - b))
    return g_t

def convolve(f_t, g_t, time, start, stop , mult):

    #global mult
    ims = []
    backshift = f_t.shape[0]#int((stop - start) * mult)
    # f_g_t = np.zeros((backshift * 3))
    # f_g_t2 = np.zeros((backshift * 3))
    f_g_t = np.zeros((backshift * 3))
    f_g_t2 = np.zeros((backshift * 3))

    time2 = np.linspace(0, backshift * 2, num = (backshift * 2))
    #time2 = np.linspace(0, backshift * 2 - 1, num = (backshift * 2) - 1)
    print(f_g_t.shape)
    print(f_t.shape)

    flipshift = np.flip(f_t)

    f_g_t[0:backshift] = flipshift

    count = 0 
    axe1.set_ylim(top = max(np.max(g_t), np.max(f_t)) + 1)
    axe1.set_ylim(bottom = min(np.min(g_t), np.min(f_t)) - 1)
    axe0.set_ylim(top = max(np.max(g_t), np.max(f_t)) + 1)
    axe0.set_ylim(bottom = min(np.min(g_t), np.min(f_t)) - 1)
    axe0.grid(True)
    axe1.grid(True)
    axe2.grid(True)

    for i in range(backshift * 2):
        f_g_t = np.zeros_like(f_g_t, dtype=float)
        f_g_t[i:backshift+i] = flipshift
        f_g_t2[i] = np.sum(f_g_t[backshift:2*backshift]*g_t)
        if(np.sum(f_g_t2) != 0 and count == 0):
            count = i%backshift

    axe2.set_ylim(top = np.max(f_g_t2)+1 * mult)
    axe2.set_ylim(bottom = np.min(f_g_t2)-1 * mult)
    f_g_t2 = np.zeros_like(f_g_t2, dtype=float)
    time2 += time[count]

    for i in range(backshift * 2):
        f_g_t = np.zeros_like(f_g_t, dtype=float)
        f_g_t[i:backshift+i] = flipshift
        f_g_t2[i] = np.sum(f_g_t[backshift:2*backshift]*g_t)

        if(np.sum(f_g_t[backshift:2*backshift]) != 0):
            im0, = axe0.plot(time, f_t , color = 'blue')
            im1, = axe1.plot(time, f_g_t[backshift:2*backshift], color = 'blue')
            im3, = axe3.plot(time, g_t, color = 'red')
            im2, = axe2.plot(time2, f_g_t2[0:2*backshift], color = 'black')
            ims.append([im0, im1,im3, im2])
            count += 1
        
    return ims, f_g_t2, time2

def main(t, f_t, g_t, start, stop, mult):
    # global mult, start, stop
    # mult = 3
    # start = -10
    # stop = 10
    # t = np.linspace(start, stop, num = (stop - start) * mult) #- stop/2
    # f_t = f(t, a=-5, b=10)
    # g_t = g(t, a=0, b=2, amplitude = 1)

    ims, conv, newt = convolve(f_t, g_t, t, start, stop, mult)

    #ani = animation.ArtistAnimation(fig, ims, repeat_delay = 0, interval = int(50/(2*mult)))
    #plt.show()
    return ims, conv, newt, ani


def square_init(period):
    def square(i):
        t = i%period
        if (t > period/2):
            return 1
        else:
            return -1
    return square

def saw_init(period):
    def sawtooth(i):
        t = i%period
        return (1/period) * t
    return sawtooth

def getsignal(t_, f):
    signal = []
    for time in t_:
        signal.append(f(time))
    return np.array(signal)

def gauss_sig(a, t):
    #return np.sqrt(np.pi/a) * np.exp(-np.square(t)/(4 * a))
    return np.sqrt(np.pi/a) * np.exp(-np.square(t)/(4 * a))

def main2():
    # global mult, start, stop
    mult = 15
    start = -5
    stop = 5
    t = np.linspace(start, stop, num = (stop - start) * mult) #- stop/2
    f_t = ((np.sin(t) + np.cos(2 * np.pi *t)) * gauss_sig(2, -t) * gauss_sig(2, t)) * u(t) #gauss_sig(1, t)#f(t, a=-5, b=10)
    #g_t = getsignal(t, square_init(5)) + getsignal(t, saw_init(10)) + np.cos(2 * np.pi *t) + gauss_sig(3, t) #g(t, a=0, b=2, amplitude = 1)
    g_t = (np.sin(t) + np.cos(2 * np.pi *t)) * gauss_sig(2, -t) * gauss_sig(2, t) 
    ims, conv, newt = convolve(f_t, g_t, t, start, stop, mult)

    ani = animation.ArtistAnimation(fig, ims, repeat_delay = 0, interval = int(50/(2*mult)))
    plt.show()
    return


if __name__ == "__main__":
    main2()