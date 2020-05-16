import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os
from time import *
import shutil
import re
from fs import forierSeries

points_per_period = 10000
t_ = np.concatenate((np.linspace(-6*np.pi,-4*np.pi, points_per_period), np.linspace(-4*np.pi,-2*np.pi, points_per_period), np.linspace(-2*np.pi,-0*np.pi, points_per_period), np.linspace(0*np.pi,2*np.pi, points_per_period), np.linspace(2*np.pi,4*np.pi, points_per_period), np.linspace(4*np.pi,6*np.pi, points_per_period)))
period = 2 * np.pi

def sawtooth(i, period = period):
    t = i%period
    return (1/period) * t

def square(i, period = period):
    t = i%period
    if (t > period/2):
        return 1
    else:
        return -1

def bn(n, period = period):
    n = int(n)
    if (n%2 != 0):
        return 4/(np.pi * n)
    else:
        return 0

def wn(n, period = period):
    wn = (2 * np.pi * n)/period
    return wn

def intergralco(ft, time, n = None, signal = None, period = 2 * np.pi):
    ft = np.asarray(ft)
    #print(ft.shape)
    if (signal == None):
        return np.sum(ft)/period
    elif(signal == 'cos'):
        gt = np.cos(wn(n) * time)
        #print(gt.shape)
        return np.sum(gt * ft) * 2/period
    elif(signal == 'sin'):
        gt = np.sin(wn(n) * time)
        #print(gt.shape)
        return np.sum(gt * ft) * 2/period

def forierserier(n_max, x, period):
    x = np.array(x)
    t = t_ # expand_T
    a0 = intergralco(x, t)
    a = [a0]
    b = [0]
    
    for i in range(1, n_max):
        a.append(intergralco(x, t, n = i, signal="cos", period=period))
        b.append(intergralco(x, t, n = i, signal="sin", period=period))
    
    a = np.array(a)
    b = np.array(b)
    print(f"+ {a[i]}")
    psum = a0 * np.ones_like(t_)
    for i in range(1,len(a)):
        psum += a[i] * np.cos(wn(i) * t_)
        print(f"+ {a[i]} * cos(w * {i} * t)")
        psum += b[i] * np.sin(wn(i) * t_)
        print(f"+ {b[i]} * sin(w * {i} * t)")
    val = psum
    return np.array(val)/points_per_period

def getsignal(t_, f = square):
    signal = []
    for time in t_:
        signal.append(f(time))
    return np.array(signal)

def f(t):
    return sawtooth(t)


def main():
    signal = getsignal(t_, f)
    fs = forierSeries(period, sects = 8, points_per_period=mult)
    #series = forierserier(10, signal, period)

    # signal = []
    # for time in t_:
    #     signal.append(sawtooth(time))
    plt.plot(t_, signal)
    plt.plot(t_, np.array(series))
    plt.show()
    return



if __name__ == "__main__":
    main()