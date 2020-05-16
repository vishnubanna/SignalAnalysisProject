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

def sawtooth(i, period = 2 * np.pi):
    t = i%period
    return (1/period) * t

def square_init(period):
    def square(i):
        t = i%period
        if (t > period/2):
            return 1
        else:
            return -1
    return square

def saw_init(period):
    def saw(i):
        t = i%period
        return (1/period) * t
    return saw


def getsignal(t_, f = square_init(2 * np.pi)):
    signal = []
    for time in t_:
        signal.append(f(time))
    return np.array(signal)

def f(t):
    return sawtooth(t)


def main():
    points_per_period = 1000
    period = 2*np.pi
    fs = forierSeries(period, sects = 6, points_per_period=points_per_period, mod_div=False)
    fs.setsignal(saw_init)

    fs.forierserier(10)
    print(fs)
    fs.plot()

    return



if __name__ == "__main__":
    main()