import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import matplotlib.animation as animation
from basic_units import radians, degrees, cos
import os
import sys
from time import *
import shutil
import re

class forierSeries():
    def __init__(self, period, sects = 2, offset = 0, signal = None, time = None, points_per_period = 1000, convolved = False, mod_div = True):
        if (convolved):
            self.ppp = 200
        else:
            self.ppp = points_per_period
        self.period = period
        self.omega = (2 * np.pi)/period
        self.frequency = 1/self.period
        self.signal_func = None
        self.a = []
        self.b = []
        self.a_min = []
        self.b_min = []
        self.fs = []
        self.n_max = 0

        self.sects = sects
        self.offset = offset
        
        if mod_div:
            if self.ppp >= 200:
                self.div = self.sects * self.ppp
            else:
                self.div = self.ppp
        else:
            self.div = self.ppp
        #self.div = (self.omega/np.pi * self.ppp)
            

        #defaults to 2 periods
        if (type(time) == type(None)):
            lim = (self.sects * self.period / 2) + self.period * self.offset
            #self.t = np.concatenate((np.linspace(-period, 0, points_per_period), np.linspace(1/(points_per_period), period, points_per_period - 1)))
            self.t = np.linspace(-lim, lim, points_per_period * sects)
        else:
            self.t = time
        self.insig = signal
        return

    @property
    def D(self):
        D_neg = np.vectorize(complex)(self.a_min, self.b_min)
        D = np.vectorize(complex)(self.a, self.b)
        D[0] = D[0]/2
        D = np.concatenate((D_neg, D))
        D = D/2
        return D 

    def setsignal(self, signal):
        self.signal_func = signal(self.period)
        self.getsignal()
        return

    def bn(self, n):
        n = int(n)
        if (n%2 != 0):
            return 4/(np.pi * n)
        else:
            return 0

    def wn(self, n):
        wn = (2 * np.pi * n)/self.period
        return wn


    def getsignal(self):
        signal = []
        for time in self.t:
            signal.append(self.signal_func(time))
        self.insig = np.array(signal)
        return #np.array(signal)
    
    def getsignals(self, t):
        signal = []
        for time in t:
            signal.append(self.signal_func(time))
        return np.array(signal)
        #np.array(signal)

    def intergralco(self, ft, n = None, signal = None):
        ft = np.asarray(self.insig)
        if (self.ppp < 200):
            t = np.linspace(0, self.period - 1/self.ppp, self.ppp)
            ft = self.getsignals(t)
        else:
            t = self.t
        #print(ft.shape)
        if (signal == None):
            return np.sum(ft)/self.period
        elif(signal == 'cos'):
            gt = np.cos(self.wn(n) * t)
            #print(gt.shape)
            return np.sum(gt * ft) * 2/self.period
        elif(signal == 'sin'):
            gt = np.sin(self.wn(n) * t)
            #print(gt.shape)
            return np.sum(gt * ft) * 2/self.period
    
    def forierserier(self, n_max):
        self.n_max = n_max
        x = np.array(self.insig)
        a0 = self.intergralco(x, self.t)


        div = self.div

        for i in range(-n_max + 1, 0):
            self.a_min.append(self.intergralco(x, n = i, signal="cos")/div)
            self.b_min.append(self.intergralco(x, n = i, signal="sin")/div)

        a = [a0]
        b = [0]
        
        if (n_max > self.ppp * 0.9):
            n_max = self.ppp

        for i in range(1, n_max):
            a.append(self.intergralco(x, n = i, signal="cos"))
            b.append(self.intergralco(x, n = i, signal="sin"))
        
        self.a = np.array(a)
        self.b = np.array(b)
        print(f"+ {self.a[i]}")
        psum = a0 * np.ones_like(self.t)
        # self.a = self.a/(self.omega/np.pi * self.ppp)
        # self.b = self.b/(self.omega/np.pi * self.ppp)
        
        for i in range(1,len(a)):
            psum += self.a[i] * np.cos(self.wn(i) * self.t)
            #print(f"+ {a[i]} * cos(w * {i} * t)")
            psum += self.b[i] * np.sin(self.wn(i) * self.t)
            #print(f"+ {b[i]} * sin(w * {i} * t)")
        val = psum
        #if (self.ppp >= 130):
        self.a = self.a/div
        self.b = self.b/div
        self.fs = np.array(val)/div 
        # else:
        #     self.fs = np.array(val)/(2 * np.pi)
        return self.fs
    
    def plot(self, axe = None):
        font = {'family' : 'normal',
                'size'   : 10, }
        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(12, 10))
        axe0 = fig.add_subplot(4,1,1)
        axe0.set_title(f"Forier Expansion mapped to signal, n = {self.n_max}")
        axe0.set_ylabel("y")
        axe0.set_xlabel("t", x = 1, y = -2)
        axe1 = fig.add_subplot(4,1,2)
        axe1.set_title("noise :: Forier Expansion - signal")
        axe1.set_ylabel("Difference")
        axe1.set_xlabel("t", x = 1, y = -2)
        axe2 = fig.add_subplot(4,1,3) 
        axe2.set_title("Frequency domain Amplitude")
        axe2.set_ylabel("Amplitude")
        axe2.set_xlabel("w", x = 1, y = -2)
        axe3 = fig.add_subplot(4,1,4) 
        axe3.set_title("Frequency domain Phase")
        axe3.set_ylabel("Phase")
        axe3.set_xlabel("w", x = 1, y = -2)
        print(self.D)
        if axe == None:
            axe0.plot(self.t, self.insig)
            axe0.plot(self.t, self.fs)
            axe1.plot(self.t, np.abs(self.insig - self.fs), color = "red")
            #n = np.linspace(0,self.n_max, self.n_max)
            n = np.linspace(-self.n_max + 1,self.n_max - 1, self.n_max * 2 - 1)
            axe2.stem(self.wn(n)/np.pi, np.abs(self.D))#, use_line_collection=True)
            axe3.stem(self.wn(n)/np.pi, np.angle(self.D))#, use_line_collection=True)
            plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.05, top = 0.9, wspace = 0.35, hspace = 0.35)
            plt.show()
        return

    def __add__ (self, other):
        new_fs = forierSeries(self.period, self.sects, self.offset, self.insig, self.t);
        new_fs.a_min = self.a_min + other.a_min
        new_fs.a = self.a + other.a
        new_fs.b_min = self.b_min + other.b_min
        new_fs.b = self.b + other.b
        new_fs.fs = self.fs + other.fs
        new_fs.div = self.div
        new_fs.n_max = self.n_max
        return new_fs
    
    def __sub__ (self, other):
        new_fs = forierSeries(self.period, self.sects, self.offset, self.insig - other.insig, self.t);
        new_fs.a_min = np.array(self.a_min) - np.array(other.a_min)
        new_fs.a = np.array(self.a) - np.array(other.a)
        new_fs.b_min = np.array(self.b_min) - np.array(other.b_min)
        new_fs.b = np.array(self.b) - np.array(other.b)
        new_fs.fs = np.array(self.fs) - np.array(other.fs)
        new_fs.div = self.div
        new_fs.n_max = self.n_max
        return new_fs

    def __str__(self):
        a = self.a#/self.div#(self.omega/np.pi * self.ppp)
        b = self.b#/self.div

        rounder = 4
        out = f"{np.around(a[0], rounder)}"
        for i in range(1,len(a)):
            if a[i] >= 0:
                out += f" + {np.around(a[i], rounder)}cos({i}w * t)"
            else:
                out += f" - {np.abs(np.around(a[i], rounder))}cos({i}w * t)"
            if b[i] >= 0:
                out += f" + {np.around(b[i], rounder)}sin({i}w * t)"
            else:
                out += f" - {np.abs(np.around(b[i], rounder))}sin({i}w * t)"
        return out
        
    def __repr__(self):
        return str(self)

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

def sin_init(period):
    def sin(i):
        return np.sin((2 * np.pi * i)/period)
    return sin

# f = forierSeries(2)
# f.setsignal(square_init)
# f.forierserier(30)
# #f.plot()
# print(f.insig)


