import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from time import *
from fs import forierSeries

#import problem1
#import problem2


def convolve(f_t, g_t, time, start, stop , mult):
    fig = plt.figure() 
    axe0 = fig.add_subplot(3,1,1)
    axe1 = fig.add_subplot(3,1,2)
    axe3 = fig.add_subplot(3,1,2)
    axe2 = fig.add_subplot(3,1,3)
    #global mult
    ims = []
    backshift = f_t.shape[0] #int((stop - start) * mult)
    # f_g_t = np.zeros((backshift * 3))
    # f_g_t2 = np.zeros((backshift * 3))
    f_g_t = np.zeros((backshift * 3))
    f_g_t2 = np.zeros((backshift * 3))

    time2 = np.linspace(start, (start + (backshift) * 2), num = (backshift * 2))
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
            if (backshift <= 40):
                im0, = axe0.plot(time, f_t , color = 'blue')
                im1, = axe1.plot(time, f_g_t[backshift:2*backshift], color = 'blue')
                im3, = axe3.plot(time, g_t, color = 'red')
                im2, = axe2.plot(time2, f_g_t2[0:2*backshift], color = 'black')
                ims.append([im0, im1,im3, im2])
                count += 1
            elif (i % ((backshift * 2)//20) == 0):
                im0, = axe0.plot(time, f_t , color = 'blue')
                im1, = axe1.plot(time, f_g_t[backshift:2*backshift], color = 'blue')
                im3, = axe3.plot(time, g_t, color = 'red')
                im2, = axe2.plot(time2, f_g_t2[0:2*backshift], color = 'black')
                ims.append([im0, im1,im3, im2])
                count += 1
    ani = animation.ArtistAnimation(fig, ims, repeat_delay = 0, interval = int(50/(2*mult)))
    plt.show()
    return




def u(t):
    ufunc = np.zeros_like(t)
    for i,time in enumerate(t):
        ufunc[i] = _u(time)
    return ufunc

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

def main(): 

    mult = 200
    period = 1
    R = 10000
    C = 5e-6

    #fs = forierSeries(period, sects = 8, points_per_period=mult, convolved=True)
    fs = forierSeries(period, sects = 8, points_per_period=mult)
    #fs.setsignal(square_init)
    fs.setsignal(saw_init)
    time = fs.t
    start = time[0]
    stop = time[-1]

    print(start, stop)

    f = fs.insig
    g = h(R, C, time)
    
    conv1 = np.convolve(f, g)
    i = 0
    while conv1[i] == 0:
        i += 1
    i -= 1
    fs.insig = conv1[i : i + fs.sects * fs.ppp]

    fs.forierserier(10)
    fs.plot()

    plt.plot(fs.insig)
    return


if __name__ == "__main__":
    main()