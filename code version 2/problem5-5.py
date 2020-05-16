import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.lines as mlines
import sys

def init_plot():
    fig = plt.figure()
    axe = fig.add_subplot(3,1,1)
    axe1 = fig.add_subplot(3,1,2)
    axe2 = fig.add_subplot(3,1,3)
    parent_r = 1
    axe.set_aspect('equal')
    axe.set_xlim([-parent_r - 0.1, parent_r + 0.1])
    axe.set_ylim([-parent_r - 0.1, parent_r + 0.1])
    axe1.set_ylim([-parent_r - 0.2, parent_r + 0.1])
    axe1.grid(True)
    axe2.set_xlim([-9, 9])
    return fig, axe, axe1, axe2, parent_r

def compute_xy(r = 1, angle = 0, angle_shift = 1, frequency = 1, offset = 0, deg = False):
    '''
    freq = in radians/sec
    '''
    if deg:
        angle = np.deg2rad(angle * angle_shift + offset)
    else:
        angle = angle * angle_shift + offset
    x = r * np.cos(frequency * angle)
    y = r * np.sin(frequency * angle)
    return x, y

def update(i = 0):
    global f0, r, frame_rate, nyquist_freq, data_x, data_y, idx
    i = i * int(nyquist_width)
    return_frame = []
    x = data_x[i]
    y = data_y[i]
    circle2.set_data(x, y)
    return_frame.append(circle2)
    circle3,  = axe1.plot([time[i]], [x], 'ro')
    return_frame.append(circle3)
    return return_frame

#initialization vars
samples = 60
fig, axe, axe1, axe2, r = init_plot()
f0 = 1 #rad/sec

filename = "../problem5/case6.gif"
frame_rate = 0.25 * f0 # how many frames to sample per second
nyquist_freq = frame_rate/2
num_periods = 11
time_dilation = 360
nyquist_width = time_dilation/(nyquist_freq * 2) 
alsfreq= np.abs((nyquist_width - time_dilation)/time_dilation)

axe.set_title(f"constant rotation at f = {f0} Hz, sample freq is {frame_rate} Hz")
axe1.set_title(f"nyquist frequency {nyquist_freq} Hz < 0.5 * signal frequency {f0} Hz")
axe1.set_ylabel(f"x position")
axe1.set_xlabel(f"time")

axe2.set_ylabel(f"x position")
axe2.set_yticks([0, 0.5, 1])
axe2.set_xlabel(f"Frequency")

#init figure 
circle = plt.Circle((0,0), radius=r, fc='black', ec='black', linewidth=3, zorder=0)
time = np.linspace(0, num_periods, num = time_dilation * num_periods + 1)
indexs = np.linspace(0, time_dilation * num_periods, num = time_dilation * num_periods + 1)

ndata = np.argwhere(indexs%nyquist_width == 0).flatten()

shrink = 0.1
data_x = (r - shrink)*np.cos(f0 * 2 * np.pi * time)
data_y = (r - shrink)*np.sin(f0 * 2 * np.pi * time)
data_n = data_x[ndata]

alsf = (np.max(data_n) - np.min(data_n))
print(time_dilation, nyquist_width, alsf/2)

axe.add_patch(circle)
circle2, = axe.plot([0], [1], 'ro', zorder=1)
axe1.plot(time, data_x)

if (nyquist_width > 180):
    axe2.set_title(f"Frequency Domain: alias freq = {np.around(alsfreq, 2)}Hz")
    axe2.stem(np.array([f0]), np.array([1]), 'b', markerfmt='bo', label = "frequency")
    axe2.stem(np.array([alsfreq]), np.array([1]) , 'r', markerfmt='ro', label = "alias freq")
    axe2.stem(np.array([nyquist_freq]), np.array([alsf/2]), 'g', markerfmt='go', label = "nyquist freq")
    axe2.legend()
else:
    axe2.set_title(f"Frequency Domain: no alias")
    axe2.stem(np.array([f0]), np.array([1]), 'b', markerfmt='bo', label = "frequency")
    axe2.stem(np.array([nyquist_freq]), np.array([alsf/2]), 'r', markerfmt='ro', label = "nyquist freq")
    axe2.legend()

anim = animation.FuncAnimation(fig, update, frames=int((time_dilation * num_periods)/nyquist_width), interval=240, blit = False, repeat = True)
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.12, top = 0.9, wspace = 0.35, hspace = 0.8)
#anim.save(filename, writer = 'imagemagick')
plt.show()
