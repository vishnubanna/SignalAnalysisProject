import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.lines as mlines
import sys

fig = plt.figure()
axe = fig.add_subplot(3,1,1)
axe1 = fig.add_subplot(3,1,2)

parent_r = 1
axe.set_aspect('equal')
axe.set_xlim([-parent_r - 0.1, parent_r + 0.1])
axe.set_ylim([-parent_r - 0.1, parent_r + 0.1])
axe1.set_ylim([-parent_r - 0.2, parent_r + 0.1])

circle = plt.Circle((0,0), radius=parent_r, fc='white', ec='black', linewidth=3, zorder=0)
axe.add_patch(circle)

r = parent_r
base_angle = 8 * 45
x = 0#r * np.cos(np.radians(base_angle))
y = 1#r * np.sin(np.radians(base_angle))
circle2, = axe.plot([x], [y], 'ro', zorder=1)

samples = 360
ratemod = 4
periods = 4
interval = 1e-3
sample_rate =  1#8# nyquist frequency: the fraction of the frequency
multiply_samples = 10
j = 0
freq =  np.deg2rad((1 * ratemod * sample_rate)/interval)

time = np.linspace(0, (2 * np.pi * periods), num = int(samples * periods * multiply_samples))
data = r * np.sin(time)

axe1.plot(time, data)
data2 = r * np.sin(time*sample_rate)
axe1.plot(time, data2)
circle3, = axe1.plot([x], [y], 'ro')
print(x)

axis_cross = 0
place_dot = 0
def update(i = 0):
    global axe, axe1, time, circle2, circle3, parent_r, r, base_angle, ratemod, sample, j, prev_y, axis_cross, place_dot
    ret = []

    offset = np.radians(ratemod * i * sample_rate)
    #offset2 = np.radians(ratemod * i)
    theta = np.radians(base_angle) + offset

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    val = y - data[j]
    lastval = axis_cross - data[j]

    circle3.set_data(time[j], y)
    ret.append(circle3,)
    
    if (sample_rate < 1):
        lim = 0.05
        rounder = 2
    else:
        lim = 0.1
        rounder = 2
    
    
    if (np.abs(np.around(data2[j], rounder) - np.around(data[j], rounder)) < lim):
        circle2.set_data(x, y)
        ret.append(circle2,)
        #circle3.set_data(time[j], data[j])


        temp, = axe1.plot([time[j]], [data[j]], 'ro')
        ret.append(temp, )
    axis_cross = val


    j = int(i * ratemod * multiply_samples) % time.shape[0] 
    sys.stdout.write(f"\rnyquist rads per sec : {freq:5}, rads per sec : {freq*sample_rate:5}, val : {np.abs(np.round(np.around(data2[j], rounder)) - np.round(np.around(data[j], rounder)))}")
    sys.stdout.flush()
    return ret


anim = animation.FuncAnimation(fig, update, frames=samples * periods, interval=interval * 1e3, blit = False, cache_frame_data = False, repeat = True)
plt.show()
print("")
