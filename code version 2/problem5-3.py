import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.lines as mlines
import sys

fig = plt.figure()
axe = fig.add_subplot(3,1,1)
axe1 = fig.add_subplot(3,1,2)
axe2 = fig.add_subplot(3,1,3)


parent_r = 1
axe.set_aspect('equal')
axe.set_xlim([-parent_r - 0.1, parent_r + 0.1])
axe.set_ylim([-parent_r - 0.1, parent_r + 0.1])
# axe1.set_aspect('equal')
# axe1.set_xlim([-parent_r, parent_r])
axe1.set_ylim([-parent_r - 0.2, parent_r + 0.1])
patches = []

circle = plt.Circle((0,0), radius=parent_r, fc='white', ec='black', linewidth=3)

r = parent_r
base_angle = 45
theta = np.radians(base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
circle2, = axe.plot([x], [y], 'ro')
patches.append(axe.add_patch(circle))



samples = 360
ratemod = 1
interval = 1e-3 #* (ratemod % samples if ratemod % samples != 0 else 1)

freq =  np.deg2rad((1)/interval)
sample_rate = 1
#sample_rate = 180
periods = 8
time = -x + np.linspace(0, 2 * periods * np.pi * freq, num = int(samples/ratemod) * periods)/freq
data = np.cos(time)
axe1.plot(time, data, zorder=0)
circle3, = axe1.plot([x], [y], 'ro')
j = 0
prev_y = 0
def update(i = 0):
    global axe, axe1, time,  circle, circle2, parent_r, ratemod, sample, j, prev_y
    ret = []

    offset = np.radians(ratemod * i)
    r = parent_r
    base_angle = 45
    theta = np.radians(base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    circle2.set_data(x, y)
    temp = circle2,
    ret.append(temp)



    circle3.set_data((time[j]), y)
    if (j % sample_rate == 0):
        if (np.sqrt(np.square(y - data[j])) < 0.2):
            tempr, = axe1.plot([time[j]], [data[j]], 'ro')
            prev_y = y
            temp2 = tempr,
            ret.append(temp2)
            circle2.set_data(x, y)
            temp = circle2,
            ret.append(temp)




    j = (j + 1)%(time.shape[0])

    return ret#[temp, temp2]


anim = animation.FuncAnimation(fig, update, frames=samples, interval=interval * 1e3, blit = False, cache_frame_data = False, repeat = True)
#anim2 = animation.FuncAnimation(fig, update2, frames=360, interval=120, blit = False, cache_frame_data = False, repeat = True)
plt.show()
print("")
