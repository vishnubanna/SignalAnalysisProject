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
axe.set_xlim([-parent_r, parent_r])
axe.set_ylim([-parent_r, parent_r])
# axe1.set_aspect('equal')
# axe1.set_xlim([-parent_r, parent_r])
axe1.set_ylim([-parent_r, parent_r])
patches = []

circle = plt.Circle((0,0), radius=parent_r, fc='white', ec='black', linewidth=3)
circle2 = plt.Circle((0,0), radius=parent_r/10 , fc='gray')

r2 = parent_r
r = parent_r/2
base_angle = 45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line1 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)

base_angle = 2*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line2 = mlines.Line2D([x, x2], [y, y2], mfc = color, mec = color, color = color, linewidth=3)

base_angle = 3*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line3 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)

base_angle = 4*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line4 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)

base_angle = 5*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line5 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)

base_angle = 6*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line6 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)

base_angle = 7*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line7 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)

base_angle = 8*45
color = 'black'
theta2 = np.radians(90 - base_angle)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line8 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
patches.append(axe.add_line(line1))
patches.append(axe.add_line(line2))
patches.append(axe.add_line(line3))
patches.append(axe.add_line(line4))
patches.append(axe.add_line(line5))
patches.append(axe.add_line(line6))
patches.append(axe.add_line(line7))
patches.append(axe.add_line(line8))



r = parent_r/2
base_angle = 72
theta = np.radians(90 - base_angle)
x = r * np.cos(theta)
y = r * np.sin(theta)
line10 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)
circle3 = plt.Circle((x,y), radius=0.01 , fc='red', zorder = 2)


theta = np.radians(90 - 2*(base_angle))
x = r * np.cos(theta)
y = r * np.sin(theta)
line11 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

theta = np.radians(90 - 3*(base_angle))
x = r * np.cos(theta)
y = r * np.sin(theta)
line12 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

theta = np.radians(90 - 4*(base_angle))
x = r * np.cos(theta)
y = r * np.sin(theta)
line13 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

theta = np.radians(90 - 5*(base_angle))
x = r * np.cos(theta)
y = r * np.sin(theta)
line14 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)
patches.append(axe.add_line(line10))
patches.append(axe.add_line(line11))
patches.append(axe.add_line(line12))
patches.append(axe.add_line(line13))
patches.append(axe.add_line(line14))

patches.append(axe.add_patch(circle))
patches.append(axe.add_patch(circle2))
patches.append(axe1.add_patch(circle3))

ratemod = 500
# time = np.linspace(0, 500* 2 * np.pi, num = 100)/np.radians(ratemod * 60)
# data = np.sin(time)
# axe1.plot(data, zorder=0)
def update(i = 0):
    global axe, axe1, time, ratemod, parent_r, line1, line2,line3,line4,line5, line6,line7,line8,line10,line11, line12, line13, line14 
    patches = []
    sys.stdout.write(f"\r60 fps -> each frame coresponds to {ratemod} degrees, so signal frequency is {60 * ratemod}Hz ")
    sys.stdout.flush()
    offset = np.radians(ratemod * i)

    patches.append(circle2)
    patches.append(circle)
    r2 = parent_r
    r = parent_r/2
    base_angle = 45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line1.set_data([x, x2], [y, y2])
    
    base_angle = 2*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line2.set_data([x, x2], [y, y2])
    
    base_angle = 3*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line3.set_data([x, x2], [y, y2])
    
    base_angle = 4*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line4.set_data([x, x2], [y, y2])
    
    base_angle = 5*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line5.set_data([x, x2], [y, y2])
    
    base_angle = 6*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line6.set_data([x, x2], [y, y2])
    
    base_angle = 7*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line7.set_data([x, x2], [y, y2])
    
    base_angle = 8*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line8.set_data([x, x2], [y, y2])
    circle3 = plt.Circle((x2,y2), radius=0.1 , fc='red')
    patches.append(axe.add_line(line1))
    patches.append(axe.add_line(line2))
    patches.append(axe.add_line(line3))
    patches.append(axe.add_line(line4))
    patches.append(axe.add_line(line5))
    patches.append(axe.add_line(line6))
    patches.append(axe.add_line(line7))
    patches.append(axe.add_line(line8))

    r = parent_r/2
    base_angle = 72
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line10.set_data([0, x], [0, y])
    

    theta = np.radians(90 - 2*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line11.set_data([0, x], [0, y])
    

    theta = np.radians(90 - 3*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line12.set_data([0, x], [0, y])
    

    theta = np.radians(90 - 4*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line13.set_data([0, x], [0, y])

    theta = np.radians(90 - 5*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line14.set_data([0, x], [0, y])
    # circle3 = plt.Circle((x,y), radius=0.1 , fc='red')

    patches.append(axe.add_line(line10))
    patches.append(axe.add_line(line11))
    patches.append(axe.add_line(line12))
    patches.append(axe.add_line(line13))
    patches.append(axe.add_line(line14))

    patches.append(axe.add_patch(circle3))

    return patches


# def update2(i = 0):
#     global time, ratemod, axe1

#     return 

# # 60 fps  -> each frame coresponds to 1000 degrees, so signal frequency is 60000Hz
# framedata = animate(360)
#anim = animation.FuncAnimation(fig, update, frames=360, interval=16.667, blit = True, cache_frame_data = False, repeat = True)
anim = animation.FuncAnimation(fig, update, frames=360, interval=120, blit = True, cache_frame_data = False, repeat = True)
#anim2 = animation.FuncAnimation(fig, update2, frames=360, interval=12, blit = True, cache_frame_data = False, repeat = True)
plt.show()
