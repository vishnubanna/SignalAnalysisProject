import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.lines as mlines

fig = plt.figure()
axe0 = fig.add_subplot(3,1,1)
axe1 = fig.add_subplot(3,1,2)
axe3 = fig.add_subplot(3,1,2)
axe2 = fig.add_subplot(3,1,3)

def update(offset = 0):
    axe = plt.gca(animated = True)
    patches = []
    offset = np.radians(offset)
    circle = plt.Circle((0,0), radius=1, fc='white', ec='black', linewidth=3)
    circle2 = plt.Circle((0,0), radius=0.1 , fc='gray')

    r2 = 1
    r = 0.5
    base_angle = 45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line1 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
    
    base_angle = 2*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line2 = mlines.Line2D([x, x2], [y, y2], mfc = color, mec = color, color = color, linewidth=3)
    
    base_angle = 3*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line3 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
    
    base_angle = 4*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line4 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
    
    base_angle = 5*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line5 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
    
    base_angle = 6*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line6 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
    
    base_angle = 7*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line7 = mlines.Line2D([x, x2], [y, y2], color = color, linewidth=3)
    
    base_angle = 8*45
    color = 'black'
    theta2 = np.radians(90 - base_angle) + offset
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    theta = np.radians(90 - base_angle) + offset
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

    r = 0.5
    base_angle = 72
    theta = np.radians(90 - base_angle) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line10 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

    theta = np.radians(90 - 2*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line11 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

    theta = np.radians(90 - 3*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line12 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

    theta = np.radians(90 - 4*(base_angle)) + offset
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    line13 = mlines.Line2D([0, x], [0, y], color = "gray", linewidth=3)

    theta = np.radians(90 - 5*(base_angle)) + offset
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

    plt.axis('scaled')
    return patches

def unim(i):
    global fig
    fig.clf()
    return update(i)



anim = animation.FuncAnimation(fig, update, init_func=update, frames=360, interval=20, blit = True, cache_frame_data = False)
plt.show()