import matplotlib.pyplot as plt
import random
from PIL import Image
from PIL import ImageOps 
import time
import pandas as pd 
import math
import os

NUM_IMAGES = 200
NUM_TRAJECTORIES_PER_IMAGE = 1

IMAGE_WIDTH, IMAGE_HEIGHT = 500, 500

dt = .01
G = 1000
EARTH_ACCEL = -9.8
X_start_pos = 0
Y_start_pos = 0
NUM_PARTIAL_TRAJECTORY = 8

PATH = "/Users/chopinboy/Desktop/pyro/"


def getXYAccel(x, y, planet):
    dist = ((planet["x"] - x)**2 + (planet["y"] - y)**2)**.5
    x_dist = planet["x"] - x
    y_dist = planet["y"] - y
    
    accel_mag = G*planet["m"]/(dist**2)
    x_accel = accel_mag * x_dist/dist
    y_accel = accel_mag * y_dist/dist

    return x_accel, y_accel


def getInitialParameters():
    x = X_start_pos
    y = Y_start_pos
    vx = random.random()*50 + 20
    vy = random.random()*100 + 30
    return (x,y,vx,vy)
    

def getTrajectory(initial_parameters, planets):
    x, y, vx, vy = initial_parameters

    trajectory = []
    t = 0
    while x >= 0 and x <= IMAGE_WIDTH and y >= 0 and y <= IMAGE_HEIGHT:
        # Get accelerations from planets 
        x_accel = 0
        y_accel = 0
        for planet in planets:
            x_acc, y_acc = getXYAccel(x, y, planet)
            x_accel += x_acc
            y_accel += y_acc
    
        # Update velocities with accelerations from planets
        vx = vx + dt*x_accel
        vy = vy + dt*y_accel

        # Update y velocity with Gravity
        vy = vy + dt*EARTH_ACCEL

        # Update position
        x = x + dt*vx
        y = y + dt*vy

        trajectory.append([x,y])
    return trajectory



# planet1 = {"x": 300, "y": 400, "m": 100}
# planet2 = {"x": 400, "y": 200, "m": 100}
# planets = [planet1, planet2]
planets = []

initial_parameters_for_csv = []

tic = time.clock()
for iter in range(NUM_IMAGES):
    # Draw planets
    for planet in planets:
        circle = plt.Circle((planet["x"], planet["y"]), radius=planet["m"] * .1, fc='y')
        plt.gca().add_patch(circle)

    # initialize parameters
    initial_parameters = getInitialParameters()
    initial_parameters_for_csv.append(initial_parameters)
    traj_points = getTrajectory(initial_parameters, planets)
    for i in range(NUM_PARTIAL_TRAJECTORY):
        num_pts = len(traj_points)
        sub_line_len = math.floor(num_pts / NUM_PARTIAL_TRAJECTORY)
        if i + 1 == NUM_PARTIAL_TRAJECTORY:
            sub_line = traj_points
        else:
            sub_line = traj_points[0: (i + 1) * sub_line_len]
        line = plt.Polygon(sub_line, closed=None, fill=None, linewidth=10)
        plt.gca().add_line(line)

        # Create image
        plt.axis('off')
        plt.axis('scaled')
        plt.xlim(0, IMAGE_WIDTH)
        plt.ylim(0, IMAGE_HEIGHT)

        if i == 0:
            folder_name = PATH + "partial_trajectories/example_" + str(iter)
            os.mkdir(folder_name)

        image_file_name = folder_name + "/partial_" + str(i) + ".png"
        plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
    
        plt.gcf().clear()

        #Resize images to 28x28
        img = Image.open(image_file_name)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img.save(image_file_name, format='PNG')

    if iter != 0 and iter % 100 == 0:
        time_elapsed = time.clock() - tic
        time_remaining_string = "Time remaining: " + "{0:.2f}".format(time_elapsed * (NUM_IMAGES - iter)/iter)
        print ("Created " + str(iter) + " of " + str(NUM_IMAGES) + ". Its been " + "{0:.2f}".format(time.clock()) + " seconds. " + time_remaining_string)
        

initial_parameters_for_csv = [["{0:.2f}".format(el) for el in row] for row in initial_parameters_for_csv]
df = pd.DataFrame(initial_parameters_for_csv)
df.to_csv(folder_name +"/initial_parameters.csv", header=None, index=None)