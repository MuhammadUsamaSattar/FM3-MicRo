import csv
import os
import matplotlib.pyplot as plt
import numpy as np

from gymnasium_env.envs.System import initializations

# File location. Change this to plot other .csv files.
# There are two ways to specify paths:
# 1) Relative - Input directories or filepath relative to the location of this file e.g.,
# Data/Simulator_Data/1_Log_Data.csv tries to find the file in a folder "Simulator_Data"
# which is placed inside a folder "Data" which is placed in the same folder as path_plotter.py file
# 2) Global - Specify the entire path of the file which allows you to plot files that are not in
# directories that are child of directory of this file e.g., C:/MyFiles/PlotFiles/1_Log_Data.csv
#
# You can input both solenoid calibration and log data files.
filename = "Data/Simulator_Data_2024-08-26_13.18.39/1_Log_Data.csv"

x = []
y = []
goal_locs = []

# Opens the file and reads data
with open(os.path.dirname(__file__) + "/" + filename, mode="r") as file:
    csvFile = csv.reader(file)

    i = 0
    for line in csvFile:
        # The first line only contains the line headers so calculates the index of 
        # the column at which goal location is present.
        # This allows user to input both solenoid calibration and log data files.
        if i == 0:
            i += 1

            for header in line:
                if header == "Goal Location" or header == "Coil_Location":
                    goal_header = line.index(header)

        # Adds data for all other lines
        else:
            # Convert particle location in str format to list
            particle_loc = line[1][1:-1].split(", ")
            particle_loc = list(map(int, particle_loc))

            # Convert goal location in str format to list
            goal_loc = line[goal_header][1:-1].split(", ")
            goal_loc = list(map(int, goal_loc))

            # Add x and y locations of particle after converting them to mm scale
            x.append(particle_loc[0] * 8 / (initializations.GUI_FRAME_WIDTH / 2))
            y.append(particle_loc[1] * 8 / (initializations.GUI_FRAME_HEIGHT / 2))

            # Add goal location to the list if its not already in the list
            if goal_loc not in goal_locs:
                goal_locs.append(goal_loc)

# Plots the manipulation area
theta = np.linspace(0, 2 * np.pi, 100)

# Calculate x and y coordinates based on the radius
circle_x = (
    initializations.SIM_SOL_CIRCLE_RAD
    * np.cos(theta)
    * 8
    / (initializations.GUI_FRAME_WIDTH / 2)
)
circle_y = (
    initializations.SIM_SOL_CIRCLE_RAD
    * np.sin(theta)
    * 8
    / (initializations.GUI_FRAME_HEIGHT / 2)
)
plt.plot(circle_x, circle_y, color="g", label="Manipulation Area")

# Calculates the locations of coil in mm and plots them
coil_locs = [
    [
        initializations.SIM_SOL_CIRCLE_RAD
        * np.cos((np.pi / 2) - (x * (2 * np.pi / 8)))
        * 8
        / (initializations.GUI_FRAME_WIDTH / 2),
        initializations.SIM_SOL_CIRCLE_RAD
        * np.sin((np.pi / 2) - (x * (2 * np.pi / 8)))
        * 8
        / (initializations.GUI_FRAME_HEIGHT / 2),
    ]
    for x in range(8)
]  # Calculates the location for each solenoid in mm using the angles. 0th solenoid is at pi/2.

for loc in coil_locs:
    plt.scatter(loc[0], loc[1], s=100, marker="o", color="k")
plt.scatter([], [], s=100, marker="o", color="k", label="Solenoids")

# Plots the path of the particle
plt.plot(x, y, color="b", label="Path")

# Plots the goal locations using "x" markers
plt.scatter(
    [point[0] * 8 / (initializations.GUI_FRAME_WIDTH / 2) for point in goal_locs],
    [point[1] * 8 / (initializations.GUI_FRAME_HEIGHT / 2) for point in goal_locs],
    s=250,
    marker="x",
    color="r",
    label="Goals",
)

plt.gca().set_aspect("equal", "box")  # Makes the aspect ratio of the plot 1:1

# Sets x and y limits of the plot by converting the frame size of the simulator in mm
plt.xlim(
    -8 * initializations.SIM_FRAME_WIDTH / initializations.GUI_FRAME_WIDTH,
    8 * initializations.SIM_FRAME_WIDTH / initializations.GUI_FRAME_WIDTH,
)
plt.ylim(
    -8 * initializations.SIM_FRAME_HEIGHT / initializations.GUI_FRAME_HEIGHT,
    8 * initializations.SIM_FRAME_HEIGHT / initializations.GUI_FRAME_HEIGHT,
)

# Sets the x and y axis labels
plt.xlabel("X location (mm)")
plt.ylabel("Y location (mm)")

# Shows legend
plt.legend()

# Shows plot
plt.show()
