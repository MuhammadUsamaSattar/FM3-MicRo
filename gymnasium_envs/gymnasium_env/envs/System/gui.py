"""
Coils are numbered 0 through 7 in clockwise direction, starting with the northern-most coil. The terms "Coil" and "Solenoid" are used interchangeably

               coil 0 (N)

    coil 7 (NW)            coil 1 (NE) 


coil 6 (W)                      coil 2 (E)


    coil 5 (SW)            coil 3 (SE) 

               coil 4 (S)

               
There are two Arduino Mega microcontrollers that are responsible for controlling the solenoids. 
    One microcontroller, refered to as Cart or Cartesian, controls the North, South, East, and West solenoids
    The other microcontroller, refered to as Diag or Diagonal, controls the NE, SE, SW, and NW solenoids
               
"""

import datetime
import math
import os
import sys
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.System.Library import blob_detection as Blob_detect
from gymnasium_env.envs.System.Library import data_extractor
from gymnasium_env.envs.System.Library import functions
from gymnasium_env.envs.System.Library import pyspin_wrapper as PySpin_lib
from gymnasium_env.envs.System.Control_Algorithms import control_algorithm
from gymnasium_env.envs.System.Interface.Ui_interface import *


class mywindow(QMainWindow, Ui_MainWindow):
    """QT window class based upon designed interface in QT Designer.
    Control and Camera function are also built within this class.
    """

    def __init__(self, init_with_camera_on = False):
        """Initializes the class attributes such as particle location, goal locations and various flags."""
        super(mywindow, self).__init__()
        self.setupUi(self)

        # Flags are used to record the state of the buttons on the user interface.
        # True means the button is toggled on. False means toggled off.
        self.flag_video_on = False  # Flag to determine if the video display is on
        self.flag_detecting_objects = (
            False  # Flag to determine if object detection is on
        )
        self.flag_record = False  # Flag to determine if the video recording is on
        self.flag_grid_on = False  # Flag to determine if the grid display is on
        self.flag_show_target_position = (
            False  # Flag to determine if the goal display is on
        )
        self.flag_show_goal_vector = False  # Flag to determine if display of the vector from particle to goal is on
        self.flag_automatic_control = (
            False  # Flag to determine if automatic control is on
        )
        self.flag_show_solenoid_state = (
            True  # Flag to determine if solenoid location and state display is on
        )
        self.flag_snapshot = False  # Flag to determine if a snapshot has to be taken
        self.flag_gen_sol_data = (
            False  # Flag to determine if solenoid calibration is on
        )
        self.flag_gen_log_data = False  # Flag to determine if data logging is on
        self.flag_setting_multiple_goals = (
            False  # Flag to determine if multiple goals are being set
        )

        # Detected location of the particle. Starts at top-right of the screen. This is a list to allow multiple particles.
        self.particle_locs = [
            [
                initializations.GUI_FRAME_WIDTH / 2,
                initializations.GUI_FRAME_HEIGHT / 2,
            ]
        ]
        self.particle_rads = [0]  # Detected radius of the particle. Starts at 0 value
        self.blob_lost = False  # Determines if the blob has been lost

        # List of size 8 containing scaled current values. Oth element is the Northern coil
        self.coil_vals = [0.0 for _ in initializations.COIL_NAMES]

        self.coil_locs = [
            [
                int(
                    initializations.GUI_SOL_CIRCLE_RAD
                    * math.cos(
                        (math.pi / 2) - (x * (2 * math.pi / len(self.coil_vals)))
                    )
                ),
                int(
                    initializations.GUI_SOL_CIRCLE_RAD
                    * math.sin(
                        (math.pi / 2) - (x * (2 * math.pi / len(self.coil_vals)))
                    )
                ),
            ]
            for x in range(len(self.coil_vals))
        ]  # List of size 8 containing coil locations. Oth element is the Northern coil

        self.grid_dim = 10  # Number of grid lines on each side of the center line

        self.goal_locs = [list(map(int, [0, 0]))]  # List containing locations of goals

        # Create a VideoCapture object
        self.cap = cv2.VideoCapture(0)

        # Gets names of the paths for video and snapshot functions
        (
            self.path,
            self.video_name_root,
            self.video_num,
            self.snapshot_name_root,
            self.snapshot_num,
        ) = self.getFileNames()

        # Data extractor attribute which allows logging and saving of data for "data logging" and "solenoid calibration mode"
        self.data_extractor = data_extractor.data_extractor(
            self.coil_locs,
            initializations.COIL_NAMES,
        )

        if init_with_camera_on:
            self.toggleVideoDisplay()

    def getFileNames(self):
        """Generates video and files names and directories.

        Returns:
            str: Directory in which to save videos, snapshots and log files
            str: Name without number plus directory for video
            int: Number to append at end of video name
            str: Name without number plus directory for snapshot
            int: Number to append at end of snapshot name
        """
        # The date and time that the program was started at is used to create folder names to store videos and snapshots
        date_time = str(
            datetime.datetime.now()
        )  # Initial format is --> YYYY-MM-DD HH:MM:SS.SSSSSS
        date_time = date_time.replace(
            ":", "."
        )  # Replace the colons --> YYYY-MM-DD HH.MM.SS.SSSSSS
        date_time = date_time.replace(
            " ", "_"
        )  # Replace the space --> YYYY-MM-DD_HH.MM.SS.SSSSSS
        date_time = date_time[
            :-7
        ]  # Remove the fractions of the second --> YYYY-MM-DD_HH.MM.SS

        # Creates the folder name
        path = "Data/Experiment_Data_" + date_time

        video_num = 1  # Number to append at end of video name
        video_name_root = (
            path + "/Recording_"
        )  # Name without number plus directory for video
        video_name = (
            video_name_root + str(video_num) + ".avi"
        )  # Name with number plus directory for video

        snapshot_num = 1  # Number to append at end of snapshot name
        snapshot_name_root = (
            path + "/Snapshot_"
        )  # Name without number plus directory for snapshot

        return path, video_name_root, video_num, snapshot_name_root, snapshot_num

    def toggleVideoDisplay(self):
        """Toggles video display"""
        # If video was on, turns it off and stops the PySpin cam so
        # that on turning it on again, a new cam object can be created
        # without crashing
        if self.flag_video_on:
            self.flag_video_on = False

            print("Video turned off")
            self.pushButton.setText("Video: OFF")
            PySpin_lib.Cam_PySpin_Stop(self.cam)
            self.label_7.clear()

        # If video was off, turns it on.
        else:
            self.flag_video_on = True

            print("Video turned on")
            self.pushButton.setText("Video: ON")

            _, _, self.cam_list, self.system1 = (
                PySpin_lib.Cam_PySpin_Init()
            )  # Intializes the camera object and gets the list of all cameras (one in our case)

            self.cam = self.cam_list[0]  # Retrieves the first camera in the list
            print("Camera: " + str(self.cam))  # Prints the camera id
            PySpin_lib.Cam_PySpin_Connect(self.cam)  # Connect to the first camera

            self.run()

    def toggleGridDisplay(self):
        """Toggles grid display"""
        # If grid display was on, turns it off
        if self.flag_grid_on:
            self.flag_grid_on = False

            print("Grid turned off")
            self.pushButton_5.setText("Grid: OFF")

        # If grid display was off, turns it on
        else:
            self.flag_grid_on = True

            print("Grid turned on")
            self.pushButton_5.setText("Grid: ON")

    def toggleVideoRecording(self):
        """Toggles video recording"""
        # If recording was on, turns it off and saves the recorded frames as an .avi file.
        # Then increments the video number by 1 so that the next video has a unique name.
        if self.flag_record:
            self.flag_record = False

            print("Recording ended")
            self.pushButton_3.setText("Recording: OFF")

            self.cap.release()
            self.out.release()
            print(
                "Recording created: ",
                os.getcwd() + "/" + self.video_name_root + str(self.video_num) + ".avi",
            )
            self.video_num = self.video_num + 1

        # If recording was off, turns it on and creates the directory if it does not already exist.
        # Also creates the cv2.VidoeWriter object which writes the frames to a bufffer so they can be saved later.
        else:
            self.flag_record = True

            os.makedirs(
                self.path, exist_ok=True
            )  # Creates a unique directory for this run of the program if it doesn't already exist
            video_name = self.video_name_root + str(self.video_num) + ".avi"

            # Creates the video writer object
            self.out = cv2.VideoWriter(
                video_name,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                initializations.GUI_FRAME_RATE_VIDEO_RECORDING,
                (initializations.GUI_FRAME_WIDTH, initializations.GUI_FRAME_HEIGHT),
            )

            print("Recording started")
            self.pushButton_3.setText("Recording: ON")

    def takeSnapshot(self):
        """Turns flag for taking snapshot on"""
        # If flag for taking snapshot was off, turns it on
        if self.flag_snapshot == False:
            self.flag_snapshot = True

            print("Taking snapshot")

    def toggleObjectDetection(self):
        """Toggles object detection"""
        # If object detection was on, turns it off
        if self.flag_detecting_objects:
            self.flag_detecting_objects = False

            print("Object detection turned off")
            self.pushButton_2.setText("Object Detection: OFF")

        # If object detection was off, turns it on
        else:
            self.flag_detecting_objects = True

            print("Object detection turned on")
            self.pushButton_2.setText("Object Detection: ON")

            # Background subtractor object which extracts the moving foregorund objects
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False
            )

    def toggleSolenoidDisplay(self):
        """Toggles solenoid location and state display"""
        # If solenoid display was on, turns it off
        if self.flag_show_solenoid_state:
            self.flag_show_solenoid_state = False

            print("Show solenoid state turned off")
            self.pushButton_4.setText("Show Solenoid State: OFF")

        # If solenoid display was off, turns it on
        else:
            self.flag_show_solenoid_state = True

            print("Show solenoid state turned on")
            self.pushButton_4.setText("Show Solenoid State: ON")

    def toggleGoalsDisplay(self):
        """Toggles goal(s) loction display"""
        # If goal display was on, turns it off
        if self.flag_show_target_position:
            self.flag_show_target_position = False

            print("Show Target Position turned off")
            self.pushButton_7.setText("Show Goal: OFF")

        # If goal display was off, turns it on
        else:
            self.flag_show_target_position = True

            print("Show Target Position turned on")
            self.pushButton_7.setText("Show Goal: ON")

    def toggleGoalVectorDisplay(self):
        """Toggles display of vector from particle to current goal"""
        # If goal vector was on, turns it off
        if self.flag_show_goal_vector:
            self.flag_show_goal_vector = False

            print("Show Goal Vector turned off")
            self.pushButton_8.setText("Show Goal Vector: OFF")

        # If goal vector was off, turns it on
        else:
            self.flag_show_goal_vector = True

            print("Show Goal Vector turned on")
            self.pushButton_8.setText("Show Goal Vector: ON")

    def sendCartString(self, coil_vals):
        """Converts the scaled coil values to formatted string and sends them to the Cartesian solenoids

        Args:
            coil_vals : List of size 8 containing scaled coil currents
        """
        vals = []
        for val in coil_vals:
            vals.append(
                round(
                    functions.current_frac_to_formatted(
                        val,
                        initializations.GUI_ZERO_VAL,
                        initializations.GUI_OUTPUT_RANGE,
                    ),
                    2,
                )
            )

        # Converts the scaled coil values to the required string format
        send_str_temp = "".join(
            [
                "A",
                str(int(vals[0])),
                ",",
                str(int(vals[2])),
                ",",
                str(int(vals[4])),
                ",",
                str(int(vals[6])),
                ",",
                "B",
            ]
        )

        initializations.GUI_SER_CART.write(
            send_str_temp.encode("utf-8")
        )  # Sends the formatted current values to the cartesian solenoid

    def sendDiagString(self, coil_vals):
        """Converts the scaled coil values to formatted string and sends them to the Diagonal solenoids

        Args:
            coil_vals : List of size 8 containing scaled coil currents
        """
        vals = []
        for val in coil_vals:
            vals.append(
                round(
                    functions.current_frac_to_formatted(
                        val,
                        initializations.GUI_ZERO_VAL,
                        initializations.GUI_OUTPUT_RANGE,
                    ),
                    2,
                )
            )

        # Converts the scaled coil values to the required string format
        send_str_temp = "".join(
            [
                "A",
                str(int(vals[1])),
                ",",
                str(int(vals[3])),
                ",",
                str(int(vals[5])),
                ",",
                str(int(vals[7])),
                ",",
                "B",
            ]
        )
        initializations.GUI_SER_DIAG.write(
            send_str_temp.encode("utf-8")
        )  # Sends the formatted current values to the diagonal solenoid

    def manual(self, value, coil_index, coil_vals):
        """Sends inputted percentage coil currents to the coil with inputted coil index

        Args:
            value : Current value in percentage
            coil_index : Index of coil to send current to
            coil_vals : List of size 8 containing scaled coil currents
        """
        coil_vals[coil_index] = value / initializations.GUI_INPUT_RANGE

        # Determines which set of coils (cartesian or diagonal) this coil being changed belongs to.
        # 0 = Cartesian and 1 = Diagonal.
        if coil_index % 2 == 0:
            self.sendCartString(coil_vals)

        else:
            self.sendDiagString(coil_vals)

        time.sleep(initializations.GUI_SLEEP_INT)

    def manualNorth(self):
        """Receives input from spinbox of North coil and sends current values to the same coil"""
        self.manual(self.spinBox_N.value(), 0, self.coil_vals)

    def manualNorthEast(self):
        """Receives input from spinbox of NorthEast coil and sends current values to the same coil"""
        self.manual(self.spinBox_NE.value(), 1, self.coil_vals)

    def manualEast(self):
        """Receives input from spinbox of East coil and sends current values to the same coil"""
        self.manual(self.spinBox_E.value(), 2, self.coil_vals)

    def manualSouthEast(self):
        """Receives input from spinbox of SouthEast coil and sends current values to the same coil"""
        self.manual(self.spinBox_SE.value(), 3, self.coil_vals)

    def manualSouth(self):
        """Receives input from spinbox of South coil and sends current values to the same coil"""
        self.manual(self.spinBox_S.value(), 4, self.coil_vals)

    def manualSouthWest(self):
        """Receives input from spinbox of SouthWest coil and sends current values to the same coil"""
        self.manual(self.spinBox_SW.value(), 5, self.coil_vals)

    def manualWest(self):
        """Receives input from spinbox of West coil and sends current values to the same coil"""
        self.manual(self.spinBox_W.value(), 6, self.coil_vals)

    def manualNorthWest(self):
        """Receives input from spinbox of NorthWest coil and sends current values to the same coil"""
        self.manual(self.spinBox_NW.value(), 7, self.coil_vals)

    def resetVals(self):
        """Resets all coil and spinbox values to zero"""
        self.spinBox_N.setValue(0)
        self.spinBox_NE.setValue(0)
        self.spinBox_E.setValue(0)
        self.spinBox_SE.setValue(0)
        self.spinBox_S.setValue(0)
        self.spinBox_SW.setValue(0)
        self.spinBox_W.setValue(0)
        self.spinBox_NW.setValue(0)

        for i in range(len(self.coil_vals)):
            self.manual(0, i, self.coil_vals)

    def automaticControl(self):
        """Toggles automatic control"""
        # If automatic control was on, turns it off
        if self.flag_automatic_control:
            self.flag_automatic_control = False

            self.resetVals()

            # Send the strings twice, incase there was an error
            for _ in range(2):
                self.sendCartString(self.coil_vals)
                self.sendDiagString(self.coil_vals)
                time.sleep(initializations.GUI_SLEEP_INT)

            print("Automatic control turned off")
            self.pushButton_9.setText("Automatic Control: OFF")

        # If automatic control was off, turns it on
        else:
            self.flag_automatic_control = True

            print("Automatic control turned on")
            self.pushButton_9.setText("Automatic Control: ON")

            if not self.flag_detecting_objects:
                self.toggleObjectDetection()

    def toggleSettingMultipleGoals(self):
        """Toggles multiple goals mode"""
        # If multiple gaols mode was on, turns it off
        if self.flag_setting_multiple_goals:
            if len(self.goal_locs) > 1:
                print("Executing Multiple Goals Motion")
                self.pushButton_14.setText("Cancel Motion")

                self.goal_locs.pop(
                    0
                )  # Removes the first goal from the list since it was the goal prior to multiple goals

                # Duplicates the last goal because "Executing Multiple Goals Motion" is
                # only displayed as long length of self.goal_locs is greater than 1
                self.goal_locs.append(self.goal_locs[-1])

            else:
                print("Multiple Goals Mode: OFF")
                self.pushButton_14.setText("Multiple Goals Mode: OFF")

            self.flag_setting_multiple_goals = not self.flag_setting_multiple_goals

        # If multiple gaols mode was off, turns it on
        else:
            # Checks if length of self.goal_locs is equal to 1, then just turns on multiple goals mode
            if len(self.goal_locs) == 1:
                print("Multiple Goals Mode: ON")
                self.pushButton_14.setText("Multiple Goals Mode: ON")
                self.pushButton_15.setText("Add Goal")

                if not self.flag_show_target_position:
                    self.toggleGoalsDisplay()

                self.flag_setting_multiple_goals = not self.flag_setting_multiple_goals

            # If length of self.goal_locs is greater than 1, then that means that multiple goals were being
            # executed which are now being cancelled. So self.goal_locs need to be reset as just having the
            # current goal
            else:
                print("Multiple Goals Motion Cancelled")
                self.pushButton_14.setText("Multiple Goals Mode: OFF")
                self.pushButton_15.setText("Set Goal")

                self.goal_locs = [self.goal_locs[0]]

    def addGoal(self):
        """Add/Replace the entered goal to self.goal_locs list"""
        val = list(
            map(int, [self.doubleSpinBox_X.value(), self.doubleSpinBox_Y.value()])
        )

        # Check if entered value is out of bounds from the circle fromed by the solenoids
        if (
            abs(val[0]) > initializations.GUI_SOL_CIRCLE_RAD
            or abs(val[1]) > initializations.GUI_SOL_CIRCLE_RAD
        ):
            if abs(val[0]) > initializations.GUI_SOL_CIRCLE_RAD:
                print(
                    "X value out of bound! Value must be within "
                    + str(-initializations.GUI_SOL_CIRCLE_RAD)
                    + " and "
                    + str(initializations.GUI_SOL_CIRCLE_RAD)
                    + "."
                )

            if abs(val[1]) > initializations.GUI_SOL_CIRCLE_RAD:
                print(
                    "Y value out of bound! Value must be within "
                    + str(-initializations.GUI_SOL_CIRCLE_RAD)
                    + " and "
                    + str(initializations.GUI_SOL_CIRCLE_RAD)
                    + "."
                )

        # If values are in bound either add it to the list or replace
        # the existing goal depending upon if multiple goals mode is on
        else:
            if self.flag_setting_multiple_goals:
                print("Goal added:", val)
                self.goal_locs.append(val)

            else:
                print("Goal changed to:", val)
                self.goal_locs = [val]

    def loadPathFromFile(self):
        try:
            # Opens the Path Data.txt file which is placed in the same directory as this file
            with open(os.path.dirname(__file__) + "/Path Data.txt", "r") as f:
                path_data = f.readlines()
                temp = []

                for point in path_data:
                    temp.append(point.replace(" ", "").split(","))
                    temp[-1] = list(map(int, temp[-1]))

                    for val in temp[-1]:
                        if abs(val) > initializations.SIM_SOL_CIRCLE_RAD:
                            raise ValueError

                self.goal_locs = temp
                self.pushButton_14.setText("Cancel Motion")

        except FileNotFoundError:
            print(
                'File does not exist. File must be named "Path Data.txt" and placed in the same directory as this file.'
            )

        except ValueError:
            print(
                "One or more values are out of bound. Ensure that values are within +",
                initializations.GUI_SOL_CIRCLE_RAD,
                "and -",
                initializations.GUI_SOL_CIRCLE_RAD,
            )

        except:
            print(
                "Invalid format of path locations. The file must consists of each path point on separate line."
                'Each point must consist of two numbers separated by ",".'
            )

    def toggleGenLogData(self):
        """Toggles data logging mode"""
        self.flag_gen_sol_data = False

        # If data logging was on, turns it off and save the data in a .csv file
        if self.flag_gen_log_data:
            self.pushButton_11.setText("Log Data: OFF")
            self.data_extractor.save_data(self.path)

        # If data logging was off, turns it on
        else:
            # If solenoid calibration was active, deacitvate it and save the data in a .csv file
            if self.flag_gen_sol_data:
                self.data_extractor.save_data(self.path)

                self.flag_gen_sol_data = False

            self.data_extractor.startDataSet(mode="log_data")
            self.data_extractor.startDataSeries()

            self.pushButton_11.setText("Log Data: ON")

        self.flag_gen_log_data = not self.flag_gen_log_data

    def toggleGenSolData(self):
        """Toggles solenoid data calibration mode"""
        self.flag_automatic_control = False

        # If solenoid calbiration mode was active, deactivates it and saves the recorded data to a .csv file
        if self.flag_gen_log_data:
            self.pushButton_12.setText("Generate Solenoid Data")
            self.data_extractor.save_data(self.path)
            self.flag_gen_log_data = False

        # If solenoid calbiration mode was inactive, starts calibration scheme which varies the current
        # on a single solenoid with different sessions each starting at a different location.
        if not self.flag_gen_sol_data:
            self.flag_gen_sol_data = True

            print("Data recording routine for solenoids started")
            self.pushButton_12.setText("Generating Solenoid Data...")

            # Number of current values to calibrate on. Values are equally spaced and start from the first non-zero value.
            current_vals = 20

            # Number of sessions. Each session consists of going through each current value once.
            # Starting location of each session is also different if the two values in r_bounds are different
            sessions = 12

            # Starting radial distance of the first and last session from the solenoid.
            # Intermediate sessions interpolate between these two bounds.
            # Values represent fraction of the radius of the circle traced out by solenoids.
            # For example, with a GUI_SOL_CIRCLE_RAD = 320, an r_bounds = [-0.6, 0.4] gives
            # first and last starting radial distance of -192 and 128.
            # Notice that you have to provide the signs yourself and values are around a center point
            # i.e., between -1.0 and 1.0
            r_bounds = [-0.6, -0.6]

            # Index of solenoid to calibrate for. If you pass a list containing [0, ..., 7] all
            # solenoids are calibrated with "sessions" number of sessions each
            # Note that solenoids have non insignificant variation and therefore the response is
            # very spread out which is why I only calibrate using one solenoid
            coils_to_calibrate = [4]

            self.toggleObjectDetection = True

            self.resetVals()
            self.data_extractor.startDataSet(mode="solenoid_calibration")
            self.demagnetizeAllSolenoids()

            # Iterates over all coils to calibrate (which is just the coil with index 4 in our case)
            for i in coils_to_calibrate:

                # Repeat the calibration on all current vals "sessions" number of time
                for j in range(sessions):
                    print("Coil#", i, ", Session#", j)
                    factor = r_bounds[0] + (
                        (j / (sessions - 1)) * (r_bounds[1] - r_bounds[0])
                    )  # Interpolates between bounds to get the factor to get starting location for this session

                    self.goal_locs = [
                        [
                            int(factor * self.coil_locs[i][0]),
                            int(factor * self.coil_locs[i][1]),
                        ]
                    ]  # Sets the starting location of this session as goal so particle can move there

                    # Records data for each current value once
                    for k in range(1, current_vals + 1):
                        self.flag_automatic_control = True  # Turns on automatic control so particle can move to the starting location

                        # Automatic control is kept on for 5 seconds so particle position can stablize
                        data_time = time.time()
                        while (time.time() - data_time) < 5:
                            self.step()

                            # If program is closed, ensures that all solenoids are turned off
                            if not self.isVisible():
                                self.close()
                        self.flag_automatic_control = False
                        self.resetVals()

                        lost_for_frames = 0  # Variable to detect how many frames the particle position remains unchanged for

                        # Starts a new data series which will be saved in a different file from the previous calibration, if any
                        self.data_extractor.startDataSeries()

                        data_time = time.time()
                        self.manual(
                            k * (100 / current_vals), i, self.coil_vals
                        )  # Sends the current value to the solenoid

                        print("Current Value#", k, ": ", self.coil_vals)

                        # Keep logging data until either particle location remains static for 100 frame or 15 seconds
                        # have passed whichever comes first.
                        # Frame limit is set so that trial can be ended quicker since static positions are useless data
                        # Time limit is set so that data from lower current values does not dominate the dataset
                        while lost_for_frames < 100 and (time.time() - data_time) < 15:
                            self.step()

                            self.data_extractor.record_datapoint(
                                self.particle_locs, self.coil_vals, coil_index=i
                            )  # Record the data in the current frame

                            if self.blob_lost:
                                lost_for_frames += 1
                            else:
                                lost_for_frames = 0

                            # If program is closed, ensures that all solenoids are turned off
                            if not self.isVisible():
                                self.close()

                    self.demagnetizeSolenoid(
                        i
                    )  # Demagnetize solenoid afterwards to remove residual magnetism

            self.goal_locs = [list(map(int, [0, 0]))]
            self.data_extractor.save_data(self.path)
            self.resetVals()

    def demagnetizeSolenoid(self, index):
        """Demagnetizes the coil

        Args:
            index : Index of coil to demagnetize
        """
        steps = 8  # Number of steps of decreasing current to take for demagnetization

        for j in range(steps):
            val = (steps - j) * (80 / steps)

            # At each step value, alternates twice between +step and -step
            for _ in range(2):
                for mult in [1, -1]:
                    self.manual(mult * val, index, self.coil_vals)

                    time_stamp = time.time()

                    # Maintain current for 0.1 to allow magnetic field to fully develop
                    while time.time() - time_stamp < 0.1:
                        self.step()

            # If program is closed, ensures that all solenoids are turned off
            if not self.isVisible():
                self.close()

        self.manual(0, index, self.coil_vals)

    def drawSolenoidState(self, frame, coil_value, x, y):
        """Draws the solenoid circles along with their state

        Args:
            frame : Frame object on which to draw on
            coil_value : Scaled coil current
            x : x location of solenoid
            y : y location of solenoid
        """
        color = [0, 0, 0]
        for i in range(len(initializations.GUI_SOL_COLOR)):
            if initializations.GUI_SOL_COLOR[i] != 0:

                # Calculates solenoid color by creating SIM_COLOR_DIVS number
                # of divisions of color component and determining the smallest
                # division higher than the color value determined from
                # scaling the color component according to the coil_value.
                color[i] = (
                    (coil_value * initializations.GUI_SOL_COLOR[i])
                    // (
                        initializations.GUI_SOL_COLOR[i]
                        / initializations.GUI_COLOR_DIVS
                    )
                ) * (initializations.GUI_SOL_COLOR[i] / initializations.GUI_COLOR_DIVS)

        cv2.circle(frame, (x, y), initializations.GUI_SOL_RAD, color, -1)

    def demagnetizeAllSolenoids(self):
        """Demagnetizes all solenoids"""
        # Turns off automatic control before demagnetization
        if self.flag_automatic_control:
            self.automaticControl()
        print("Demagnetizing Solenoids")
        self.pushButton_6.setText("Demagnetizing Solenoids...")

        # Iterates through each coils and demagnetizes it
        for i in range(len(self.coil_vals)):
            self.demagnetizeSolenoid(i)

        print("Demagnetizing Complete")
        self.pushButton_6.setText("Demagnetize Solenoids")

    def drawCameraFrame(self, frame):
        """Draws the camera frame

        Args:
            frame : Frame on which to draw
        """
        frame = QImage(
            frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888
        )  # Converts the captured camera image to RGB888 format

        self.label_7.setPixmap(
            QPixmap.fromImage(frame)
        )  # Applies the frame to the label representing the camera feed display on QT window

    def drawGrids(self, frame, frame_size):
        """Draws the grids

        Args:
            frame : Frame object on which to draw
            frame_size : Size of camera frame
        """
        if self.flag_grid_on:
            # Draws the main axes
            cv2.line(
                frame,
                (0, int(initializations.GUI_FRAME_WIDTH / 2)),
                (frame_size, int(initializations.GUI_FRAME_WIDTH / 2)),
                initializations.GUI_GRID_AXIS_COLOR,
                initializations.GUI_GRID_AXIS_THK,
            )  # Central horizontal line
            cv2.line(
                frame,
                (int(initializations.GUI_FRAME_WIDTH / 2), 0),
                (int(initializations.GUI_FRAME_WIDTH / 2), frame_size),
                initializations.GUI_GRID_AXIS_COLOR,
                initializations.GUI_GRID_AXIS_THK,
            )  # Central vertical line

            # # Label the axes
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale = 0.5
            # cv2.putText(frame,"X", (5,int(initializations.FRAME_WIDTH/2)-5), font,fontScale,initializations.GRID_AXIS_COLOR,2)
            # cv2.putText(frame,"Y", (int(initializations.FRAME_WIDTH/2)+5,d-5), font,fontScale,initializations.GRID_AXIS_COLOR,2)

            # Draws the rest of the grid
            for i in range(1, self.grid_dim):
                offset = int(
                    i * (int(initializations.GUI_FRAME_WIDTH / 2) / self.grid_dim)
                )

                cv2.line(
                    frame,
                    (0, int(initializations.GUI_FRAME_WIDTH / 2) + offset),
                    (frame_size, int(initializations.GUI_FRAME_WIDTH / 2) + offset),
                    initializations.GUI_GRID_COLOR,
                    initializations.GUI_GRID_THK,
                )  # Horizontal line in postive y-axis quadrants
                cv2.line(
                    frame,
                    (0, int(initializations.GUI_FRAME_WIDTH / 2) - offset),
                    (frame_size, int(initializations.GUI_FRAME_WIDTH / 2) - offset),
                    initializations.GUI_GRID_COLOR,
                    initializations.GUI_GRID_THK,
                )  # Horizontal line in negative y-axis quadrants
                cv2.line(
                    frame,
                    (int(initializations.GUI_FRAME_WIDTH / 2) + offset, 0),
                    (int(initializations.GUI_FRAME_WIDTH / 2) + offset, frame_size),
                    initializations.GUI_GRID_COLOR,
                    initializations.GUI_GRID_THK,
                )  # Vertical line in postive x-axis quadrants
                cv2.line(
                    frame,
                    (int(initializations.GUI_FRAME_WIDTH / 2) - offset, 0),
                    (int(initializations.GUI_FRAME_WIDTH / 2) - offset, frame_size),
                    initializations.GUI_GRID_COLOR,
                    initializations.GUI_GRID_THK,
                )  # Vertical line in negative x-axis quadrants

    def getBlobData(self, frame):
        """Gets blob location and size

        Args:
            frame : Frame object from which to extract particle data

        Returns:
            numpy array: The frame with all required processings applied to detect the object
        """
        # Subtracts the background to obtain foreground object.
        #
        # Learning rate of 0.0001 allows the object to forget foreground object by reinitialziing
        # the background after every 10000 frames which is useful if there are moving bubbles
        # etc. But it also introduces the issue that if the object remains at one location
        # for too long, the detection algorithm starts losing track of it as indicated by
        # blob size getting smaller and location getting erratic.
        #
        # Or you can set learningRate to 0. At this value, the object doesnt get lost but
        # occasionally, some large objects might be tracked instead and those won't be frogotten
        # cv2.imshow("Initial Frame", frame)
        ret, detection_frame = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Thresholded Frame", detection_frame)

        detection_frame = self.background_subtractor.apply(
            detection_frame, learningRate=0.0000
        )
        # cv2.imshow("Subtracted Frame", detection_frame)

        # Blurs the frame to convert the sharp shape of the blob to a more circular one.
        # Other blur types were tried such as Gaussian blur, but those did not give as good
        # performance as Median Blur.
        detection_frame = cv2.medianBlur(detection_frame, 5)
        # cv2.imshow("Blurred Frame", detection_frame)

        # Inverts the color of the frame since background subtractor sets foreground object as white
        # and background as black. OpenCV blob detector only detects blobs of color black.
        detection_frame = cv2.bitwise_not(detection_frame)
        # cv2.imshow("Inverted Frame", detection_frame)

        particle_locs_img = []
        for particle_loc in self.particle_locs:
            particle_locs_img.append(
                functions.absolute_to_image(
                    particle_loc[0],
                    particle_loc[1],
                    initializations.GUI_FRAME_WIDTH,
                    initializations.GUI_FRAME_HEIGHT,
                )
            )

        self.blob_lost, particle_locs_img, self.particle_rads = (
            Blob_detect.iden_blob_detect(
                detection_frame, particle_locs_img, self.particle_rads
            )
        )  # Detects blob using openCV blob detection object

        self.particle_locs = []
        for particle_loc_img in particle_locs_img:
            self.particle_locs.append(
                functions.image_to_absolute(
                    particle_loc_img[0],
                    particle_loc_img[1],
                    initializations.GUI_FRAME_WIDTH,
                    initializations.GUI_FRAME_HEIGHT,
                )
            )

        return detection_frame

    def drawBlobs(self, frame, particle_locs):
        """Draws the particle's blob

        Args:
            frame : Frame object on which to draw
            particle_locs : Particle location
        """
        # If particle has been found once, then draw the particle at the last know location
        if self.flag_detecting_objects:
            for i in range(len(particle_locs)):
                particle_loc_img = functions.absolute_to_image(
                    particle_locs[i][0],
                    particle_locs[i][1],
                    initializations.GUI_FRAME_WIDTH,
                    initializations.GUI_FRAME_HEIGHT,
                )

                cv2.circle(
                    frame,
                    particle_loc_img,
                    self.particle_rads[i],
                    initializations.GUI_BLOB_COLOR,
                    initializations.GUI_BLOB_THK,
                )

    def drawSolenoids(self, frame, coil_locs, coil_vals):
        """Draws the solenoid locations and states

        Args:
            frame : Frame object on which to draw
            coil_locs : List of size 8 containing locations of coils
            coil_vals : List of size 8 containing scaled coil currents
        """
        # If flag to show solenoid state is on, iterates over all coils and draws them
        if self.flag_show_solenoid_state:
            for i in range(len(self.coil_locs)):
                coil_loc_img = functions.absolute_to_image(
                    coil_locs[i][0],
                    coil_locs[i][1],
                    initializations.GUI_FRAME_WIDTH,
                    initializations.GUI_FRAME_HEIGHT,
                )

                self.drawSolenoidState(
                    frame, coil_vals[i], coil_loc_img[0], coil_loc_img[1]
                )

    def drawGoal(self, frame, goal_locs):
        """Draws the goal

        Args:
            frame : Frame object on which to draw
            goal_locs : Locations of goals
        """
        if self.flag_show_target_position:
            for i in range(len(goal_locs)):
                # Draw first goal only if user is not setting multiple goals right now.
                # When user is setting multiple goals, the first goal is the goal prior to
                # multiple goals mode, so it should not be drawn.
                if not self.flag_setting_multiple_goals or i != 0:
                    goal_loc_img = functions.absolute_to_image(
                        goal_locs[i][0],
                        goal_locs[i][1],
                        initializations.GUI_FRAME_WIDTH,
                        initializations.GUI_FRAME_HEIGHT,
                    )

                    cv2.line(
                        frame,
                        (
                            goal_loc_img[0] + initializations.GUI_GOAL_SIZE,
                            goal_loc_img[1] + initializations.GUI_GOAL_SIZE,
                        ),
                        (
                            goal_loc_img[0] - initializations.GUI_GOAL_SIZE,
                            goal_loc_img[1] - initializations.GUI_GOAL_SIZE,
                        ),
                        initializations.GUI_GOAL_COLOR,
                        initializations.GUI_GOAL_THICKNESS,
                    )
                    cv2.line(
                        frame,
                        (
                            goal_loc_img[0] + initializations.GUI_GOAL_SIZE,
                            goal_loc_img[1] - initializations.GUI_GOAL_SIZE,
                        ),
                        (
                            goal_loc_img[0] - initializations.GUI_GOAL_SIZE,
                            goal_loc_img[1] + initializations.GUI_GOAL_SIZE,
                        ),
                        initializations.GUI_GOAL_COLOR,
                        initializations.GUI_GOAL_THICKNESS,
                    )

    def drawGoalVector(self, frame, particle_loc, goal_locs):
        """Draws the vector from particle to goal

        Args:
            frame : Frame object on which to draw
            particle_loc : Location of particle
            goal_locs : Locations of goals
        """
        # If user is not setting multiple goals right now, and flag to show goal vector is on,
        # then draws the goal vector to the first goal in list
        if self.flag_show_goal_vector and not self.flag_setting_multiple_goals:
            particle_loc_img = functions.absolute_to_image(
                particle_loc[0],
                particle_loc[1],
                initializations.GUI_FRAME_WIDTH,
                initializations.GUI_FRAME_HEIGHT,
            )

            goal_loc_img = functions.absolute_to_image(
                goal_locs[0][0],
                goal_locs[0][1],
                initializations.GUI_FRAME_WIDTH,
                initializations.GUI_FRAME_HEIGHT,
            )

            cv2.line(
                frame,
                particle_loc_img,
                goal_loc_img,
                initializations.GUI_GOAL_VECTOR_COLOR,
                initializations.GUI_GOAL_VECTOR_THICKNESS,
            )

    def drawElements(
        self, frame, particle_locs, goal_locs, coil_locs, coil_vals, frame_size
    ):
        """Draws all screen elements

        Args:
            frame : Frame object on which to draw
            particle_locs : Location of particle
            goal_locs : Locations of goals
            coil_locs : List of size 8 containing locations of coils
            coil_vals : List of size 8 containing scaled coil currents
            frame_size : Size of the camera frame
        """
        self.drawGrids(frame, frame_size)
        self.drawGoalVector(frame, particle_locs[0], goal_locs)
        self.drawGoal(frame, goal_locs)
        self.drawBlobs(frame, particle_locs)
        self.drawSolenoids(frame, coil_locs, coil_vals)

    def resetAtRandomLocs(self, seed, particle_reset, goal_reset):
        """Resets the game by resetting all values and placing the partcile and goal at random locations.

        Args:
            seed : Seed value that determines the pseudo-random number.
            particle_reset : Resets particle location on start of episode.
            goal_reset : Resets goal location on start of episode.
        """

        if goal_reset:
            factor = 0.8 * 3 / 4
            # Attributes to hold the particle and goal locations
            self.goal_locs = [
                list(
                    np.random.randint(
                        -initializations.GUI_SOL_CIRCLE_RAD * factor,
                        initializations.GUI_SOL_CIRCLE_RAD * factor,
                        size=2,
                        dtype=int,
                    )
                )
            ]

        else:
            self.goal_locs = [[0, 0]]

        if particle_reset:
            self.particle_start_loc = self.goal_locs[0].copy()
            factor = 0.8

            r = np.random.randint(
                200,
                initializations.GUI_SOL_CIRCLE_RAD * factor,
                dtype=int,
            )

            self.particle_start_loc[0] = np.random.randint(
                -r,
                r,
                dtype=int,
            )

            self.particle_start_loc[1] = math.sqrt((r**2) - (self.particle_start_loc[0]**2))
            self.particle_start_loc[1] *= np.random.choice([-1, 1])

        temp = self.goal_locs[0].copy()
        self.goal_locs[0] = self.particle_start_loc.copy()
        self.flag_automatic_control = True  # Turns on automatic control so particle can move to the starting location

        # Automatic control is kept on for 5 seconds so particle position can stablize
        data_time = time.time()
        while (time.time() - data_time) < 5:
            self.step()

            # If program is closed, ensures that all solenoids are turned off
            if not self.isVisible():
                self.close()

        self.flag_automatic_control = False
        self.resetVals()
        self.goal_locs[0] = temp.copy()

        self.flag_setting_multiple_goals = (
            False  # Flag to detect if multiple goals mode has been turned on
        )
        self.flag_gen_log_data = False  # Flag to detect if logging mode has been turned on

        self.flag_record = False  # Flag to detect if canvas is being recorded

        self.data_extractor = data_extractor.data_extractor(
            self.coil_locs,
            initializations.COIL_NAMES,
        )  # Stores the data extractor object in an attribute

        # Gets names of the paths for video and snapshot functions
        (
            self.path,
            self.video_name_root,
            self.video_num,
            self.snapshot_name_root,
            self.snapshot_num,
        ) = self.getFileNames()

    def getState(self, render, n_obs=1):
        particle_locs = [self.particle_locs[0].copy()]

        while (n_obs - len(particle_locs)) > 1:
            particle_locs.append([0, 0])

        while len(particle_locs) < n_obs:
            self.step()

            particle_locs.append(self.particle_locs[0].copy())

        return (
            particle_locs.copy(),
            self.goal_locs[0].copy(),
            self.coil_vals.copy(),
            self.coil_locs.copy(),
        )

    def process_frame(self):
        """Displays video and associated screen elements"""
        time_stamp = (
            time.time()
        )  # Time stamp to measure how much time each block of code takes
        frame = PySpin_lib.Cam_PySpin_GetImg(
            self.cam
        )  # Gets the frame using PySpin. Note: Original video size is 1920 x 1200.

        w_crop = int(
            (len(frame[0]) - initializations.GUI_FRAME_WIDTH) / 2
        )  # Pixels to crop on each side in horizontal direction so that the frame is square
        h_crop = int(
            (len(frame) - initializations.GUI_FRAME_HEIGHT) / 2
        )  # Pixels to crop on each side in vertical direction so that the frame is square

        frame = frame[
            h_crop : len(frame) - h_crop, w_crop : len(frame[0]) - w_crop
        ]  # Crops the camera feed
        frame = cv2.rotate(
            frame, cv2.ROTATE_90_CLOCKWISE
        )  # Rotates the camera feed so that Northern coil is on top of screen

        # Prints the time taken to process the camera frame. Then updates the time stamp to current time.
        if initializations.VERBOSE:
            print(
                "##################################################\nTime taken to process camera frame:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        if self.flag_detecting_objects:
            detection_frame = self.getBlobData(frame)

            # Uncomment this line to open a separate window that shows what the detection frame looks like
            # cv2.imshow("Detector View", detection_frame)

            # Uncomment this line to replace the camera frame in QT window to show what the detection frame looks like
            # frame = detection_frame

        frame = cv2.cvtColor(
            frame, cv2.COLOR_GRAY2RGB
        )  # Converts the frame to RGB. Allows creation of colored circle around the particle.

        # Prints the time taken to detect the blob. Then updates the time stamp to current time.
        if initializations.VERBOSE:
            print(
                "Time taken to detect blob:", (time.time() - time_stamp) * 1000, " ms"
            )
            time_stamp = time.time()

        if self.flag_gen_log_data:
            self.data_extractor.record_datapoint(
                self.particle_locs, self.coil_vals, self.goal_locs
            )

        self.drawElements(
            frame,
            self.particle_locs,
            self.goal_locs,
            self.coil_locs,
            self.coil_vals,
            initializations.GUI_FRAME_WIDTH,
        )

        self.drawCameraFrame(frame)

        # Prints the time taken to draw all screen elements and record log.
        # Then updates the time stamp to current time.
        if initializations.VERBOSE:
            print(
                "Time taken to draw screen elements and create log datapoint (if activated):",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        if self.flag_record:
            frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame_BGR)

        # Prints the time taken to record video frame. Then updates the time stamp to current time.
        if initializations.VERBOSE:
            print(
                "Time taken to record video frame:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        # Saves snapshot of the frame if flag is true
        if self.flag_snapshot:
            self.flag_snapshot = False

            os.makedirs(
                self.path, exist_ok=True
            )  # Creates the new folder for current run of the program, if it does not exist

            frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            name = self.snapshot_name_root + str(self.snapshot_num) + ".jpg"
            self.snapshot_num = self.snapshot_num + 1

            cv2.imwrite(name, frame_BGR)

            print("Snapshot created: ", os.getcwd() + "/" + name)

        # Prints the time taken to take snapshot.
        if initializations.VERBOSE:
            print(
                "Time taken to take snapshot:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )

    def step(self, coil_vals = None, render = True):
        """Runs one video frame and calculates the control values"""
        time_stamp = (
            time.time()
        )  # Time stamp to measure how much time each block of code takes
        time_stamp_total = (
            time_stamp  # Time stamp to measure how much total time each the frame takes
        )

        self.process_frame()

        # Distance of particle to the goal to compare to GUI_MULTIPLE_GOALS_ACCURACY.
        # If it is lower, then the goal is considered "reached" and removed from the list.
        d = functions.distance(
            self.particle_locs[0][0],
            self.particle_locs[0][1],
            self.goal_locs[0][0],
            self.goal_locs[0][1],
        )

        # Removes the reached goal if there are more goals after it
        if len(self.goal_locs) > 1:
            if not self.flag_setting_multiple_goals:
                if d < initializations.GUI_MULTIPLE_GOALS_ACCURACY:
                    self.goal_locs.pop(0)

            else:
                self.pushButton_14.setText("Execute Goal Motion")

            if len(self.goal_locs) == 1:
                print("Multiple Goals Mode: OFF")
                self.pushButton_14.setText("Multiple Goals Mode: OFF")
                self.pushButton_15.setText("Set Goal")

        # Prints the time taken to run video loop. Then updates the time stamp to current time.
        if initializations.VERBOSE:
            print(
                "##################################################\nTime taken to run video frame:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        # If coil values are provided then use them. This is used to interface with the gym env.
        if coil_vals != None or self.flag_automatic_control:
            # If flag for automatic control is True, then get coil values using the control algorithm
            if self.flag_automatic_control:
                self.coil_vals = control_algorithm.get_coil_vals(
                    self.particle_locs[0], self.goal_locs[0], self.coil_vals, self.coil_locs
                )

            self.coil_vals = functions.limit_coil_vals(self.coil_vals)
            self.sendCartString(self.coil_vals)
            self.sendDiagString(self.coil_vals)

        # Prints the time taken to run control alogrithm. Then updates the time stamp to current time.
        if initializations.VERBOSE:
            print(
                "Time taken to run control algorithm:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        QApplication.processEvents()

        # Prints the time taken to process events in QApplication.
        if initializations.VERBOSE:
            print(
                "Time taken to process events in QApplication:",
                (time.time() - time_stamp) * 1000,
                " ms\n##################################################\n",
            )

        # Prints the total time taken to run one frame.
        if initializations.VERBOSE:
            print(
                "Total time taken to run one frame:",
                (time.time() - time_stamp_total) * 1000,
                " ms\n\n##################################################\n\n",
            )

    def run(self):
        """Keeps running frames until video is turned off or the window is closed"""
        while self.flag_video_on:
            self.step()

            # If program is closed, ensures that all solenoids are turned off
            if not self.isVisible():
                self.close()

    def close(self):
        self.resetVals()
        PySpin_lib.Cam_PySpin_Stop(self.cam)
        sys.exit()



if __name__ == "__main__":
    app = QApplication(sys.argv)  # Initialization of QT app
    mywin = mywindow()
    mywin.show()
    
    # If program crashes due to an error, this line resets all solenoids to zero.
    # But it causes the solenoid to be unaccessible for about 10 - 30 when program is started.
    # Need more investigation as to why this is so and more importantly how to solve this.
    # mywin.resetVals()

    sys.exit(app.exec_())  # Ensures we get a clean exit when QApplication exists
