import math
import os
import sys
import time

import cv2
import numpy as np
import pygame

from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.System.Control_Algorithms import control_algorithm
from gymnasium_env.envs.System.Library import blob_detection as Blob_detect
from gymnasium_env.envs.System.Library import data_extractor
from gymnasium_env.envs.System.Library import functions
from gymnasium_env.envs.System.Library import pyspin_wrapper as PySpin_lib
from gymnasium_env.envs.System.Library.pygame_recorder import ScreenRecorder


class CameraPygame:
    def __init__(self, init_with_camera_on: bool = False) -> None:
        """Initializes the pygame window and atrributes.

        Args:
            init_with_camera_on (bool, optional): Turns on camera on startup. Defaults to False.
        """
        pygame.init()
        self.window = pygame.display.set_mode(
            (initializations.GUI_FRAME_WIDTH, initializations.GUI_FRAME_HEIGHT)
        )
        pygame.display.set_caption("Solenoid Control")

        self.flag_video_on = False
        self.flag_detecting_objects = False
        self.flag_record = False
        self.flag_show_target_position = True
        self.flag_show_goal_vector = True
        self.flag_automatic_control = True
        self.flag_show_solenoid_state = True
        self.flag_snapshot = False
        self.flag_gen_log_data = False
        self.flag_setting_multiple_goals = False
        self.flag_log = False

        self.particle_locs = [
            [initializations.GUI_FRAME_WIDTH / 2, initializations.GUI_FRAME_HEIGHT / 2]
        ]
        self.particle_rads = [0]
        self.blob_lost = False

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
        ]  # Calculates the location for each solenoid using the angles. 0th solenoid is at pi/2.

        self.goal_locs = [list(map(int, [0, 0]))]

        self.cap = cv2.VideoCapture(0)
        self.data_extractor = data_extractor.data_extractor(
            self.coil_locs, initializations.COIL_NAMES
        )

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False
        )

        if init_with_camera_on:
            self.toggleVideoDisplay()
        time.sleep(1)

        self.coil_vals[1] = 1.0
        data_time = time.time()
        while (time.time() - data_time) < 5:
            self.coil_vals[1] = 1.0
            self.step(self.coil_vals)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        self.resetVals()
        self.flag_detecting_objects = True

    def toggleVideoDisplay(self) -> None:
        """Connects to camera and hold its feed."""
        self.flag_video_on = True
        print("Video turned on")

        _, _, self.cam_list, self.system1 = PySpin_lib.Cam_PySpin_Init()
        self.cam = self.cam_list[0]
        print("Camera: " + str(self.cam))
        PySpin_lib.Cam_PySpin_Connect(self.cam)

    def drawSolenoidState(self, frame, coil_value: list, x: int, y: int):
        """Draws state of solenoids on the frame.

        Args:
            frame : Frame object to draw on.
            coil_value (list): Values of coil current.
            x (int): x-coordinate of location at which to draw.
            y (int): y-coordinate of location at which to draw.
        """
        color = [0, 0, 0]
        for i in range(len(initializations.GUI_SOL_COLOR)):
            if initializations.GUI_SOL_COLOR[i] != 0:
                color[i] = (
                    (coil_value * initializations.GUI_SOL_COLOR[i])
                    // (
                        initializations.GUI_SOL_COLOR[i]
                        / initializations.GUI_COLOR_DIVS
                    )
                ) * (initializations.GUI_SOL_COLOR[i] / initializations.GUI_COLOR_DIVS)

        cv2.circle(frame, (x, y), initializations.GUI_SOL_RAD, color, -1)

    def getBlobData(self, frame):
        """Gets particle's location and size.

        Args:
            frame : Frame object from which to extract particle data.

        Returns:
            Frame object: Frame object with all processing applied related to particle detection.
        """
        ret, detection_frame = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
        detection_frame = self.background_subtractor.apply(
            detection_frame, learningRate=0.0000
        )
        detection_frame = cv2.medianBlur(detection_frame, 5)
        detection_frame = cv2.bitwise_not(detection_frame)

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
        )

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

    def drawBlobs(self, frame, particle_locs: list):
        """Draw detected location of particle.

        Args:
            frame : Frame object to draw on.
            particle_locs (list): Location of particle.
        """
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
        """Draws location of solenoids.

        Args:
            frame : Frame object to draw on.
            coil_locs (list): Location of coils.
            coil_vals (list): Current value for each coil.
        """
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

    def drawGoal(self, frame, goal_locs: list):
        """Draws goal location.

        Args:
            frame : Frame object to draw on.
            goal_locs (list): Location of goal position.
        """
        if self.flag_show_target_position:
            for i in range(len(goal_locs)):
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

    def drawGoalVector(self, frame, particle_loc: list, goal_locs: list):
        """Draws vector from particle to goal.

        Args:
            frame : Frame object to draw on.
            particle_loc (list): Location of particle.
            goal_locs (list): Location of goal.
        """
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
        self,
        frame,
        particle_locs: list,
        goal_locs: list,
        coil_locs: list,
        coil_vals: list,
        frame_size: int,
    ):
        """Draws all elements on the frame.

        Args:
            frame : Frame object to draw on.
            particle_locs (list): Location of particles.
            goal_locs (list): Location of goals.
            coil_locs (list): Location of coils.
            coil_vals (list): Current applied at each solenoid.
            frame_size (int): Size of frame.
        """
        self.drawGoalVector(frame, particle_locs[0], goal_locs)
        self.drawGoal(frame, goal_locs)
        self.drawBlobs(frame, particle_locs)
        self.drawSolenoids(frame, coil_locs, coil_vals)

    def process_frame(self):
        """Crops camera feed to set size, detects particle position and draws all screen elements."""
        time_stamp = time.time()
        frame = PySpin_lib.Cam_PySpin_GetImg(self.cam)

        w_crop = int((len(frame[0]) - initializations.GUI_FRAME_WIDTH) / 2)
        h_crop = int((len(frame) - initializations.GUI_FRAME_HEIGHT) / 2)

        frame = frame[h_crop : len(frame) - h_crop, w_crop : len(frame[0]) - w_crop]
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if initializations.VERBOSE:
            print(
                "##################################################\nTime taken to process camera frame:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        if self.flag_detecting_objects:
            detection_frame = self.getBlobData(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if initializations.VERBOSE:
            print(
                "Time taken to detect blob:", (time.time() - time_stamp) * 1000, " ms"
            )
            time_stamp = time.time()

        self.drawElements(
            frame,
            self.particle_locs,
            self.goal_locs,
            self.coil_locs,
            self.coil_vals,
            initializations.GUI_FRAME_WIDTH,
        )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flip(frame, 0)
        frame = pygame.surfarray.make_surface(frame)
        self.window.blit(frame, (0, 0))
        pygame.display.update()

        if self.flag_record:
            self.recorder.capture_frame(frame)

        if initializations.VERBOSE:
            print(
                "Time taken to draw screen elements and create log datapoint (if activated):",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        if initializations.VERBOSE:
            print(
                "Time taken to record video frame:",
                (time.time() - time_stamp) * 1000,
                " ms",
            )
            time_stamp = time.time()

        if self.flag_snapshot:
            self.flag_snapshot = False
            os.makedirs(self.path, exist_ok=True)
            frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            name = self.snapshot_name_root + str(self.snapshot_num) + ".jpg"
            self.snapshot_num = self.snapshot_num + 1
            cv2.imwrite(name, frame_BGR)
            print("Snapshot created: ", os.getcwd() + "/" + name)

        if initializations.VERBOSE:
            print(
                "Time taken to take snapshot:", (time.time() - time_stamp) * 1000, " ms"
            )

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
        """Sends inputted percentage (0-100) coil currents to the coil with inputted coil index

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

    def resetVals(self):
        """Resets all coil and spinbox values to zero"""
        for i in range(len(self.coil_vals)):
            self.manual(0, i, self.coil_vals)

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

            self.particle_start_loc[1] = math.sqrt(
                (r**2) - (self.particle_start_loc[0] ** 2)
            )
            self.particle_start_loc[1] *= np.random.choice([-1, 1])

        temp = self.goal_locs[0].copy()
        self.goal_locs[0] = self.particle_start_loc.copy()
        self.flag_automatic_control = True  # Turns on automatic control so particle can move to the starting location

        # Automatic control is kept on for 5 seconds so particle position can stablize
        data_time = time.time()
        while (time.time() - data_time) < 5:
            self.step()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        self.flag_automatic_control = False
        self.resetVals()
        self.goal_locs[0] = temp.copy()

        self.flag_setting_multiple_goals = (
            False  # Flag to detect if multiple goals mode has been turned on
        )
        self.flag_gen_log_data = (
            False  # Flag to detect if logging mode has been turned on
        )

    def getState(self, render: bool, n_obs: int = 1) -> tuple[list, list, list, list]:
        """Gets state of the environment.

        Args:
            render (bool): Whether to render the screen.
            n_obs (int, optional): Number of observations to take. Defaults to 1.

        Returns:
            tuple[list, list, list, list]: Tuple containing particle locations, goal location, coil values and coil location
            each in a separate list.
        """
        particle_locs = self.particle_locs.copy()

        while len(particle_locs) < n_obs:
            running = self.step()

            if not running:
                self.close()

            particle_locs.append(self.particle_locs[0].copy())

        return (
            particle_locs.copy(),
            self.goal_locs[0].copy(),
            self.coil_vals.copy(),
            self.coil_locs.copy(),
        )

    def step(self, coil_vals: list = None, render: bool = True) -> bool:
        """Steps the system by one frame.

        Args:
            coil_vals (list, optional): Coil values to apply. Defaults to None.
            render (bool, optional): Whether to render the system. Defaults to True.

        Returns:
            bool: Whether the system is still running.
        """
        try:
            time_stamp = time.time()
            time_stamp_total = time_stamp

            if coil_vals is None:
                if self.flag_automatic_control:
                    self.coil_vals = control_algorithm.get_coil_vals(
                        self.particle_locs[0],
                        self.goal_locs[0],
                        self.coil_vals,
                        self.coil_locs,
                    )

            else:
                self.coil_vals = list(functions.limit_coil_vals(coil_vals))

            self.sendCartString(self.coil_vals)
            self.sendDiagString(self.coil_vals)

            self.process_frame()

            if self.flag_gen_log_data:
                self.data_extractor.record_datapoint(
                    self.particle_locs, self.coil_vals, self.goal_locs
                )

            d = functions.distance(
                self.particle_locs[0][0],
                self.particle_locs[0][1],
                self.goal_locs[0][0],
                self.goal_locs[0][1],
            )

            if len(self.goal_locs) > 1:
                if not self.flag_setting_multiple_goals:
                    if d < initializations.GUI_MULTIPLE_GOALS_ACCURACY:
                        self.goal_locs.pop(0)
                else:
                    pass

                if len(self.goal_locs) == 1:
                    pass

            if initializations.VERBOSE:
                print(
                    "##################################################\nTime taken to run video frame:",
                    (time.time() - time_stamp) * 1000,
                    " ms",
                )
                time_stamp = time.time()

            if initializations.VERBOSE:
                print(
                    "Time taken to run control algorithm:",
                    (time.time() - time_stamp) * 1000,
                    " ms",
                )
                time_stamp = time.time()

            if initializations.VERBOSE:
                print(
                    "Time taken to process events in Pygame:",
                    (time.time() - time_stamp) * 1000,
                    " ms\n##################################################\n",
                )

            if initializations.VERBOSE:
                print(
                    "Total time taken to run one frame:",
                    (time.time() - time_stamp_total) * 1000,
                    " ms\n\n##################################################\n\n",
                )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.flag_log:
                        self.data_extractor.save_data("Data/Camera_Test")

                    self.close()

                elif event.key == pygame.K_l:

                    # If data was being logged, stop logging data and save the stored data in an excel file
                    if self.flag_log:
                        self.data_extractor.save_data("Data/Camera_Test")

                    # If data was not being logged, start data logging
                    else:
                        self.data_extractor.startDataSet("log_data")
                        self.data_extractor.startDataSeries()

                    self.flag_log = not self.flag_log

                # Checks if "v" was pressed
                elif event.key == pygame.K_v:

                    # If canvas was being recorded, stop recording canvas and save it
                    if self.flag_record:
                        print(
                            "Video saved to: Data/Camera_Test"
                            + self.date_time
                            + "/Recording.avi"
                        )
                        self.recorder.end_recording()  # Finishes and saves the recording

                    # If canvas was not being recorded, start recording canvas
                    else:
                        print("Starting")

                        os.makedirs(
                            "Data/Data/Camera_Test", exist_ok=True
                        )  # Creates the directory if it does not exist
                        self.recorder = ScreenRecorder(
                            initializations.SIM_FRAME_WIDTH,
                            initializations.SIM_FRAME_HEIGHT,
                            60,
                            "Data/Data/Camera_Test"
                            + "/Recording.avi",
                        )  # Passes the desired fps and starts the recorder

                    self.flag_record = not self.flag_record

        except Exception as e:
            print(e)
            return False

        return True

    def run(self):
        """Continuously runs the system."""
        running = True

        while running:
            running = self.step()

        self.close()

    def close(self):
        """Closes the window and system."""
        self.resetVals()
        pygame.display.quit()
        pygame.quit()

        PySpin_lib.Cam_PySpin_Stop(self.cam)
        sys.exit()


if __name__ == "__main__":
    mywin = CameraPygame(init_with_camera_on=True)
    mywin.run()
