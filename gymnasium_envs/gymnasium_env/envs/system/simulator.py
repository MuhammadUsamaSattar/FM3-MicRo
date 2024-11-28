import math
import time
import datetime
import os

import pygame
import pickle
import numpy as np

from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.System.Control_Algorithms import control_algorithm

from gymnasium_env.envs.System.Library import data_extractor
from gymnasium_env.envs.System.Library import functions
from gymnasium_env.envs.System.Library.pygame_recorder import ScreenRecorder


class Simulator:
    """Class for the simulator object which runs and updates the pygame canvas and hanldes inputs
    """    

    def __init__(self, framerate):
        """Initializes all the required class attributes such as particle location, particle velocity,
        goal locations, solenoids currents and flags for modifying the runFrame() method.
        """
        # Attribute to hold the pygame window
        self.window = None

        # Attribute to hold the pygame clock
        self.clock = None

        # Attribute to hold the pygame canvas
        self.canvas = pygame.Surface([initializations.SIM_FRAME_WIDTH, initializations.SIM_FRAME_HEIGHT])

        # Attribute to hold the pygame canvas framerate
        self.framerate = framerate

        # Attributes to hold the particle and goal locations
        self.particle_loc = [0, 0]
        self.goal_locs = [[0, 0]]

        self.particle_vel = [0, 0]  # Attribute to hold the particle velocity

        # Flags to detect if the user is continuously holding down the mouse button to move the particle or goal
        self.flag_edit_particle_mouse = False
        self.flag_edit_move_goal_mouse = False

        # Flags to detect if the user has clicked on the particle or goal text to modify the position
        self.flag_edit_particle_keyboard = False
        self.flag_edit_goal_keyboard = False

        # Text attributes to hold the entered text by the user after clicking on the text for particle and goal
        self.txt_particle = ""
        self.txt_goal = ""

        self.flag_setting_multiple_goals = False  # Flag to detect if multiple goals mode has been turned on
        self.flag_log = False  # Flag to detect if logging mode has been turned on

        self.flag_record = False  # Flag to detect if canvas is being recorded

        self.coil_vals = [
            0.0 for _ in initializations.COIL_NAMES
        ]  # Creates a 1x8 list of floats that will hold the current value for each coil

        self.coil_locs = [
            [
                int(
                    initializations.SIM_SOL_CIRCLE_RAD
                    * math.cos((math.pi / 2) - (x * (2 * math.pi / len(self.coil_vals))))
                ),
                int(
                    initializations.SIM_SOL_CIRCLE_RAD
                    * math.sin((math.pi / 2) - (x * (2 * math.pi / len(self.coil_vals))))
                ),
            ]
            for x in range(len(self.coil_vals))
        ]  # Calculates the location for each solenoid using the angles. 0th solenoid is at pi/2.

        self.data_extractor = data_extractor.data_extractor(
            self.coil_locs,
            initializations.COIL_NAMES,
        )  # Stores the data extractor object in an attribute

        self.date_time = str(datetime.datetime.now())  # Initial format is --> YYYY-MM-DD HH:MM:SS.SSSSSS
        self.date_time = self.date_time.replace(":", ".")  # Replace the colons --> YYYY-MM-DD HH.MM.SS.SSSSSS
        self.date_time = self.date_time.replace(" ", "_")  # Replace the space --> YYYY-MM-DD_HH.MM.SS.SSSSSS
        self.date_time = self.date_time[:-7]  # Remove the fractions of the second --> YYYY-MM-DD_HH.MM.SS

        self.video_num = 1  # Number of video so that each video has a unique name

        # Loads the saved prediction model for predicting the motion of the particle in the simulator
        model_filename = "Predict_rdot_from_I_r_R2_0.916_No_0.0_to_0.15_shuffled"
        with open(os.path.dirname(__file__) + "/Models/" + model_filename + ".pkl", "rb") as f:
            self.model = pickle.load(f)

    def renderInit(self):
        pygame.init()

        self.window = pygame.display.set_mode(
            [initializations.SIM_FRAME_WIDTH, initializations.SIM_FRAME_HEIGHT]
        )  # Set up the drawing window

        self.clock = pygame.time.Clock()

    def drawSolenoidState(self, canvas, coil_value, x, y):
        """Draws a circle at the [x, y] location with color depending upon coil_value

        Args:
            canvas : pygame canvas object on which to draw
            coil_value : Scaled value of the current
            x : x co-ordinate of the solenoid
            y : y co-ordinate of the solenoid
        """
        color = [0, 0, 0]

        # Iterates over every component in SIM_SOL_COLOR i.e., R, G and B
        for i in range(len(initializations.SIM_SOL_COLOR)):

            # Calculate color only if color component is not 0 and coil_value is not 0 since otherwise, the color is just 0
            if initializations.SIM_SOL_COLOR[i] != 0 and coil_value != 0:

                # Calculates solenoid color by creating SIM_COLOR_DIVS number
                # of division of color component and determining the smallest
                # division higher than the color value determined from
                # scaling the color component according to the coil_value.
                # An offset of 8 divisions is applied otherwise, at low
                # currents the color is too dark.
                color[i] = (
                    (((coil_value * initializations.SIM_SOL_COLOR[i])
                        // (initializations.SIM_SOL_COLOR[i] / initializations.SIM_COLOR_DIVS)) + 8)
                         * (initializations.SIM_SOL_COLOR[i] / initializations.SIM_COLOR_DIVS)
                )

        pygame.draw.circle(canvas, color, (x, y), initializations.SIM_SOL_RAD)

    def drawSolenoids(self, canvas, coil_locs, coil_vals):
        """Draws all the solenoids at their respective locations according to the coil_vals

        Args:
            canvas : pygame canvas object on which to draw
            coil_locs : List of size 8 containing locations of coils starting from Northern coil
            coil_vals : List of size 8 containing scaled currents starting from Northern coil
        """        
        for i in range(len(self.coil_locs)):
            coil_loc_img = functions.absolute_to_image(
                coil_locs[i][0],
                coil_locs[i][1],
                initializations.SIM_FRAME_WIDTH,
                initializations.SIM_FRAME_HEIGHT,
            )

            self.drawSolenoidState(
                canvas, coil_vals[i], coil_loc_img[0], coil_loc_img[1]
            )

    def drawParticle(self, canvas, particle_loc):
        """Draws the particle

        Args:
            canvas : pygame canvas object on which to draw
            particle_loc :Location of the particle
        """        
        particle_loc_img = functions.absolute_to_image(
            particle_loc[0],
            particle_loc[1],
            initializations.SIM_FRAME_WIDTH,
            initializations.SIM_FRAME_HEIGHT,
        )

        pygame.draw.circle(
            canvas,
            initializations.SIM_PARTICLE_COLOR,
            particle_loc_img,
            initializations.SIM_PARTICLE_RAD,
        )

    def drawGoalPosition(self, canvas, goal_loc):
        """Draws the goals

        Args:
            canvas : canvas object on which to draw
            goal_loc : Location of the goal
        """        
        goal_loc_img = functions.absolute_to_image(
            goal_loc[0],
            goal_loc[1],
            initializations.SIM_FRAME_WIDTH,
            initializations.SIM_FRAME_HEIGHT,
        )

        pygame.draw.line(
            canvas,
            initializations.SIM_GOAL_COLOR,
            (
                goal_loc_img[0] + initializations.SIM_GOAL_SIZE,
                goal_loc_img[1] + initializations.SIM_GOAL_SIZE,
            ),
            (
                goal_loc_img[0] - initializations.SIM_GOAL_SIZE,
                goal_loc_img[1] - initializations.SIM_GOAL_SIZE,
            ),
            initializations.SIM_GOAL_THICKNESS,
        )

        pygame.draw.line(
            canvas,
            initializations.SIM_GOAL_COLOR,
            (
                goal_loc_img[0] + initializations.SIM_GOAL_SIZE,
                goal_loc_img[1] - initializations.SIM_GOAL_SIZE,
            ),
            (
                goal_loc_img[0] - initializations.SIM_GOAL_SIZE,
                goal_loc_img[1] + initializations.SIM_GOAL_SIZE,
            ),
            initializations.SIM_GOAL_THICKNESS,
        )

    def drawGoalVector(self, canvas, particle_loc, goal_loc):
        """Draws the vector from particle to goal

        Args:
            canvas : canvas object on which to draw
            particle_loc : Location of particle
            goal_loc : Location of goal
        """        
        particle_loc_img = functions.absolute_to_image(
            particle_loc[0],
            particle_loc[1],
            initializations.SIM_FRAME_WIDTH,
            initializations.SIM_FRAME_HEIGHT,
        )
        goal_loc_img = functions.absolute_to_image(
            goal_loc[0],
            goal_loc[1],
            initializations.SIM_FRAME_WIDTH,
            initializations.SIM_FRAME_HEIGHT,
        )

        pygame.draw.line(
            canvas,
            initializations.SIM_GOAL_VECTOR_COLOR,
            (particle_loc_img[0], particle_loc_img[1]),
            (goal_loc_img[0], goal_loc_img[1]),
            initializations.SIM_GOAL_VECTOR_THICKNESS,
        )

    def drawText(self, canvas, particle_loc, goal_loc, flag_edit_particle_keyboard, flag_edit_goal_keyboard):
        """Draws all text objects on canvas

        Args:
            canvas : pygame canvas object on which to draw
            particle_loc : Location of the particle
            goal_loc : Location of the current goal
            flag_edit_particle_keyboard : Flag to detect if the user has clicked on the particle text to modify the position
            flag_edit_goal_keyboard : Flag to detect if the user has clicked on the goal text to modify the position
        """        
        font = pygame.font.Font("freesansbold.ttf", initializations.SIM_TEXT_SIZE)

        # Draws the text "Particle Location: "
        txt = "Particle Location: "
        text = font.render(txt, True, initializations.SIM_PARTICLE_LOC_TEXT_COLOR)
        textRect = text.get_rect()
        textRect.bottomleft = (
            initializations.SIM_TEXT_SIZE / 4,
            initializations.SIM_FRAME_HEIGHT - (3 * initializations.SIM_TEXT_SIZE),
        )
        canvas.blit(text, textRect)

        # Draws the text "Goal Location: "
        txt = "Goal Location: "
        text = font.render(txt, True, initializations.SIM_PARTICLE_LOC_TEXT_COLOR)
        textRect = text.get_rect()
        textRect.bottomleft = (
            initializations.SIM_TEXT_SIZE / 4,
            initializations.SIM_FRAME_HEIGHT - (1 * initializations.SIM_TEXT_SIZE),
        )
        canvas.blit(text, textRect)

        # If user is not editing particle location through text input, then display
        # the particle position otherwise display the text that has been entered so far.
        if not flag_edit_particle_keyboard:
            txt = "{: 4.0f} , {: 4.0f}".format(particle_loc[0], particle_loc[1])
            text = font.render(txt, True, initializations.SIM_PARTICLE_LOC_TEXT_COLOR)
            self.textRect_particle = text.get_rect()
            self.textRect_particle.bottomleft = (
                initializations.SIM_TEXT_SIZE * 10,
                initializations.SIM_FRAME_HEIGHT - (3 * initializations.SIM_TEXT_SIZE),
            )
            canvas.blit(text, self.textRect_particle)

        else:
            text = font.render(
                self.txt_particle, True, initializations.SIM_PARTICLE_LOC_TEXT_COLOR
            )
            self.textRect_particle = text.get_rect()
            self.textRect_particle.bottomleft = (
                initializations.SIM_TEXT_SIZE * 10,
                initializations.SIM_FRAME_HEIGHT - (3 * initializations.SIM_TEXT_SIZE),
            )
            canvas.blit(text, self.textRect_particle)

        # If user is not editing goal location through text input, then display
        # the goal position otherwise display the text that has been entered so far.
        if not flag_edit_goal_keyboard:
            txt = "{: 4.0f} , {: 4.0f}".format(goal_loc[0], goal_loc[1])
            text = font.render(txt, True, initializations.SIM_PARTICLE_LOC_TEXT_COLOR)
            self.textRect_goal = text.get_rect()
            self.textRect_goal.bottomleft = (
                initializations.SIM_TEXT_SIZE * 8,
                initializations.SIM_FRAME_HEIGHT - (1 * initializations.SIM_TEXT_SIZE),
            )
            canvas.blit(text, self.textRect_goal)

        else:
            text = font.render(
                self.txt_goal, True, initializations.SIM_PARTICLE_LOC_TEXT_COLOR
            )
            self.textRect_goal = text.get_rect()
            self.textRect_goal.bottomleft = (
                initializations.SIM_TEXT_SIZE * 8,
                initializations.SIM_FRAME_HEIGHT - (1 * initializations.SIM_TEXT_SIZE),
            )
            canvas.blit(text, self.textRect_goal)

        # Displays text indication for Multiple Goals Mode and its current status which can be On, Off or Executing Multiple Goals Motion
        if self.flag_setting_multiple_goals:
            txt = "(M)ultiple Goals Mode: On"
        else:
            if len(self.goal_locs) > 1:
                txt = "Executing Multiple Goals Motion"
            else:
                txt = "(M)ultiple Goals Mode: Off"
        text = font.render(txt, True, initializations.SIM_MSC_TXT_COLOR)
        textRect = text.get_rect()
        textRect.bottomleft = (
            initializations.SIM_TEXT_SIZE * 0.5,
            initializations.SIM_TEXT_SIZE * 2,
        )
        canvas.blit(text, textRect)

        # Displays text indication for Logging Mode and its current status which can be On or Off
        if self.flag_log:
            txt = "(L)ogging Mode: On"
        else:
            txt = "(L)ogging Mode: Off"
        text = font.render(txt, True, initializations.SIM_MSC_TXT_COLOR)
        textRect = text.get_rect()
        textRect.bottomleft = (
            initializations.SIM_TEXT_SIZE * 0.5,
            initializations.SIM_TEXT_SIZE * 4,
        )
        canvas.blit(text, textRect)

        # Displays text indication for Video Recording and its current status which can be On or Off
        if self.flag_record:
            txt = "(V)ideo Recording: On"
        else:
            txt = "(V)ideo Recording: Off"
        text = font.render(txt, True, initializations.SIM_MSC_TXT_COLOR)
        textRect = text.get_rect()
        textRect.bottomleft = (
            initializations.SIM_TEXT_SIZE * 0.5,
            initializations.SIM_TEXT_SIZE * 6,
        )
        canvas.blit(text, textRect)

    def drawElements(
        self,
        canvas,
        particle_loc,
        goal_locs,
        coil_locs,
        coil_vals,
        flag_edit_particle_keyboard,
        flag_edit_goal_keyboard,
        render,
    ):
        """Draws all canvas elements

        Args:
            canvas : canvas on which to draw
            particle_loc : Location of particle
            goal_locs : Locations of goals
            coil_locs : List of size 8 containing locations of solenoids
            coil_vals : List of size 8 containing scaled currents
            flag_edit_particle_keyboard : Flag to detect if the user has clicked on the particle text to modify the position
            flag_edit_goal_keyboard : Flag to detect if the user has clicked on the goal text to modify the position
        """
        if render:
            self.drawText(canvas, particle_loc, goal_locs[0], flag_edit_particle_keyboard, flag_edit_goal_keyboard)
        self.drawSolenoids(canvas, coil_locs, coil_vals)
        self.drawGoalVector(canvas, particle_loc, goal_locs[0])

        for i in range(len(goal_locs)):
            # If multiple goals are being set, skip the first goal since it is the goal before multiple goals were being set.
            # This old goal is necessary to keep in case the user quits the multiple goal mode without setting any new ones.
            if self.flag_setting_multiple_goals and i == 0:
                continue

            self.drawGoalPosition(canvas, goal_locs[i])

        self.drawParticle(canvas, particle_loc)

    def calcRDot(self, I, r, alpha):
        """Calculates the velocity of particle towards a solenoid

        Args:
            I : Current applied to the solenoid
            r : Distance between particle and solenoid
            alpha : Angle between particle and solenoid

        Returns:
            float: Velocity towards the solenoid
        """        
        r_dot = [0, 0]

        # If current is 0, the no calculation is done since velocity should be 0
        if I != 0:
            # Predicts the velocity from current and distance and multiplies it with a factor which was determined by trial
            r_dot_pred = 0.20 * self.model.predict([[abs(I), r]])[0]

            # Calculates the components of velocity in x and y directions
            r_dot = [
                -r_dot_pred * math.cos(alpha),
                -r_dot_pred * math.sin(alpha),
            ]

        return r_dot

    def updateParticleLocation(
        self, particle_loc, particle_vel, flag_edit_particle_mouse, coil_vals, coil_locs, prev_time
    ):
        """Updates the particle location

        Args:
            particle_loc : Location of the particle
            particle_vel : Velocity of the particle
            flag_edit_particle_mouse : Flag to detect if the user is moving the particle through mouse
            coil_vals : List of size 8 containing scaled current values where 0th value corresponds to the Northern coil
            coil_locs : List of size 8 containing coil locations where 0th value corresponds to the Northern coil
            prev_time : Time stamp of previous frame

        Returns:
            list: 1x2 list containing the new particle location
            list: 1x2 list containing the new particle velocity
            float: Time stamp of current frame
        """        
        # Only updates location if particle is not being moved by the mouse
        if not flag_edit_particle_mouse:
            dt = time.time() - prev_time

            temp_particle_vel = [0, 0]

            # For each coil, calculates the predicted velocity and adds it to temp_particle_vel
            for i in range(len(coil_vals)):
                r = functions.distance(
                    coil_locs[i][0], coil_locs[i][1], particle_loc[0], particle_loc[1]
                )
                alpha = math.atan2(
                    coil_locs[i][1] - particle_loc[1], coil_locs[i][0] - particle_loc[0]
                )

                r_dot = self.calcRDot(coil_vals[i], r, alpha)

                temp_particle_vel = [
                    temp_particle_vel[0] + r_dot[0],
                    temp_particle_vel[1] + r_dot[1],
                ]

            # New particle velocity = current particle velocity - fluid drag + total added velocity + small random value
            fluid_drag_factor = 0.90
            particle_vel = [
                fluid_drag_factor * particle_vel[0]
                + temp_particle_vel[0]
                + np.random.normal(loc=0.0, scale=3),
                fluid_drag_factor * particle_vel[1]
                + temp_particle_vel[1]
                + np.random.normal(loc=0.0, scale=3),
            ]

            # New particle location = current particle location + particle velocity * dt
            particle_loc = [
                particle_loc[0] + particle_vel[0] * dt,
                particle_loc[1] + particle_vel[1] * dt,
            ]

            # Prevents the particle from going too far off from the canvas.
            # Most likely not required anymore since particle position is updated according to a robust model now.
            # Nevertheless good to keep in case of unexpectedly high velocities.
            out_of_bounds_control_factor = 1.1
            for i in range(2):
                if particle_loc[i] < (
                    -out_of_bounds_control_factor * initializations.SIM_FRAME_WIDTH / 2
                ):
                    particle_loc[i] = (
                        -out_of_bounds_control_factor
                        * initializations.SIM_FRAME_WIDTH
                        / 2
                    )
                    particle_vel[i] = 0

                elif particle_loc[i] > (
                    out_of_bounds_control_factor * initializations.SIM_FRAME_WIDTH / 2
                ):
                    particle_loc[i] = (
                        out_of_bounds_control_factor
                        * initializations.SIM_FRAME_WIDTH
                        / 2
                    )
                    particle_vel[i] = 0

        else:
            particle_vel = [0, 0]

        return particle_loc, particle_vel, time.time()

    def getClosetPointInBounds(self, mouse_pos):
        """Gets the closet location from the mouse_pos within the circle traced by the solenoids

        Args:
            mouse_pos : Position of the mouse

        Returns:
            list: Closest location from mouse in bounds
        """     
        # Gets the sign of the x and y locations
        if mouse_pos[0] > 0:
            mult_x = 1
        else:
            mult_x = -1

        if mouse_pos[1] > 0:
            mult_y = 1
        else:
            mult_y = -1

        # Calculates the closet location on the solenoid circle through equations x = r/(mult_x*(1 + ((y/x)^2))) and y = r/(mult_y*(1 + ((x/y)^2))).
        # If x or y is 0, then the new x or y is set as 0 otherwise y/x or x/y will become undefined.
        if mouse_pos[0] == 0:
            loc = [
                0,
                mult_y
                * initializations.SIM_SOL_CIRCLE_RAD
                / math.sqrt(1 + (mouse_pos[0] / mouse_pos[1]) ** 2),
            ]

        elif mouse_pos[1] == 0:
            loc = [
                mult_x
                * initializations.SIM_SOL_CIRCLE_RAD
                / math.sqrt(1 + (mouse_pos[1] / mouse_pos[0]) ** 2),
                0,
            ]

        else:
            loc = [
                mult_x
                * initializations.SIM_SOL_CIRCLE_RAD
                / math.sqrt(1 + (mouse_pos[1] / mouse_pos[0]) ** 2),
                mult_y
                * initializations.SIM_SOL_CIRCLE_RAD
                / math.sqrt(1 + (mouse_pos[0] / mouse_pos[1]) ** 2),
            ]

        return loc

    def getBoundedMouseLocation(self):
        """Get location of mouse that is within the bounds formed by the circle connecting all solenoids

        Returns:
            list: In-bounds location of mouse
        """        
        pos = list(pygame.mouse.get_pos())
        pos = functions.image_to_absolute(
            pos[0],
            pos[1],
            initializations.SIM_FRAME_WIDTH,
            initializations.SIM_FRAME_HEIGHT,
        )

        # If mouse location is out of bounds formed by the solenoid circle, get the closest point within bounds
        if (
            math.sqrt((pos[0] ** 2) + (pos[1] ** 2))
            > initializations.SIM_SOL_CIRCLE_RAD
        ):
            pos = self.getClosetPointInBounds(pos)

        pos = list(map(int, pos))

        return pos

    def eventHandler(self):
        """Handles all events such as mouse inputs, keyboard inputs and window closes

        Returns:
            bool: Determines if the game is still running
        """        
        running = True

        # If the game has been quit, sets running as False and saves log data if data logging mode was active
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.flag_log:
                    self.data_extractor.save_data(
                        "Data/Simulator_Data_" + self.date_time
                    )

                running = False

            # Handles mouse button pressed down input
            elif event.type == pygame.MOUSEBUTTONDOWN:

                # Checks if left click was pressed
                if event.button == 1:
                    pos = pygame.mouse.get_pos()

                    # If text for particle location was pressed, starts editing particle location
                    if (
                        (pos[0] > self.textRect_particle[0])
                        and (pos[0] < (self.textRect_particle[0] + self.textRect_particle[2]))
                        and (pos[1] > self.textRect_particle[1])
                        and (pos[1] < (self.textRect_particle[1] + self.textRect_particle[3]))
                    ):
                        self.flag_edit_particle_keyboard = True

                        self.flag_edit_goal_keyboard = False
                        self.txt_goal = ""

                    # If text for goal location was pressed, starts editing goal location
                    elif (
                        (pos[0] > self.textRect_goal[0])
                        and (pos[0] < (self.textRect_goal[0] + self.textRect_goal[2]))
                        and (pos[1] > self.textRect_goal[1])
                        and (pos[1] < (self.textRect_goal[1] + self.textRect_goal[3]))
                    ):
                        self.flag_edit_goal_keyboard = True

                        self.flag_edit_particle_keyboard = False
                        self.txt_particle = ""

                    # Otherwise, starts moving the particle
                    else:
                        self.flag_edit_particle_mouse = True

                        self.flag_edit_goal_keyboard = False
                        self.txt_goal = ""

                        self.flag_edit_particle_keyboard = False
                        self.txt_particle = ""

                # Checks if right click was pressed and start moving the goal
                elif event.button == 3:
                    self.flag_edit_move_goal_mouse = True

                    self.flag_edit_goal_keyboard = False
                    self.txt_goal = ""

                    self.flag_edit_particle_keyboard = False
                    self.txt_particle = ""

            # Checks if mouse button has been let go and deactivates the flag for mouse edit of either particle or goal
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.flag_edit_particle_mouse = False

                elif event.button == 3:
                    self.flag_edit_move_goal_mouse = False

            # Checks if a keyboard button was pressed
            elif event.type == pygame.KEYDOWN:

                # If user was editing particle or goal location through keyboard, then add the pressed key to the entered text
                # In case of Return key, attempts to assign the entered text to the particle location
                if self.flag_edit_particle_keyboard or self.flag_edit_goal_keyboard:

                    # Checks if return key was pressed
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:

                        # If user was editing the particle or goal location, attempts to break the entered string into
                        # two values separated by a comma otherwise throws an exception.
                        # If value was out-of-bounds, then reverts to the previous location and displays "Location out of bounds!"
                        if self.flag_edit_particle_keyboard:
                            self.flag_edit_particle_keyboard = False

                            try:
                                data = self.txt_particle.replace(" ", "").split(",")
                                data = list(map(int, data))

                                if (
                                    math.sqrt((data[0] ** 2) + (data[1] ** 2))
                                    <= initializations.SIM_SOL_CIRCLE_RAD
                                ):
                                    self.particle_loc = data
                                else:
                                    print("Location out of bounds!")

                            except:
                                print("Invalid format")

                            self.txt_particle = ""

                        else:
                            self.flag_edit_goal_keyboard = False

                            try:
                                data = self.txt_goal.replace(" ", "").split(",")
                                data = list(map(int, data))

                                if (
                                    math.sqrt((data[0] ** 2) + (data[1] ** 2))
                                    <= initializations.SIM_SOL_CIRCLE_RAD
                                ):
                                    if self.flag_setting_multiple_goals:
                                        if self.goal_locs[-1] != data:
                                            self.goal_locs.append(data)

                                    else:
                                        self.goal_locs = [data]

                                else:
                                    print("Location out of bounds!")

                            except:
                                print("Invalid format")

                            self.txt_goal = ""

                    # Checks if backspace was pressed and removes the last character in the attribute self.txt_particle which contains the entered text
                    elif event.key == pygame.K_BACKSPACE:
                        if self.flag_edit_particle_keyboard:
                            self.txt_particle = self.txt_particle[:-1]

                        else:
                            self.txt_goal = self.txt_goal[:-1]

                    # Checks if the entered character is space, comma, minus, plus or a number.
                    # If it is one of these characters, then adds it to the text.
                    elif (
                        (event.key > 47 and event.key < 58)
                        or (event.key > 1073741912 and event.key < 1073741923)
                        or event.key == pygame.K_SPACE
                        or event.key == pygame.K_COMMA
                        or event.key == pygame.K_MINUS
                        or event.key == pygame.K_PLUS
                        or event.key == pygame.K_KP_MINUS
                        or event.key == pygame.K_KP_PLUS
                    ):

                        if self.flag_edit_particle_keyboard:
                            self.txt_particle += event.unicode

                        else:
                            self.txt_goal += event.unicode

                # Checks if "m" was pressed
                elif event.key == pygame.K_m:
                    # If number of goals is more than one, then removes the first element and adds a duplicate of the last value.
                    # First element is removed since it is the goal prior to the multiple goals mode.
                    # Last element is duplicated because "Executing Multiple Goals Motion" is only displayed as long as number
                    # of elements in goals list is more than 1.
                    if len(self.goal_locs) > 1:
                        if self.flag_setting_multiple_goals:
                            self.goal_locs.pop(0)
                            self.goal_locs.append(self.goal_locs[-1])

                        else:
                            self.goal_locs = [self.goal_locs[0]]

                    self.flag_setting_multiple_goals = (
                        not self.flag_setting_multiple_goals
                    )

                # Checks if "l" was pressed
                elif event.key == pygame.K_l:

                    # If data was being logged, stop logging data and save the stored data in an excel file
                    if self.flag_log:
                        self.data_extractor.save_data(
                            "Data/Simulator_Data_" + self.date_time
                        )

                    # If data was not being logged, start data logging
                    else:
                        self.data_extractor.startDataSet("log_data")
                        self.data_extractor.startDataSeries()

                    self.flag_log = not self.flag_log

                # Checks if "v" was pressed
                elif event.key == pygame.K_v:

                    # If canvas was being recorded, stop recording canvas and save it
                    if self.flag_record:
                        print("Video saved to: Data/Simulator_Data_" + self.date_time + "/Recording_" + str(self.video_num) + ".avi")
                        self.recorder.end_recording()  # Finishes and saves the recording

                        self.video_num += 1

                    # If canvas was not being recorded, start recording canvas
                    else:
                        print("Starting")

                        os.makedirs("Data/Simulator_Data_" + self.date_time, exist_ok=True)  # Creates the directory if it does not exist
                        self.recorder = ScreenRecorder(
                            initializations.SIM_FRAME_WIDTH,
                            initializations.SIM_FRAME_HEIGHT,
                            60,
                            "Data/Simulator_Data_" + self.date_time + "/Recording_" + str(self.video_num) + ".avi"
                        )  # Passes the desired fps and starts the recorder

                    self.flag_record = not self.flag_record

                # Checks if "f" was pressed
                elif event.key == pygame.K_f:

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
                            'Invalid format of path locations. The file must consists of each path point on separate line.' 
                            'Each point must consist of two numbers separated by ",".'
                        )

        return running

    def step(self, running, prev_time, coil_vals, render):
        """Runs one frame

        Args:
            running : Determines whether program is still running
            coil_vals : Normalized coil current values
            prev_time : Time stamp of previous frame in seconds
            render : Determines if canvas should be rendered

        Returns:
            bool: Determines whether program is still running
            float: Time stamp of previous frame in seconds
        """        
        # Calculates d, distance from first goal, and removes the goal from the list if d is less than initializations.SIM_MULTIPLE_GOALS_ACCURACY
        d = functions.distance(
            self.particle_loc[0],
            self.particle_loc[1],
            self.goal_locs[0][0],
            self.goal_locs[0][1],
        )

        if (
            not self.flag_setting_multiple_goals
            and len(self.goal_locs) > 1
            and d < initializations.SIM_MULTIPLE_GOALS_ACCURACY
        ):
            self.goal_locs.pop(0)

        # Records a data point in the data logger
        if self.flag_log:
            self.data_extractor.record_datapoint(
                self.particle_loc, coil_vals, self.goal_locs, 1
            )

        self.particle_loc, self.particle_vel, prev_time = self.updateParticleLocation(
            self.particle_loc,
            self.particle_vel,
            self.flag_edit_particle_mouse,
            coil_vals,
            self.coil_locs,
            prev_time,
        )

        coil_vals = functions.limit_coil_vals(coil_vals)

        self.canvas.fill((255, 255, 255))

        if render:
            if self.window is None and self.clock is None:
                self.renderInit()

            running = self.eventHandler()

        else:
            if self.window != None:
                self.close()

        self.drawElements(
            self.canvas,
            self.particle_loc,
            self.goal_locs,
            self.coil_locs,
            coil_vals,
            self.flag_edit_particle_keyboard,
            self.flag_edit_goal_keyboard,
            render,
        )

        if render:
            #pygame.display.flip()
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.framerate)  # Sleeps in between loop so that fps is equal to self.framerate

        if self.flag_record:
            self.recorder.capture_frame(self.canvas)

        # Changes the position of the particle to the in-bound mouse location
        if self.flag_edit_particle_mouse:
            self.particle_loc = self.getBoundedMouseLocation()

        # Changes the position of the goal to the in-bound mouse location. If multiple goals mode was on, then goal is added to the list,
        # otherwise, it replaces the previous goal
        if self.flag_edit_move_goal_mouse:
            pos = self.getBoundedMouseLocation()

            if self.flag_setting_multiple_goals:
                if self.goal_locs[-1] != pos:
                    self.goal_locs.append(pos)

            else:
                self.goal_locs = [pos]

        return running, prev_time

    def getState(self):
        return self.particle_loc.copy(), self.goal_locs[0].copy(), self.coil_vals.copy(), self.coil_locs.copy()

    def resetAtRandomLocs(self, seed):
        """Resets the game by resetting all values and placing the partcile and goal at random locations.

        Args:
            seed : Seed value that determines the pseudo-random number
        """
        factor = 1 / math.sqrt(2)
        # Attributes to hold the particle and goal locations
        self.particle_loc = list(
            np.random.randint(
                -initializations.SIM_SOL_CIRCLE_RAD * factor,
                initializations.SIM_SOL_CIRCLE_RAD * factor,
                size=2,
                dtype=int,
            )
        )

        self.goal_locs[0] = self.particle_loc.copy()
        factor *= 3 / 4

        while (
            functions.distance(
                self.particle_loc[0],
                self.particle_loc[1],
                self.goal_locs[0][0],
                self.goal_locs[0][1],
            )
            < 200
        ):
            self.goal_locs = [
                list(
                    np.random.randint(
                        -initializations.SIM_SOL_CIRCLE_RAD * factor,
                        initializations.SIM_SOL_CIRCLE_RAD * factor,
                        size=2,
                        dtype=int,
                    )
                )
            ]

        self.particle_vel = [0, 0]  # Attribute to hold the particle velocity

        # Flags to detect if the user is continuously holding down the mouse button to move the particle or goal
        self.flag_edit_particle_mouse = False
        self.flag_edit_move_goal_mouse = False

        # Flags to detect if the user has clicked on the particle or goal text to modify the position
        self.flag_edit_particle_keyboard = False
        self.flag_edit_goal_keyboard = False

        # Text attributes to hold the entered text by the user after clicking on the text for particle and goal
        self.txt_particle = ""
        self.txt_goal = ""

        self.flag_setting_multiple_goals = False  # Flag to detect if multiple goals mode has been turned on
        self.flag_log = False  # Flag to detect if logging mode has been turned on

        self.flag_record = False  # Flag to detect if canvas is being recorded

        self.coil_vals = [
            0.0 for _ in initializations.COIL_NAMES
        ]  # Creates a 1x8 list of floats that will hold the current value for each coil

        self.data_extractor = data_extractor.data_extractor(
            self.coil_locs,
            initializations.COIL_NAMES,
        )  # Stores the data extractor object in an attribute

        self.date_time = str(datetime.datetime.now())  # Initial format is --> YYYY-MM-DD HH:MM:SS.SSSSSS
        self.date_time = self.date_time.replace(":", ".")  # Replace the colons --> YYYY-MM-DD HH.MM.SS.SSSSSS
        self.date_time = self.date_time.replace(" ", "_")  # Replace the space --> YYYY-MM-DD_HH.MM.SS.SSSSSS
        self.date_time = self.date_time[:-7]  # Remove the fractions of the second --> YYYY-MM-DD_HH.MM.SS

        self.video_num = 1  # Number of video so that each video has a unique name

    def close(self):
        pygame.display.quit()
        pygame.quit()

if __name__ == "__main__":
    sim = Simulator(1)

    running = True
    prev_time = time.time()
    while running:
        particle_loc, goal_locs, coil_vals, coil_locs = sim.getState()

        coil_vals = control_algorithm.get_coil_vals(
            particle_loc, goal_locs, coil_vals, coil_locs
        )

        running, prev_time = sim.step(running, prev_time, coil_vals, False)

    pygame.quit()
