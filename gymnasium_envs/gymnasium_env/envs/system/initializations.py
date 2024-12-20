import serial

####################################################################################################
# Common Parameters
####################################################################################################

VERBOSE = False
COIL_NAMES = (
    "North",
    "NorthEast",
    "East",
    "SouthEast",
    "South",
    "SouthWest",
    "West",
    "NorthWest",
)  # Names of coil in the same order as they are saved in coil_locs and coil_vals
MAX_FRAC_CURR = 0.6  # Maximum fraction of current value that is allowed for automatic control

####################################################################################################
# GUI Parameters
####################################################################################################

GUI_MULTIPLE_GOALS_ACCURACY = 40  # Pixel distance of particle from goal before it is considered to have "reached" the goal

# Establish communication with the microcontrollers
GUI_BAUD_RATE = 500000  # This must match the baudRate set in the Arduio IDE
#GUI_SER_CART = serial.Serial("COM4", GUI_BAUD_RATE, timeout=0.5)  # Defines the COM port for Cartesian solenoids
#GUI_SER_DIAG = serial.Serial("COM3", GUI_BAUD_RATE, timeout=0.5)  # Defines the COM port for Diagonal solenoids

# These variables are used when scaling the solenoid values that are sent to the Arduino microcontrollers
# An arbitrary value is added to all coil values before sending them. This value is the subtracted from the recieved values in the Arduino code
GUI_INPUT_RANGE = float(100)  # This value coresponds to the -100 to +100 range of manual values accepted from the interface. It is used to normalize the output value.
GUI_OUTPUT_RANGE = float(1299)  # This variable is used to scale values that are sent to the microcontrollers
GUI_ZERO_VAL = float(2000)  # In order to always send a positive number to the microcontrollers (and thus avoid sending a negative sign),

# Frame's parameters
# These values are according to the size of the picture display designed in QT Designer. 
# If this needs to be changed then the UI must also be changed
# The viewing area visible through the viewing shaft is 1050x1050 so it represents the 
# maximum limit of size that can be set in UI.
GUI_FRAME_WIDTH = 720  
GUI_FRAME_HEIGHT = GUI_FRAME_WIDTH
GUI_FRAME_RATE_VIDEO_RECORDING = 24

GUI_SLEEP_INT = 0.01  # Sets the sleep values to prevent the computer from sending the signals too fast which cause them to be discarded sometimes

# Grid's parameters
GUI_GRID_COLOR = (0, 0, 0)
GUI_GRID_AXIS_COLOR = (0, 0, 0)
GUI_GRID_THK = 1
GUI_GRID_AXIS_THK = 2

# Goal's parameters
GUI_GOAL_SIZE = 10
GUI_GOAL_COLOR = (0, 0, 255)
GUI_GOAL_THICKNESS = 2

# Goal vector's parameters
GUI_GOAL_VECTOR_COLOR = (0, 255, 0)
GUI_GOAL_VECTOR_THICKNESS = 2

# Blob's parameters
GUI_BLOB_COLOR = (255, 0, 0)
GUI_BLOB_THK = 2

# Solenoids' parameters
GUI_SOL_RAD = 10
GUI_SOL_COLOR = (0, 255, 0)
GUI_SOL_CIRCLE_RAD = 320  # Solenoid circle radius
GUI_COLOR_DIVS = 32  # Number of divisions of each component of color to make. This is basically the resolution of the state display.

####################################################################################################
# Simulator Parameters
####################################################################################################

SIM_MULTIPLE_GOALS_ACCURACY = 10  # Pixel distance of particle from goal before it is considered to have "reached" the goal

# Frame's parameters
SIM_FRAME_WIDTH = 800
SIM_FRAME_HEIGHT = SIM_FRAME_WIDTH

# Particle's parameters
SIM_PARTICLE_COLOR = (47, 79, 79)
SIM_PARTICLE_RAD = 10

# Goals' parameters
SIM_GOAL_SIZE = GUI_GOAL_SIZE
SIM_GOAL_COLOR = (255, 0, 0)
SIM_GOAL_THICKNESS = 4

# Goal vector's parameters
SIM_GOAL_VECTOR_COLOR = (0, 255, 0)
SIM_GOAL_VECTOR_THICKNESS = 4

# Solenoids' parameters
SIM_SOL_RAD = GUI_SOL_RAD
SIM_SOL_COLOR = GUI_SOL_COLOR
SIM_SOL_CIRCLE_RAD = GUI_SOL_CIRCLE_RAD
SIM_COLOR_DIVS = GUI_COLOR_DIVS  # Number of divisions of each component of color to make. This is basically the resolution of the state display.

# Text's parameters
SIM_PARTICLE_LOC_TEXT_COLOR = (0, 0, 0)  # Color of text that displays particle location
SIM_GOAL_LOC_TEXT_COLOR = SIM_PARTICLE_LOC_TEXT_COLOR  # Color of text that displays goal location
SIM_MSC_TXT_COLOR = (0, 0, 0)  # Color of text that displays multiple goals mode and logging mode

# Size of text that displays particle and goal locations. 
# Text of multiple goals mode and logging mode is also scaled according to this value.
SIM_TEXT_SIZE = 16
