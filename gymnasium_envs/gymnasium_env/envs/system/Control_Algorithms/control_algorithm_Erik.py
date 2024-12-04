import math

from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.System.Library import functions


def get_coil_vals(particle_loc, goal_loc, coil_vals, coil_locs):
    """Calculates the current values for given particle location, goal location and coil values

    Args:
        particle_loc (list): List of size 2 containing x and y co-ordinates of particle location
        goal_loc (list): List of size 2 containing x and y co-ordinates of goal location
        coil_vals (list): List of size 8 containing scaled current values where 0th element corresponds to Northern coil
        coil_locs (list): List of size 8 containing coil locations where 0th element corresponds to Northern coil

    Returns:
        list: List of size 8 containing scaled current values where 0th element corresponds to Northern coil
    """
    # Automatic control PID gains
    k = {"p": 2, "i": 0, "d": 0}

    # ************************** CONTROL THE PARTICLE IN THE North South East West DIRECTIONS **************************
    # Calculates the gains by calculating the distance between particle and goal in the direction
    # connecting the pair of solenoids. This difference is normalized by dividing by the diameter
    # of solenoid circle which is esentially the workspace.
    gainNS = (
        1.0
        * k["p"]
        * ((goal_loc[1] - particle_loc[1]) / (initializations.GUI_SOL_CIRCLE_RAD * 2))
    )
    gainEW = (
        1.0
        * k["p"]
        * ((goal_loc[0] - particle_loc[0]) / (initializations.GUI_SOL_CIRCLE_RAD * 2))
    )

    coil_vals[0], coil_vals[2], coil_vals[4], coil_vals[6] = calc_coil_vals(
        gainNS, gainEW
    )
    # ************************** CONTROL THE PARTICLE IN THE DIAGONAL DIRECTIONS **************************
    # Goal location is tranformed in a new co-ordinate system that is 45 degrees to the orignal one.
    # This simplifies the calculations for difference between current location and goal location.
    goal_loc_rotated = functions.rotate_frame(goal_loc[0], goal_loc[1], math.pi / 4)
    particle_loc_rotated = functions.rotate_frame(
        particle_loc[0], particle_loc[1], math.pi / 4
    )

    # Calculates the gains by calculating the distance between particle and goal in the direction
    # connecting the pair of solenoids. This difference is normalized by dividing by the diameter
    # of solenoid circle which is essentially the workspace.
    gainNeSw = (
        1
        * k["p"]
        * (
            (goal_loc_rotated[1] - particle_loc_rotated[1])
            / (initializations.GUI_SOL_CIRCLE_RAD * 2)
        )
    )
    gainSeNw = (
        1
        * k["p"]
        * (
            (goal_loc_rotated[0] - particle_loc_rotated[0])
            / (initializations.GUI_SOL_CIRCLE_RAD * 2)
        )
    )

    coil_vals[1], coil_vals[3], coil_vals[5], coil_vals[7] = calc_coil_vals(
        gainNeSw, gainSeNw
    )

    return coil_vals


def calc_coil_vals(UDGain, RLGain):
    """Decides which coil to turn on amongst the two pairs of coils

    Args:
        UDGain : Gain in Up and Down direction coils
        RLGain : Gain in Right and Left direction coils

    Returns:
        float: Scaled current value of Up direction coil
        float: Scaled current value of Right direction coil
        float: Scaled current value of Down direction coil
        float: Scaled current value of Left direction coil
    """

    if UDGain > 0:
        value1 = UDGain
        value3 = 0
    else:
        value1 = 0
        value3 = -1 * UDGain

    if RLGain > 0:
        value2 = RLGain
        value4 = 0
    else:
        value2 = 0
        value4 = -1 * RLGain

    return value1, value2, value3, value4
