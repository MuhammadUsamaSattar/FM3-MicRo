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
    angle = math.pi
    min_coil = -1

    for i in range(len(coil_vals)):
        coil_vals[i] = 0
        alpha = abs(
            math.atan2(goal_loc[1] - particle_loc[1], goal_loc[0] - particle_loc[0])
            - math.atan2(
                coil_locs[i][1] - particle_loc[1], coil_locs[i][0] - particle_loc[0]
            )
        )
        if alpha < angle:
            angle = alpha
            min_coil = i

    coil_vals[min_coil] = functions.distance(
        particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
    ) / (initializations.SIM_SOL_CIRCLE_RAD * 2)

    return coil_vals
