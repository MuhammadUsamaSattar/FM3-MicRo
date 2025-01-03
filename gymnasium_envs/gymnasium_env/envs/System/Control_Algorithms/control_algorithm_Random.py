import numpy as np


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

    for i in range(len(coil_vals)):
        coil_vals[i] = np.random.normal(loc=0.2, scale=0.1)

    return coil_vals
