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
    min_coil = []
    delta_angles = []

    for i in range(len(coil_vals)):
        coil_vals[i] = 0
        alpha = calc_angle_between_angles(
            calc_angle_between_points(particle_loc, goal_loc),
            calc_angle_between_points(particle_loc, coil_locs[i]),
        )

        delta_angles.append(alpha)

        if alpha < math.pi / 2:
            min_coil.append(i)

    for i in min_coil:
        coil_vals[i] = (
            functions.distance(
                particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
            )
            * math.cos(delta_angles[i])
            / (initializations.SIM_SOL_CIRCLE_RAD * 2)
        )

    return coil_vals


def calc_angle_between_points(point1, point2):
    angle = math.atan2(point2[1] - point1[1], point2[0] - point1[0])

    return angle


def calc_angle_between_angles(angle1, angle2):
    angle = abs(angle1 - angle2)

    if angle > math.pi:
        angle = 2 * math.pi - angle

    return angle
