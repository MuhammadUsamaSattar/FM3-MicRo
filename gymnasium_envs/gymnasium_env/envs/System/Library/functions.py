import math
import time

from gymnasium_env.envs.System import initializations


def rotate_frame(x, y, angle):
    """Returns x and y in a frame roated by angle

    Args:
        x : x co-ordinate of point
        y : y co-ordinate of point
        angle : Angle by which the frame is rotated

    Returns:
        float: x co-ordinate of point in rotated frame
        float: y co-ordinate of point in rotated frame
    """
    x_rotated = (x * math.cos(angle)) - (y * math.sin(angle))
    y_rotated = (x * math.sin(angle)) + (y * math.cos(angle))

    return x_rotated, y_rotated


def absolute_to_image(x, y, frame_width, frame_height):
    """Converts x and y from absolute frame (center of frame is at [0, 0]) to image frame (top-left of frame is [0, 0])

    Args:
        x : x co-ordinate of point
        y : y co-ordinate of point
        frame_width : Width of the frame
        frame_height : Height of the frame

    Returns:
        list: List containing x and y in image frame
    """
    return [int(int(x) + (frame_width / 2)), int((frame_height / 2) - int(y))]


def image_to_absolute(x, y, frame_width, frame_height):
    """Converts x and y from image frame (top-left of frame is at [0, 0]) to absolute frame (center of frame is [0, 0])

    Args:
        x : x co-ordinate of point
        y : y co-ordinate of point
        frame_width : Width of the frame
        frame_height : Height of the frame

    Returns:
        list: List containing x and y in absolute frame
    """
    return [(x - (frame_width / 2)), ((frame_height / 2) - y)]


def current_frac_to_formatted(curr, zero_val, output_range):
    """Converts fractional current value to one that is in a format acceptable by Arduino code

    Args:
        curr : Scaled current value
        zero_val : Value corresponding to zero in format
        output_range : Value corresponding to hundred percent in format

    Returns:
        float: Formatted current value
    """
    return (curr * output_range) + zero_val


def distance(x1, y1, x2, y2):
    """Calculates distance between points

    Args:
        x1 : x co-ordinate of first point
        y1 : y co-ordinate of first point
        x2 : x co-ordinate of second point
        y2 : y co-ordinate of second point

    Returns:
        float: Distance between points
    """
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def limit_coil_vals(coil_vals):
    """Limit the coil values according to initializations.MAX_FRAC_CURR

    Args:
        coil_vals : List of size 8 containing scaled current values where 0th element corresponds to Northern coil

    Returns:
        list: List of size 8 containing scaled current values limited by initializations.MAX_FRAC_CURR where 0th element corresponds to Northern coil
    """
    for i in range(len(coil_vals)):

        if coil_vals[i] > 1:
            coil_vals[i] = 1.0

        elif coil_vals[i] < -1:
            coil_vals[i] = -1.0

        coil_vals[i] = coil_vals[i] * initializations.MAX_FRAC_CURR

    return coil_vals

def sleep(duration, get_now=time.perf_counter):
    now = get_now()
    end = now + duration
    while now < end:
        now = get_now()
