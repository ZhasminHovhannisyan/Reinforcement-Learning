import numpy as np
from src.classes import DOMAIN

def square_wave(point):
    # region Summary
    """
    Square wave function
    :param point: Point from interval
    :return: 1, if point is in (0.5; 1.5); otherwise, 0
    """
    # endregion Summary

    # region Body

    if 0.5 < point < 1.5:
        return 1

    return 0

    # endregion Body

def sample(samples_number):
    # region Summary
    """
    Get samples randomly from the square wave
    :param samples_number: Number of samples to get
    :return: Samples
    """
    # endregion Summary

    # region Body

    # Create an empty list for samples
    samples = []

    # For every sample
    for i in range(samples_number):
        # draw a sample from uniform distribution
        x = np.random.uniform(DOMAIN.left, DOMAIN.right)

        # get square wave function
        y = square_wave(x)

        # append to the list of samples
        samples.append([x, y])

    return samples

    # endregion Body

def approximate(samples, value_function):
    # region Summary
    """
    Train VF with a set of samples.
    :param samples: Samples
    :param value_function: VF
    """
    # endregion Summary

    # region Body

    # For every sample
    for x, y in samples:
        # calculate update size
        delta = y - value_function.value(x)

        # update VF
        value_function.update(delta, x)

    # endregion Body
