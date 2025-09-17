from importlib.metadata import distribution

import numpy as np


def calculate_errors(branching_factor):
    # region Summary
    """
    Calculate estimation error
    :param branching_factor: Branching factor (denoted as b)
    :return: Errors in value estimates for given branching factor
    """
    # endregion Summary

    # region Body

    # Set the value distribution of the next b states
    distribution = np.random.randn(branching_factor)

    # Calculate the true value of the current state
    true_value = np.mean(distribution)

    # Create an empty list for samples
    samples = []

    # Create an empty list for errors
    errors = []

    # For every sample update
    for _ in range(2*branching_factor):

        # get the estimated value of the current state
        estimated_value = np.random.choice(distribution)

        # append the estimated value to the list of samples
        samples.append(estimated_value)

        # append the absolute difference of mean of samples and true value to the list of errors
        errors.append(np.abs(np.mean(samples) - true_value))

    return errors

    # endregion Body
