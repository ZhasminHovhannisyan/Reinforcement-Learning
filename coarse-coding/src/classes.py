import numpy as np

class Interval:
    # region Summary
    """
    A wrapper class for an interval
    """
    # endregion Summary

    # region Constructor

    def __init__(self, left, right):
        # region Summary
        """
        Constructor of Interval class
        :param left: Interval's left bound
        :param right: Interval's right bound
        """
        # endregion Summary

        # region Body

        self.left = left
        self.right = right

        # endregion Body

    # endregion Constructor

    # region Functions

    def contain(self, point):
        # region Summary
        """
        Whether a point is in this interval
        :param point: Given point
        :return: True, if point is in this interval; otherwise, False
        """
        # endregion Summary

        # region Body

        return self.left <= point < self.right

        # endregion Body

    def size(self):
        # region Summary
        """
        Length of this interval
        """
        # endregion Summary

        # region Body

        return self.right - self.left

        # endregion Body

    # endregion Functions

# Domain of the square wave is [0; 2)
DOMAIN = Interval(left=0.0, right=2.0)

class ValueFunction:
    # region Summary
    """
    A wrapper class for VF
    """
    # endregion Summary

    # region Constructor

    def __init__(self, feature_width, domain=DOMAIN, step_size=0.2, num_of_features=50):
        # region Summary
        """
        Constructor of ValueFunction class
        :param feature_width: Feature width
        :param domain: Domain of this function, an instance of Interval
        :param step_size: Step-size parameter for 1 update
        :param num_of_features: Number of features
        """
        # endregion Summary

        # region Body

        self.feature_width = feature_width
        self.domain = domain
        self.step_size = step_size
        self.num_of_features = num_of_features

        # Create an empty list for features
        self.features = []

        # There are many ways to place those feature windows, following is just one possible way
        # Calculate step
        step = (domain.size() - feature_width) / (num_of_features - 1)

        # Preserve domain's left bound
        left = domain.left

        # For every feature
        for i in range(num_of_features - 1):
            # append an interval to features
            self.features.append(Interval(left, left + feature_width))

            # add step to the left bound
            left += step

        # Append interval to features
        self.features.append(Interval(left, domain.right))

        # Initialize weight for each feature
        self.weights = np.zeros(num_of_features)

        # endregion Body

    # endregion Constructor

    # region Functions

    def get_active_features(self, point):
        # region Summary
        """
        For given point, return the indices of corresponding feature windows
        :param point: Point
        :return: Feature windows indices
        """
        # endregion Summary

        # region Body

        # Create an empty list for active features
        active_features = []

        # For every feature
        for i in range(len(self.features)):
            # if feature contains point
            if self.features[i].contain(point):
                # append the feature's number to the list of active features
                active_features.append(i)

        return active_features

        # endregion Body

    def value(self, point):
        # region Summary
        """
        Estimate the value for given point.
        :param point: Point
        :return: Point's value estimate
        """
        # endregion Summary

        # region Body

        # Get the active features for given point
        active_features = self.get_active_features(point)

        # Calculate point's value estimate as sum of weights of its active features
        value_estimate = np.sum(self.weights[active_features])

        return value_estimate

        # endregion Body

    def update(self, delta, point):
        # region Summary
        """
        Update weights given sample of point
        :param delta: Update size
        :param point: Point
        """
        # endregion Summary

        # region Body

        # Get the active features for given point
        active_features = self.get_active_features(point)

        # Calculate update size
        delta *= self.step_size / len(active_features)

        # For every active feature
        for index in active_features:
            # update weights
            self.weights[index] += delta

        # endregion Body

    # endregion Functions