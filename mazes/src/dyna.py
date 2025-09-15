class DynaParams:
    # region Summary
    """
    A wrapper class for parameters of Dyna algorithms
    """
    # endregion Summary

    # region Constructor

    def __init__(self):
        # Discount rate (denoted as 𝛾)
        self.discount = 0.95

        # Exploration probability (denoted as 𝜀)
        self.exploration_probability = 0.1

        # Step-size parameter (denoted as 𝛼)
        self.step_size = 0.1

        # Weight for elapsed time (denoted as κ (kappa))
        self.time_weight = 0

        # Number of planning steps (n-step planning)
        self.planning_steps = 5

        # Average over several independent runs
        self.runs = 10

        # Algorithm names
        self.methods = ["Dyna-Q", "Dyna-Q+"]

        # Threshold for priority queue (denoted as θ)
        self.threshold = 0

    # endregion Constructor
