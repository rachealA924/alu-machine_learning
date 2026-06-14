#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        Args:
            data (list): data used to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation

        Raises:
            TypeError: if data is not a list
            ValueError: if data has fewer than 2 values
            ValueError: if stddev is not positive
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = 0
            for value in data:
                variance += (value - self.mean) ** 2

            variance /= len(data)

            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of x

        Args:
            x (float): x-value

        Returns:
            float: z-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of z

        Args:
            z (float): z-score

        Returns:
            float: x-value
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the PDF for x

        Args:
            x (float): x-value

        Returns:
            float: PDF value
        """
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)

        exponent = e ** (
            -((x - self.mean) ** 2) /
            (2 * (self.stddev ** 2))
        )

        return coefficient * exponent

    def cdf(self, x):
        """
        Calculates the CDF for x

        Args:
            x (float): x-value

        Returns:
            float: CDF value
        """
        pi = 3.1415926536

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = (
            z
            - (z ** 3) / 3
            + (z ** 5) / 10
            - (z ** 7) / 42
            + (z ** 9) / 216
        )

        return 0.5 * (1 + (2 / (pi ** 0.5)) * erf)
