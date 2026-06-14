#!/usr/bin/env python3
"""Poisson distribution module"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Args:
            data (list): data used to estimate the distribution
            lambtha (float): expected number of occurrences

        Raises:
            TypeError: if data is not a list
            ValueError: if data has fewer than 2 values
            ValueError: if lambtha is not positive
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for k successes

        Args:
            k (int): number of successes

        Returns:
            float: PMF value
        """
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return ((e ** (-self.lambtha)) *
                (self.lambtha ** k) / factorial)

    def cdf(self, k):
        """
        Calculates the CDF for k successes

        Args:
            k (int): number of successes

        Returns:
            float: CDF value
        """
        k = int(k)

        if k < 0:
            return 0

        cumulative = 0

        for i in range(k + 1):
            cumulative += self.pmf(i)

        return cumulative
    