#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor

        Args:
            data (list): data used to estimate the distribution
            n (int): number of trials
            p (float): probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")

            if p <= 0 or p >= 1:
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            variance = 0
            for value in data:
                variance += (value - mean) ** 2

            variance /= len(data)

            p = 1 - (variance / mean)

            n = round(mean / p)

            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the PMF for k successes

        Args:
            k (int): number of successes

        Returns:
            float: PMF value
        """
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i

        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i

        nk_fact = 1
        for i in range(1, self.n - k + 1):
            nk_fact *= i

        combination = n_fact / (k_fact * nk_fact)

        return (
            combination *
            (self.p ** k) *
            ((1 - self.p) ** (self.n - k))
        )

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

        if k > self.n:
            k = self.n

        cumulative = 0

        for i in range(k + 1):
            cumulative += self.pmf(i)

        return cumulative
