"""Histogram univariate module."""

import numpy as np
from scipy.stats import rv_histogram
from copulas.univariate.base import BoundedType, ParametricType, ScipyModel

class Histogram(ScipyModel):
    """
    Histogram univariate model using SciPy's rv_histogram.

    This model constructs a histogram from the data using a specified number of bins
    (default is 10) and creates a frozen histogram distribution that provides
    PDF, CDF, inverse CDF (ppf), and sampling methods.
    """
    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.BOUNDED

    def __init__(self, bins=10):
        """
        Initialize the Histogram model with the specified number of bins.

        Args:
            bins (int): Number of bins to use for the histogram. Default is 10.
        """
        super().__init__()
        self.bins = bins

    def _fit(self, X):
        X = np.asarray(X)
        counts, bin_edges = np.histogram(X, bins=self.bins, density=False)
        total = np.sum(counts)
        if total > 0:
            probs = counts / total
        else:
            probs = np.ones_like(counts) / len(counts)
        # Create the frozen rv_histogram instance
        hist = rv_histogram((probs, bin_edges), density=True)
        self._params = {'histogram': hist}
        

    def _fit_constant(self, X):
        constant_value = np.min(X)
        # Create a trivial histogram for constant data with a tiny width.
        bin_edges = np.array([constant_value, constant_value + 1e-6])
        hist = rv_histogram((np.array([1.0]), bin_edges))
        self._params = {'histogram': hist}

    def check_fit(self):
        """
        Check if the model has been fitted.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_params') or 'histogram' not in self._params:
            raise ValueError("This Histogram instance is not fitted yet. Call 'fit' with appropriate data.")

    def cumulative_distribution(self, X):
        """
        Evaluate the cumulative distribution function at each point in X.
        """
        self.check_fit()
        return self._params['histogram'].cdf(X)

    def probability_density(self, X):
        """
        Evaluate the probability density function at each point in X.
        """
        self.check_fit()
        return self._params['histogram'].pdf(X)

    def percent_point(self, U):
        """
        Compute the inverse CDF (percent point function) for probability q.
        """
        self.check_fit()
        return self._params['histogram'].ppf(U)

    def sample(self, n_samples=1, random_state=None):
        """
        Generate samples from the fitted histogram.
        """
        self.check_fit()
        return self._params['histogram'].rvs(size=n_samples, random_state=random_state)

    def _is_constant(self):
        """
        Check if the data used to fit the model is constant.

        Returns:
            bool: True if the data is constant, False otherwise.
        """
        self.check_fit()
        bin_edges = self._params['histogram']._histogram[1]
        return len(bin_edges) == 2 and np.isclose(bin_edges[1] - bin_edges[0], 1e-6)