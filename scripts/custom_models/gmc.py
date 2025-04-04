"""
Gaussian Mixture Copula

This class models the dependency structure using a Gaussian mixture in the latent normal space.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from copulas.multivariate.base import Multivariate
from copulas.univariate import Univariate, GaussianUnivariate
from copulas.utils import EPSILON, check_valid_values, get_instance, validate_random_state

LOGGER = logging.getLogger(__name__)
DEFAULT_DISTRIBUTION = Univariate


class GaussianMixtureCopula(Multivariate):
    """
    Gaussian Mixture Copula models the joint distribution as a mixture of Gaussians in the
    latent normal space.

    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal distributions,
            or a dictionary mapping column names to the distribution names.
        random_state (int, np.random.RandomState, or None):
            Seed or RandomState for the random generator.
    """

    def __init__(self, distribution=DEFAULT_DISTRIBUTION, n_components=2, random_state=None):
        super().__init__(random_state=random_state)
        self.distribution = distribution
        self.n_components = n_components
        self.columns = None
        self.univariates = None
        self.gmm = None
        self.fitted = False

    def _transform_to_normal(self, X):
        """
        Transform the input data to the latent normal space.

        Each column is transformed by computing the cdf (using its fitted univariate) and
        then applying the inverse standard normal transformation.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(np.array(X).shape) == 1:
                X = [X]
            X = pd.DataFrame(X, columns=self.columns)

        U = []
        for column_name, univariate in zip(self.columns, self.univariates):
            if column_name in X:
                column = X[column_name]
                # Clip for numerical stability
                U.append(univariate.cdf(column.to_numpy()).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    def _fit_columns(self, X):
        """
        Fit each column to its univariate distribution.
        """
        columns = []
        univariates = []
        for column_name, column in X.items():
            distribution = self._get_distribution_for_column(column_name)
            LOGGER.debug('Fitting column %s to %s', column_name, distribution)
            univariate = self._fit_column(column, distribution, column_name)
            columns.append(column_name)
            univariates.append(univariate)
        return columns, univariates

    def _get_distribution_for_column(self, column_name):
        if isinstance(self.distribution, dict):
            return self.distribution.get(column_name, DEFAULT_DISTRIBUTION)
        return self.distribution

    def _fit_column(self, column, distribution, column_name):
        univariate = get_instance(distribution)
        try:
            univariate.fit(column)
        except Exception as error:
            LOGGER.info(
                'Unable to fit column %s with %s; Falling back to Gaussian distribution.',
                column_name,
                distribution,
            )
            univariate = GaussianUnivariate()
            univariate.fit(column)
        return univariate

    @check_valid_values
    def fit(self, X):
        """
        Fit the marginals and then fit the Gaussian mixture.
        """
        LOGGER.info('Fitting %s', self.__class__.__name__)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns, self.univariates = self._fit_columns(X)
        X_normal = self._transform_to_normal(X)

        LOGGER.debug('Fitting Gaussian Mixture with %s components.', self.n_components)
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.gmm.fit(X_normal)
        self.fitted = True
        LOGGER.debug('%s fitted successfully', self.__class__.__name__)

    def probability_density(self, X):
        """
        Compute the probability density for each point in X.
        """
        self.check_fit()
        X_normal = self._transform_to_normal(X)
        # GaussianMixture.score_samples returns log probabilities.
        log_pdf = self.gmm.score_samples(X_normal)
        return np.exp(log_pdf)

    def cumulative_distribution(self, X):
        """
        Compute the cumulative distribution for each point in X.

        A closed-form CDF is not available for a Gaussian mixture.
        """
        raise NotImplementedError("CDF is not implemented for GaussianMixtureCopula.")

    def sample(self, num_rows=1, conditions=None):
        """
        Sample values from the copula.

        If conditions are provided, they are ignored in this implementation.
        """
        self.check_fit()
        if conditions is not None:
            LOGGER.warning('Conditional sampling is not supported for GaussianMixtureCopula. Ignoring conditions.')

        # Sample points from the fitted Gaussian mixture.
        samples, _ = self.gmm.sample(num_rows)
        output = {}
        # Transform the latent samples back using the univariates percent point function.
        for idx, column_name in enumerate(self.columns):
            cdf_values = stats.norm.cdf(samples[:, idx])
            output[column_name] = self.univariates[idx].percent_point(cdf_values)

        return pd.DataFrame(output)

    def to_dict(self):
        """
        Serialize the copula into a dictionary.
        """
        self.check_fit()
        univariates_dicts = [univariate.to_dict() for univariate in self.univariates]
        return {
            'gmm_params': {
                'weights': self.gmm.weights_.tolist(),
                'means': self.gmm.means_.tolist(),
                'covariances': self.gmm.covariances_.tolist(),
                'n_components': self.n_components,
            },
            'columns': self.columns,
            'univariates': univariates_dicts,
            'type': f'{self.__class__.__module__}.{self.__class__.__name__}',
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """
        Create a new instance from a dictionary.
        """
        instance = cls(
            n_components=copula_dict['gmm_params']['n_components'],
            random_state=None
        )
        instance.columns = copula_dict['columns']
        instance.univariates = [Univariate.from_dict(u_dict) for u_dict in copula_dict['univariates']]
        # Recreate the GaussianMixture instance (values are set but re-fitting is needed for full compatibility)
        instance.gmm = GaussianMixture(n_components=instance.n_components)
        instance.gmm.weights_ = np.array(copula_dict['gmm_params']['weights'])
        instance.gmm.means_ = np.array(copula_dict['gmm_params']['means'])
        instance.gmm.covariances_ = np.array(copula_dict['gmm_params']['covariances'])
        instance.fitted = True
        return instance
    
    def bic(self, X):
        """
        Compute the Bayesian Information Criterion (BIC) for the fitted Gaussian mixture model.

        Args:
            X (pd.DataFrame): The data used to fit the model.

        Returns:
            float: The BIC value.
        """
        self.check_fit()
        if not self.fitted:
            raise ValueError("The model must be fitted before calculating BIC.")
        
        # Transform the data to the latent normal space
        X_normal = self._transform_to_normal(X)
        
        # Compute the log likelihood
        log_likelihood = self.gmm.score(X_normal)
        
        # Number of parameters in the model
        n_params = self.n_components * (X.shape[1] + 1) + X.shape[1] * (X.shape[1] + 1) / 2
        
        # Number of observations
        n_samples = X.shape[0]
        
        # Compute BIC
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return bic