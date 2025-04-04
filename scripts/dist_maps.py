# Define a mapping for marginal distributions to integer enums
marginal_types_map = {
    "Histogram": 0,
    "UniformUnivariate": 1,
    "TruncatedGaussian": 2,
    "GaussianUnivariate": 3,
    "GaussianKDE": 4,
    "BetaUnivariate": 5,
    "GammaUnivariate": 6,
    "LogLaplace": 7,
    "StudentTUnivariate": 8,
}

# Define a mapping for copula types to integer enums
copula_types_map = {
    "GaussianMultivariate": 0,
    "IndependentMultivariate": 1,
    "GaussianMixtureCopula": 2,
}