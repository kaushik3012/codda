from copulas.multivariate import GaussianMultivariate
from copulas.univariate import TruncatedGaussian, UniformUnivariate, BetaUnivariate, GaussianKDE,GaussianUnivariate, GammaUnivariate, LogLaplace, StudentTUnivariate
from ..custom_univariates import Histogram

file_paths = {
    "P": "Isabel_data_all_variables_vti/Pf25.binLE.raw_corrected_2_subsampled.vti",
    "TC": "Isabel_data_all_variables_vti/TCf25.binLE.raw_corrected_2_subsampled.vti",    # replace with your filename
    "Velocity": "Isabel_data_all_variables_vti/Velocityf25.binLE.raw_corrected_2_subsampled.vti",          
    "CLOUD": "Isabel_data_all_variables_vti/CLOUDf25.binLE.raw_corrected_2_subsampled.vti",
    "PRECIP": "Isabel_data_all_variables_vti/PRECIPf25.binLE.raw_corrected_2_subsampled.vti",
    "QCLOUD": "Isabel_data_all_variables_vti/QCLOUDf25.binLE.raw_corrected_2_subsampled.vti",
    "QGRAUP": "Isabel_data_all_variables_vti/QGRAUPf25.binLE.raw_corrected_2_subsampled.vti",
    "QICE": "Isabel_data_all_variables_vti/QICEf25.binLE.raw_corrected_2_subsampled.vti",
    "QRAIN": "Isabel_data_all_variables_vti/QRAINf25.binLE.raw_corrected_2_subsampled.vti",
    "QSNOW": "Isabel_data_all_variables_vti/QSNOWf25.binLE.raw_corrected_2_subsampled.vti",
    "QVAPOR": "Isabel_data_all_variables_vti/QVAPORf25.binLE.raw_corrected_2_subsampled.vti",
    "U": "Isabel_data_all_variables_vti/Uf25.binLE.raw_corrected_2_subsampled.vti",
    "V": "Isabel_data_all_variables_vti/Vf25.binLE.raw_corrected_2_subsampled.vti",
    "W": "Isabel_data_all_variables_vti/Wf25.binLE.raw_corrected_2_subsampled.vti",
}


# (B) Specify marginal distributions for each variable.
# You can change these to other univariate distributions if desired.
marginal_distributions = {
    "P": StudentTUnivariate,  
    "TC": StudentTUnivariate,
    "Velocity": StudentTUnivariate,
    "CLOUD": StudentTUnivariate,
    "PRECIP": StudentTUnivariate,
    "QCLOUD": StudentTUnivariate,
    "QGRAUP": StudentTUnivariate,
    "QICE": StudentTUnivariate,
    "QRAIN": StudentTUnivariate,
    "QSNOW": StudentTUnivariate,
    "QVAPOR": StudentTUnivariate,
    "U": StudentTUnivariate,
    "V": StudentTUnivariate,
    "W": StudentTUnivariate,
    "x": UniformUnivariate,
    "y": UniformUnivariate,
    "z": UniformUnivariate
}

block_size = 10

copula_type = "GaussianMultivariate"  # or "IndependentMultivariate"