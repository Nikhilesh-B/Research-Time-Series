"""Forecaster implementations."""

from benchmark2.forecasters.base import Forecaster
from benchmark2.forecasters.mean import MeanForecaster
from benchmark2.forecasters.arima import ARIMAForecaster
from benchmark2.forecasters.bayesian_ar import BayesianARForecaster
from benchmark2.forecasters.ssa import SSAForecaster
from benchmark2.forecasters.timesfm_forecaster import TimesFMForecaster

__all__ = [
    "Forecaster",
    "MeanForecaster",
    "ARIMAForecaster",
    "BayesianARForecaster",
    "SSAForecaster",
    "TimesFMForecaster",
]
