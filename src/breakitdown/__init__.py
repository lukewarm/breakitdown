"""breakitdown: A/B test analysis on Spark.

    from breakitdown import Experiment, MetricTable, Count, Binary, Ratio, Quantile

    exp = Experiment(control="control", treatments=["t1"], start="2024-03-01", end="2024-03-14",
                     user_segments=["os"])
    results = exp.analyze(MetricTable("db.app_events", [Count("clicks"), Binary("purchases")]))
    results.overall()

See README.md for the design, the input contract and the result columns.
"""

from ._common import NULL_LABEL, OVERALL
from .columns import Columns
from .experiment import Experiment
from .metrics import Binary, Count, Direction, MeanRatio, Metric, Quantile, QuantileMethod, Ratio
from .results import ExperimentResults
from .tables import FlightMapping, MetricTable

__version__ = "0.3.1"

__all__ = [
    "Experiment",
    "MetricTable",
    "FlightMapping",
    "Columns",
    "Count",
    "Binary",
    "Ratio",
    "MeanRatio",
    "Quantile",
    "Metric",
    "Direction",
    "QuantileMethod",
    "ExperimentResults",
    "OVERALL",
    "NULL_LABEL",
]
