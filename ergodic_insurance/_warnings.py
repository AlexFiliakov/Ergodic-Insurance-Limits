"""Custom warning classes for the Ergodic Insurance package.

These warning classes allow users to programmatically filter, suppress,
or capture warnings using Python's standard ``warnings`` module.

Example:
    Suppress all configuration warnings in a batch run::

        import warnings
        from ergodic_insurance._warnings import ConfigurationWarning

        warnings.filterwarnings("ignore", category=ConfigurationWarning)

    Capture data-quality warnings during simulation::

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataQualityWarning)
            # ... run simulation ...
            quality_issues = [x for x in w if issubclass(x.category, DataQualityWarning)]
"""


class ErgodicInsuranceWarning(UserWarning):
    """Base class for all ergodic-insurance warnings."""


class ConfigurationWarning(ErgodicInsuranceWarning):
    """Unusual or potentially incorrect configuration parameters.

    Raised during config validation when parameter values fall outside
    typical ranges (e.g., unusually high operating margins, negative
    cash-conversion cycles).
    """


class DataQualityWarning(ErgodicInsuranceWarning):
    """Runtime data-quality observations.

    Raised when simulation or analysis encounters data anomalies such as
    non-finite values, unexpected distributions, or duration mismatches.
    """


class ExportWarning(ErgodicInsuranceWarning):
    """Issues encountered while exporting visualizations.

    Raised when a requested export format is unavailable (e.g., missing
    *kaleido* for static Plotly images) and a fallback is used instead.
    """
