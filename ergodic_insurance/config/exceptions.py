"""Custom exceptions for configuration validation.

Since:
    Version 0.14.0 (Issue #1299)
"""


class ConfigurationError(Exception):
    """Raised when configuration validation finds critical issues.

    This exception is raised by :meth:`Config.validate` when the configuration
    contains critical issues that would lead to incorrect simulation results
    or runtime failures.

    Attributes:
        issues: List of specific configuration problems found.

    Examples:
        Catching and inspecting issues::

            try:
                config.validate()
            except ConfigurationError as e:
                for issue in e.issues:
                    print(f"  - {issue}")

    Since:
        Version 0.14.0 (Issue #1299)
    """

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        bullet_list = "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(
            f"Configuration has {len(issues)} critical "
            f"{'issue' if len(issues) == 1 else 'issues'}:\n{bullet_list}"
        )
