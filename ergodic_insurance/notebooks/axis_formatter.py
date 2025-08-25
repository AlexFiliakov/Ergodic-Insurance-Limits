"""Utility functions for formatting axis labels with K/M abbreviations."""


def format_axis_labels(value, pos=None):
    """Format axis labels with K for thousands and M for millions.

    Args:
        value: The numeric value to format
        pos: Position (unused, for matplotlib compatibility)

    Returns:
        Formatted string with K/M abbreviations
    """
    if abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.0f}K"
    else:
        return f"${value:.0f}"


def get_plotly_tickformat(max_value):
    """Get appropriate Plotly tickformat string based on data range.

    Args:
        max_value: Maximum value in the data

    Returns:
        Plotly tickformat string
    """
    if max_value >= 1e6:
        return "$.1s"  # Will show as 1M, 2M, etc.
    elif max_value >= 1e3:
        return "$.0s"  # Will show as 1K, 2K, etc.
    else:
        return "$,.0f"  # Standard format for small numbers


def format_plotly_axis(fig, row, col, axis_type="y", data_range=None):
    """Apply K/M formatting to a specific Plotly axis.

    Args:
        fig: Plotly figure object
        row: Row number of the subplot
        col: Column number of the subplot
        axis_type: 'x' or 'y' axis
        data_range: Optional tuple of (min, max) values
    """
    if data_range:
        max_val = max(abs(data_range[0]), abs(data_range[1]))
        tickformat = get_plotly_tickformat(max_val)

        if axis_type == "y":
            fig.update_yaxes(tickformat=tickformat, row=row, col=col)
        else:
            fig.update_xaxes(tickformat=tickformat, row=row, col=col)


# Custom tick text generator for more control
def generate_tick_text(values):
    """Generate custom tick text with K/M abbreviations.

    Args:
        values: List of numeric values

    Returns:
        List of formatted strings
    """
    tick_text = []
    for val in values:
        if abs(val) >= 1e6:
            tick_text.append(f"${val/1e6:.1f}M")
        elif abs(val) >= 1e3:
            tick_text.append(f"${val/1e3:.0f}K")
        else:
            tick_text.append(f"${val:.0f}")
    return tick_text
