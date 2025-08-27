"""Figure factory for creating standardized plots with consistent styling.

This module provides a factory class for creating various types of plots
with automatic styling, spacing, and formatting applied consistently.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .style_manager import StyleManager, Theme


class FigureFactory:
    """Factory for creating standardized figures with consistent styling.

    This class provides methods to create various types of plots with
    automatic application of themes, consistent formatting, and proper spacing.

    Example:
        >>> factory = FigureFactory(theme=Theme.PRESENTATION)
        >>> fig, ax = factory.create_line_plot(
        ...     x_data=[1, 2, 3, 4],
        ...     y_data=[10, 20, 15, 25],
        ...     title="Revenue Growth",
        ...     x_label="Quarter",
        ...     y_label="Revenue ($M)"
        ... )

        >>> # Create multiple subplots
        >>> fig, axes = factory.create_subplots(
        ...     rows=2, cols=2,
        ...     size_type="large",
        ...     subplot_titles=["Q1", "Q2", "Q3", "Q4"]
        ... )
    """

    def __init__(
        self,
        style_manager: Optional[StyleManager] = None,
        theme: Theme = Theme.DEFAULT,
        auto_apply: bool = True,
    ):
        """Initialize figure factory.

        Args:
            style_manager: Custom style manager (creates default if None)
            theme: Theme to use for all figures
            auto_apply: Whether to automatically apply styling
        """
        self.style_manager = style_manager or StyleManager(theme=theme)
        self.auto_apply = auto_apply

        if self.auto_apply:
            self.style_manager.apply_style()

    def create_figure(
        self,
        size_type: str = "medium",
        orientation: str = "landscape",
        dpi_type: str = "screen",
        title: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Create a basic figure with styling applied.

        Args:
            size_type: Size preset (small, medium, large, blog, technical, presentation)
            orientation: Figure orientation (landscape or portrait)
            dpi_type: DPI type (screen, web, print)
            title: Optional figure title

        Returns:
            Tuple of (figure, axes)
        """
        size = self.style_manager.get_figure_size(size_type, orientation)
        dpi = self.style_manager.get_dpi(dpi_type)

        fig, ax = plt.subplots(figsize=size, dpi=dpi)

        if title:
            fig.suptitle(title, fontweight="bold")

        self._apply_axis_styling(ax)
        return fig, ax

    def create_subplots(
        self,
        rows: int = 1,
        cols: int = 1,
        size_type: str = "large",
        dpi_type: str = "screen",
        title: Optional[str] = None,
        subplot_titles: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """Create subplots with consistent styling.

        Args:
            rows: Number of subplot rows
            cols: Number of subplot columns
            size_type: Size preset
            dpi_type: DPI type
            title: Main figure title
            subplot_titles: Titles for each subplot
            **kwargs: Additional arguments for plt.subplots

        Returns:
            Tuple of (figure, axes array)
        """
        size = self.style_manager.get_figure_size(size_type)
        dpi = self.style_manager.get_dpi(dpi_type)

        fig, axes = plt.subplots(rows, cols, figsize=size, dpi=dpi, **kwargs)

        if title:
            fig.suptitle(
                title, fontweight="bold", fontsize=self.style_manager.get_fonts().size_title + 2
            )

        # Apply styling to all axes
        axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, ax in enumerate(axes_list):
            self._apply_axis_styling(ax)
            if subplot_titles and i < len(subplot_titles):
                ax.set_title(subplot_titles[i])

        plt.tight_layout()
        return fig, axes

    def create_line_plot(
        self,
        x_data: Union[List, np.ndarray, pd.Series],
        y_data: Union[List, np.ndarray, pd.Series, Dict[str, Union[List, np.ndarray]]],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        size_type: str = "medium",
        dpi_type: str = "screen",
        show_legend: bool = True,
        show_grid: bool = True,
        markers: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Create a line plot with automatic formatting.

        Args:
            x_data: X-axis data
            y_data: Y-axis data (can be multiple series as dict)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            labels: Series labels for legend
            size_type: Figure size preset
            dpi_type: DPI type
            show_legend: Whether to show legend
            show_grid: Whether to show grid
            markers: Whether to add markers to lines
            **kwargs: Additional arguments for plot

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(size_type=size_type, dpi_type=dpi_type, title=title)

        colors = self.style_manager.get_colors()

        # Handle multiple series
        if isinstance(y_data, dict):
            for i, (label, data) in enumerate(y_data.items()):
                color = colors.series[i % len(colors.series)]
                marker = "o" if markers else None
                ax.plot(x_data, data, label=label, color=color, marker=marker, **kwargs)
        else:
            # Single series
            marker = "o" if markers else None
            label = labels[0] if labels else None
            ax.plot(x_data, y_data, color=colors.primary, label=label, marker=marker, **kwargs)

        # Labels and formatting
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        if show_legend and (isinstance(y_data, dict) or labels):
            ax.legend(loc="best", frameon=True)

        ax.grid(show_grid, alpha=self.style_manager.get_grid_config().grid_alpha)

        plt.tight_layout()
        return fig, ax

    def create_bar_plot(
        self,
        categories: Union[List, np.ndarray],
        values: Union[List, np.ndarray, Dict[str, Union[List, np.ndarray]]],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        size_type: str = "medium",
        dpi_type: str = "screen",
        orientation: str = "vertical",
        show_values: bool = False,
        value_format: str = ".1f",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Create a bar plot with automatic formatting.

        Args:
            categories: Category labels
            values: Values to plot (can be multiple series as dict)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            labels: Series labels for legend
            size_type: Figure size preset
            dpi_type: DPI type
            orientation: Bar orientation (vertical or horizontal)
            show_values: Whether to show value labels on bars
            value_format: Format string for value labels
            **kwargs: Additional arguments for bar plot

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(size_type=size_type, dpi_type=dpi_type, title=title)

        colors = self.style_manager.get_colors()

        # Handle multiple series
        if isinstance(values, dict):
            n_series = len(values)
            width = 0.8 / n_series
            x_pos = np.arange(len(categories))

            for i, (label, data) in enumerate(values.items()):
                offset = (i - n_series / 2 + 0.5) * width
                color = colors.series[i % len(colors.series)]

                if orientation == "vertical":
                    bars = ax.bar(x_pos + offset, data, width, label=label, color=color, **kwargs)
                else:
                    bars = ax.barh(x_pos + offset, data, width, label=label, color=color, **kwargs)

                if show_values:
                    self._add_value_labels(ax, bars, orientation, value_format)

            if orientation == "vertical":
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories)
            else:
                ax.set_yticks(x_pos)
                ax.set_yticklabels(categories)

            ax.legend(loc="best", frameon=True)
        else:
            # Single series
            if orientation == "vertical":
                bars = ax.bar(categories, values, color=colors.primary, **kwargs)
            else:
                bars = ax.barh(categories, values, color=colors.primary, **kwargs)

            if show_values:
                self._add_value_labels(ax, bars, orientation, value_format)

        # Labels
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.grid(
            True,
            alpha=self.style_manager.get_grid_config().grid_alpha,
            axis="y" if orientation == "vertical" else "x",
        )

        plt.tight_layout()
        return fig, ax

    def create_scatter_plot(
        self,
        x_data: Union[List, np.ndarray],
        y_data: Union[List, np.ndarray],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size_type: str = "medium",
        dpi_type: str = "screen",
        colors: Optional[Union[List, np.ndarray]] = None,
        sizes: Optional[Union[List, np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        show_colorbar: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Create a scatter plot with automatic formatting.

        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            size_type: Figure size preset
            dpi_type: DPI type
            colors: Optional colors for points (for continuous coloring)
            sizes: Optional sizes for points
            labels: Optional labels for points
            show_colorbar: Whether to show colorbar when colors provided
            **kwargs: Additional arguments for scatter

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(size_type=size_type, dpi_type=dpi_type, title=title)

        theme_colors = self.style_manager.get_colors()

        # Default sizes if not provided
        if sizes is None:
            sizes = 50

        # Create scatter plot
        if colors is not None:
            scatter = ax.scatter(x_data, y_data, c=colors, s=sizes, cmap="viridis", **kwargs)
            if show_colorbar:
                plt.colorbar(scatter, ax=ax)
        else:
            scatter = ax.scatter(x_data, y_data, s=sizes, color=theme_colors.primary, **kwargs)

        # Labels
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.grid(True, alpha=self.style_manager.get_grid_config().grid_alpha)

        plt.tight_layout()
        return fig, ax

    def create_histogram(
        self,
        data: Union[List, np.ndarray, pd.Series],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: str = "Frequency",
        bins: Union[int, str] = "auto",
        size_type: str = "medium",
        dpi_type: str = "screen",
        show_statistics: bool = False,
        show_kde: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Create a histogram with automatic formatting.

        Args:
            data: Data to plot
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            bins: Number of bins or method
            size_type: Figure size preset
            dpi_type: DPI type
            show_statistics: Whether to show mean/median lines
            show_kde: Whether to overlay KDE
            **kwargs: Additional arguments for hist

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(size_type=size_type, dpi_type=dpi_type, title=title)

        colors = self.style_manager.get_colors()

        # Create histogram
        n, bins_out, patches = ax.hist(
            data,
            bins=bins,
            color=colors.primary,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            **kwargs,
        )

        # Add statistics if requested
        if show_statistics:
            mean_val = np.mean(data)
            median_val = np.median(data)

            ax.axvline(
                mean_val,
                color=colors.warning,
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}",
            )
            ax.axvline(
                median_val,
                color=colors.success,
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.2f}",
            )
            ax.legend()

        # Add KDE if requested
        if show_kde:
            from scipy import stats

            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            kde_values = kde(x_range)

            # Scale KDE to match histogram
            ax2 = ax.twinx()
            ax2.plot(x_range, kde_values, color=colors.secondary, linewidth=2, label="KDE")
            ax2.set_ylabel("Density")
            ax2.tick_params(axis="y", labelcolor=colors.secondary)

        # Labels
        if x_label:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.grid(True, alpha=self.style_manager.get_grid_config().grid_alpha, axis="y")

        plt.tight_layout()
        return fig, ax

    def create_heatmap(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        title: Optional[str] = None,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size_type: str = "medium",
        dpi_type: str = "screen",
        cmap: str = "RdBu_r",
        show_values: bool = True,
        value_format: str = ".2f",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Create a heatmap with automatic formatting.

        Args:
            data: 2D data array or DataFrame
            title: Plot title
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            x_label: X-axis title
            y_label: Y-axis title
            size_type: Figure size preset
            dpi_type: DPI type
            cmap: Colormap name
            show_values: Whether to show values in cells
            value_format: Format string for cell values
            **kwargs: Additional arguments for imshow

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(size_type=size_type, dpi_type=dpi_type, title=title)

        # Handle DataFrame
        if isinstance(data, pd.DataFrame):
            if x_labels is None:
                x_labels = list(data.columns)
            if y_labels is None:
                y_labels = list(data.index)
            data = data.values

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect="auto", **kwargs)
        plt.colorbar(im, ax=ax)

        # Set tick labels
        if x_labels:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        if y_labels:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)

        # Add values if requested
        if show_values:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        f"{data[i, j]:{value_format}}",
                        ha="center",
                        va="center",
                        color="black",
                    )

        # Labels
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        plt.tight_layout()
        return fig, ax

    def create_box_plot(
        self,
        data: Union[List[List], Dict[str, List], pd.DataFrame],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        size_type: str = "medium",
        dpi_type: str = "screen",
        orientation: str = "vertical",
        show_means: bool = True,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Create a box plot with automatic formatting.

        Args:
            data: Data for box plot (list of lists, dict, or DataFrame)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            labels: Labels for each box
            size_type: Figure size preset
            dpi_type: DPI type
            orientation: Plot orientation (vertical or horizontal)
            show_means: Whether to show mean markers
            **kwargs: Additional arguments for boxplot

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(size_type=size_type, dpi_type=dpi_type, title=title)

        colors = self.style_manager.get_colors()

        # Prepare data
        if isinstance(data, dict):
            plot_data = list(data.values())
            if labels is None:
                labels = list(data.keys())
        elif isinstance(data, pd.DataFrame):
            plot_data = [data[col].dropna() for col in data.columns]
            if labels is None:
                labels = list(data.columns)
        else:
            plot_data = data

        # Create box plot (using updated matplotlib API)
        vert = orientation == "vertical"
        bp = ax.boxplot(
            plot_data,
            vert=vert if hasattr(ax.boxplot, "__wrapped__") else None,  # Support old API
            tick_labels=labels,  # Use tick_labels instead of labels
            showmeans=show_means,
            patch_artist=True,
            **kwargs,
        )

        # Style the boxes
        for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
            box.set_facecolor(colors.series[i % len(colors.series)])
            box.set_alpha(0.7)
            median.set_color(colors.text)
            median.set_linewidth(2)

        # Style whiskers and caps
        for whisker in bp["whiskers"]:
            whisker.set_color(colors.neutral)
            whisker.set_linewidth(1)
        for cap in bp["caps"]:
            cap.set_color(colors.neutral)
            cap.set_linewidth(1)

        # Style outliers
        for flier in bp["fliers"]:
            flier.set_marker("o")
            flier.set_markersize(4)
            flier.set_markeredgecolor(colors.warning)
            flier.set_markerfacecolor(colors.warning)
            flier.set_alpha(0.5)

        # Style means if shown
        if show_means and "means" in bp:
            for mean in bp["means"]:
                mean.set_marker("D")
                mean.set_markersize(6)
                mean.set_markeredgecolor(colors.success)
                mean.set_markerfacecolor(colors.success)

        # Labels
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.grid(
            True, alpha=self.style_manager.get_grid_config().grid_alpha, axis="y" if vert else "x"
        )

        plt.tight_layout()
        return fig, ax

    def format_axis_currency(
        self,
        ax: Axes,
        axis: str = "y",
        abbreviate: bool = True,
        decimals: int = 0,
    ) -> None:
        """Format axis labels as currency.

        Args:
            ax: Matplotlib axes
            axis: Which axis to format (x or y)
            abbreviate: Whether to abbreviate large numbers
            decimals: Number of decimal places
        """

        def currency_formatter(x, pos):
            if abbreviate:
                if abs(x) >= 1e9:
                    return f"${x/1e9:.{decimals}f}B"
                if abs(x) >= 1e6:
                    return f"${x/1e6:.{decimals}f}M"
                if abs(x) >= 1e3:
                    return f"${x/1e3:.{decimals}f}K"
            return f"${x:,.{decimals}f}"

        formatter = mticker.FuncFormatter(currency_formatter)
        if axis == "y":
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_formatter(formatter)

    def format_axis_percentage(
        self,
        ax: Axes,
        axis: str = "y",
        decimals: int = 0,
    ) -> None:
        """Format axis labels as percentages.

        Args:
            ax: Matplotlib axes
            axis: Which axis to format (x or y)
            decimals: Number of decimal places
        """

        def percentage_formatter(x, pos):
            return f"{x*100:.{decimals}f}%"

        formatter = mticker.FuncFormatter(percentage_formatter)
        if axis == "y":
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_formatter(formatter)

    def add_annotations(
        self,
        ax: Axes,
        x: float,
        y: float,
        text: str,
        arrow: bool = True,
        offset: Tuple[float, float] = (10, 10),
        **kwargs,
    ) -> None:
        """Add styled annotation to plot.

        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
            text: Annotation text
            arrow: Whether to show arrow
            offset: Text offset from point
            **kwargs: Additional arguments for annotate
        """
        colors = self.style_manager.get_colors()

        if arrow:
            ax.annotate(
                text,
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->", color=colors.neutral, connectionstyle="arc3,rad=0.2"
                ),
                fontsize=self.style_manager.get_fonts().size_base - 1,
                color=colors.text,
                **kwargs,
            )
        else:
            ax.text(
                x,
                y,
                text,
                fontsize=self.style_manager.get_fonts().size_base - 1,
                color=colors.text,
                **kwargs,
            )

    def save_figure(
        self,
        fig: Figure,
        filename: str,
        output_type: str = "web",
        **kwargs,
    ) -> None:
        """Save figure with appropriate DPI settings.

        Args:
            fig: Figure to save
            filename: Output filename
            output_type: Output type (screen, web, print)
            **kwargs: Additional arguments for savefig
        """
        dpi = self.style_manager.get_dpi(output_type)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", **kwargs)

    def _apply_axis_styling(self, ax: Axes) -> None:
        """Apply consistent styling to axes.

        Args:
            ax: Axes to style
        """
        colors = self.style_manager.get_colors()
        grid_config = self.style_manager.get_grid_config()

        # Apply grid settings
        ax.grid(
            grid_config.show_grid,
            alpha=grid_config.grid_alpha,
            linewidth=grid_config.grid_linewidth,
            color=colors.grid,
        )

        # Apply spine visibility
        ax.spines["top"].set_visible(grid_config.spine_top)
        ax.spines["right"].set_visible(grid_config.spine_right)
        ax.spines["bottom"].set_visible(grid_config.spine_bottom)
        ax.spines["left"].set_visible(grid_config.spine_left)

        # Apply spine colors and widths
        for spine in ax.spines.values():
            spine.set_edgecolor(colors.neutral)
            spine.set_linewidth(grid_config.spine_linewidth)

        # Apply tick parameters
        ax.tick_params(
            axis="both",
            which="major",
            width=grid_config.tick_major_width,
            length=5,
            color=colors.neutral,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=grid_config.tick_minor_width,
            length=3,
            color=colors.neutral,
        )

    def _add_value_labels(
        self,
        ax: Axes,
        bars: Any,
        orientation: str,
        format_str: str,
    ) -> None:
        """Add value labels to bars.

        Args:
            ax: Axes containing bars
            bars: Bar container
            orientation: Bar orientation
            format_str: Format string for values
        """
        for bar in bars:
            if orientation == "vertical":
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:{format_str}}",
                    ha="center",
                    va="bottom",
                    fontsize=self.style_manager.get_fonts().size_base - 2,
                )
            else:
                width = bar.get_width()
                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{width:{format_str}}",
                    ha="left",
                    va="center",
                    fontsize=self.style_manager.get_fonts().size_base - 2,
                )
