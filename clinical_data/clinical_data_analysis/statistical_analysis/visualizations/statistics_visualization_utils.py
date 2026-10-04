"""
Reusable plotting utilities for statistical visualizations (cognitive, MOCA, etc.).

All functions are dataframe-agnostic: you pass the dataframe, column names,
feature lists, titles, labels, etc.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

DEFAULT_FIG_SIZE = (10, 6)
DEFAULT_GOOD_COLOR = '#b4cde3' # blue
DEFAULT_BAD_COLOR = '#fbb4ae' # red
DEFAULT_PALETTE = [DEFAULT_GOOD_COLOR, DEFAULT_BAD_COLOR]


def configure_illustrator_fonts() -> None:
    """Use TrueType fonts in PDF so text stays editable in Adobe Illustrator."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_fig_for_illustrator(
    fig,
    save_path: str,
    *,
    dpi: int = 300,
    rasterize_collections: bool = True,
) -> None:
    """
    Save a figure as PDF with vector text and rasterized plot fills.

    Call configure_illustrator_fonts() before creating the figure so labels
    are exported as editable text rather than Type 3 bitmap glyphs.
    """
    if rasterize_collections:
        for ax in fig.axes:
            for artist in ax.collections:
                artist.set_rasterized(True)

    fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")


def default_significance_label(p: float) -> str:
    """Return a string of stars (or 'ns') for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# 1. Violin grid with per-feature significance annotations
# ---------------------------------------------------------------------------

def plot_violin_grid(
    df_data: pd.DataFrame,
    feature_map: Mapping[str, str],
    group_col: str,
    *,
    df_pvals: Optional[pd.DataFrame] = None,
    pval_feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    significance_label_fn: Callable[[float], str] = default_significance_label,
    x_label: str = '',
    y_labels: Mapping[str, str] = '',
    group_order: Sequence[str] = '',
    palette: Sequence[str] = DEFAULT_PALETTE,
    main_title: str = '',
    sharey: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIG_SIZE,
    save_path: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    print_show: bool = True,
):
    """
    Plot one violin plot per feature, in a single row.

    Parameters
    ----------
    df_data:
        Dataframe containing the raw values to plot.
    feature_map:
        Mapping from display name -> column name in df_data.
        Example: {"MOCA score": "MOCA_score", "TMT (Trail B)": "Trail_B_(second)"}.
    group_col:
        Column used on x-axis (e.g. biomarker category, group).
    df_pvals:
        Optional dataframe with per-feature p-values. If provided, used to
        annotate significance above each subplot.
    pval_feature_col, pval_col:
        Column names in df_pvals to match features and read p-values from.
    significance_label_fn:
        Function p -> string (e.g. "***", "**", "*", "ns").
    x_label:
        Label for x-axis; defaults to group_col if None.
    y_labels:
        Optional mapping from display name -> y-axis label. If None, uses
        the display name.
    group_order:
        Optional order of groups on the x-axis.
    palette:
        Optional list of colors for groups.
    main_title:
        Optional overall title for the figure.
    sharey:
        If True, share y-axis across features.
    figsize:
        (width, height) for the figure.
    y_lim:
        Optional (ymin, ymax) to fix y-axis limits for all subplots (e.g. (0, 1)
        for 0–1 scaled variables).

    Returns
    -------
    fig, axes
    """
    features = list(feature_map.items())
    n = len(features)

    fig, axes = plt.subplots(
        1, n, figsize=figsize, sharey=sharey, squeeze=False
    )
    axes = axes[0]

    # Build simple p-value dict if df_pvals is provided
    p_dict: Dict[str, float] = {}
    if df_pvals is not None:
        for feature_name in feature_map.keys():
            feature_row = df_pvals[df_pvals[pval_feature_col] == feature_name]
            if not feature_row.empty:
                p_dict[feature_name] = float(feature_row.iloc[0][pval_col])

    for ax, (feature_name, col_name) in zip(axes, features):
        # Filter data to remove NaN values for this feature
        plot_data = df_data[df_data[col_name].notna()].copy()

        # Create violin plot
        sns.violinplot(
            data=plot_data,
            x=group_col,
            y=col_name,
            hue=group_col,
            ax=ax,
            order=group_order,
            hue_order=group_order,
            palette=palette,
            legend=False,
        )

        # Add significance marker with adjusted p-value, if available
        p_val = p_dict.get(feature_name)
        if p_val is not None:
            sig_marker = significance_label_fn(p_val)

            # Place annotation in axes coordinates so it also works for
            # non-numeric or discrete y-data (e.g. 0/1 encoded as strings).
            ax.text(
                0.5,
                0.95,
                f"{sig_marker}\np = {p_val:.4g}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
            )

        ax.set_title(feature_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(
            y_labels.get(feature_name, feature_name), fontsize=10
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Clip y-axis at 0 to avoid showing negative density tails
        ymin, ymax = ax.get_ylim()
        if ymin < 0:
            ax.set_ylim(bottom=0, top=ymax)
        if y_lim is not None:
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    if print_show:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# 2. Boxplot grid (same layout as violin)
# ---------------------------------------------------------------------------

def plot_box_grid(
    df_data: pd.DataFrame,
    feature_map: Mapping[str, str],
    group_col: str,
    *,
    df_pvals: Optional[pd.DataFrame] = None,
    pval_feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    significance_label_fn: Callable[[float], str] = default_significance_label,
    x_label: str = '',
    y_labels: Mapping[str, str] = '',
    group_order: Sequence[str] = '',
    palette: Sequence[str] = DEFAULT_PALETTE,
    main_title: str = '',
    sharey: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIG_SIZE,
    save_path: Optional[str] = None,
    print_show: bool = True,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
):
    """
    Plot one boxplot per feature (similar to plot_violin_grid).

    Parameters are identical to plot_violin_grid, except that seaborn.boxplot
    is used instead of violinplot. If nrows/ncols are set, subplots are laid
    out in that grid (extra cells are hidden).
    """
    features = list(feature_map.items())
    n = len(features)

    if nrows is not None and ncols is not None:
        fig, axes_2d = plt.subplots(
            nrows, ncols, figsize=figsize, sharey=sharey, squeeze=False
        )
        axes = axes_2d.flatten()
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)
        axes = axes[:n]
    else:
        fig, axes_2d = plt.subplots(
            1, n, figsize=figsize, sharey=sharey, squeeze=False
        )
        axes = axes_2d[0]

    # Build simple p-value dict if df_pvals is provided
    p_dict: Dict[str, float] = {}
    if df_pvals is not None:
        for feature_name in feature_map.keys():
            feature_row = df_pvals[df_pvals[pval_feature_col] == feature_name]
            if not feature_row.empty:
                p_dict[feature_name] = float(feature_row.iloc[0][pval_col])

    for ax, (feature_name, col_name) in zip(axes, features):
        plot_data = df_data[df_data[col_name].notna()].copy()

        sns.boxplot(
            data=plot_data,
            x=group_col,
            y=col_name,
            hue=group_col,
            ax=ax,
            order=group_order,
            hue_order=group_order,
            palette=palette,
            legend=False,
        )

        p_val = p_dict.get(feature_name)
        if p_val is not None:
            sig_marker = significance_label_fn(p_val)

            # Use axes coordinates so annotation works for any y type
            ax.text(
                0.5,
                0.95,
                f"{sig_marker}\np = {p_val:.4g}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
            )

        ax.set_title(feature_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(
            y_labels.get(feature_name, feature_name), fontsize=10
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if print_show:
        plt.show()


    return fig, axes


# ---------------------------------------------------------------------------
# 3. Negative log(p) bar plot with thresholds
# ---------------------------------------------------------------------------

def plot_neg_log_p_bars(
    df_pvals: pd.DataFrame,
    feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    *,
    feature_order: Optional[Sequence[str]] = None,
    thresholds: Sequence[float] = (0.05, 0.01, 0.001),
    colors_by_level: Optional[Dict[str, str]] = None,
    title: str = "Negative log(adjusted p-values)",
    x_label: str = "Feature",
    y_label: str = "Negative log(adjusted p-value)",
    figsize: Tuple[float, float] = DEFAULT_FIG_SIZE,
    save_path: Optional[str] = None,
):
    """
    Plot a bar chart of -log(p) with color-coded significance and threshold lines.

    Parameters
    ----------
    df_pvals:
        Dataframe with at least [feature_col, pval_col].
    feature_col, pval_col:
        Column names in df_pvals.
    feature_order:
        Optional order of features on x-axis; otherwise uses dataframe order.
    thresholds:
        List of p-value thresholds for reference lines.
    colors_by_level:
        Mapping 'highly', 'very', 'sig', 'ns' -> color hex. If None, uses
        defaults similar to the notebook: red/orange/green/gray.
    """
    if colors_by_level is None:
        colors_by_level = {
            "highly": "#d62728",  # p < 0.001
            "very": "#ff7f0e",    # p < 0.01
            "sig": "#2ca02c",     # p < 0.05
            "ns": "#7f7f7f",      # not significant
        }

    df = df_pvals[[feature_col, pval_col]].dropna().copy()
    if feature_order is not None:
        df = df.set_index(feature_col).loc[list(feature_order)].reset_index()

    features = df[feature_col].tolist()
    p_values = df[pval_col].astype(float).tolist()
    neg_log = [-np.log(p) for p in p_values]

    # Colors for bars based on significance levels
    bar_colors = []
    for p in p_values:
        if p < 0.001:
            bar_colors.append(colors_by_level["highly"])
        elif p < 0.01:
            bar_colors.append(colors_by_level["very"])
        elif p < 0.05:
            bar_colors.append(colors_by_level["sig"])
        else:
            bar_colors.append(colors_by_level["ns"])

    # Thresholds in -log space
    log_thresholds = [-np.log(t) for t in thresholds]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        features,
        neg_log,
        color=bar_colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Annotate with raw p-values
    for bar, p_val in zip(bars, p_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2.0,
            f"p = {p_val:.4g}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    # Add threshold lines
    line_colors = ["red", "orange", "darkred"]
    for thr, log_thr, c in zip(thresholds, log_thresholds, line_colors):
        ax.axhline(
            y=log_thr,
            color=c,
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label=f"p = {thr:g} threshold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(neg_log + log_thresholds) * 1.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10)
    plt.xticks(rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# 4. Heatmap of -log(p) with star annotations
# ---------------------------------------------------------------------------

def plot_pvalue_heatmap(
    df_pvals: pd.DataFrame,
    feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    *,
    feature_order: Optional[Sequence[str]] = None,
    center_p: float = 0.1,
    thresholds: Sequence[Tuple[float, str]] = (
        (0.005, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ),
    cmap: str = "coolwarm",
    title: str = "Adjusted p-values",
    figsize_per_row: float = 0.5,
    min_fig_height: float = 3.0,
    vmax_p: float = 0.001,
    save_path: Optional[str] = None,
):
    """
    Plot a 1D heatmap of -log(p) with star annotations for significance.

    Parameters
    ----------
    df_pvals:
        DataFrame with at least [feature_col, pval_col].
    feature_col, pval_col:
        Column names in df_pvals.
    feature_order:
        Optional explicit order of features on y-axis.
    center_p:
        p-value to use as center of the color scale (in -log space).
    thresholds:
        List of (p_threshold, star_string) pairs, evaluated from most to
        least stringent.
    cmap:
        Matplotlib colormap name.
    title:
        Title for the heatmap.
    figsize_per_row, min_fig_height:
        Figure height is max(min_fig_height, n_rows * figsize_per_row).
    vmax_p:
        Maximum p-value (smallest p) that maps to top of color scale.
    """
    df = df_pvals[[feature_col, pval_col]].dropna().copy()
    if feature_order is not None:
        df = df.set_index(feature_col).loc[list(feature_order)].reset_index()

    df = df.set_index(feature_col)
    neg_log = -np.log(df[pval_col].astype(float))
    df_plot = pd.DataFrame({"-log(adjusted p-value)": neg_log})

    def star_for_logp(x: float) -> str:
        for thr, stars in thresholds:
            if x > -np.log(thr):
                return stars
        return ""

    annot = neg_log.apply(star_for_logp).to_frame()
    annot.columns = ["-log(adjusted p-value)"]

    n_rows = df_plot.shape[0]
    fig_height = max(min_fig_height, n_rows * figsize_per_row)

    plt.figure(figsize=(5, fig_height))

    sns.heatmap(
        data=df_plot[["-log(adjusted p-value)"]],
        cmap=cmap,
        center=-np.log(center_p),
        annot=annot,
        fmt="s",
        vmin=0,
        vmax=-np.log(vmax_p),
        yticklabels=df_plot.index,
        cbar_kws={"label": "-log(Adjusted p-value)"},
    )

    plt.xlabel("")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.gcf().savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()
    return

    # return plt.gcf(), plt.gca()


# ---------------------------------------------------------------------------
# 4b. Multi-group p-value heatmap (features x groups)
# ---------------------------------------------------------------------------

def plot_multigroup_pvalue_heatmap(
    df_pvals: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
    feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    *,
    feature_order: Optional[Sequence[str]] = None,
    center_p: float = 0.1,
    thresholds: Sequence[Tuple[float, str]] = (
        (0.005, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ),
    cmap: str = "Blues",
    title: str = "Adjusted p-values by group",
    figsize_per_row: float = 0.5,
    min_fig_height: float = 3.0,
    vmax_p: float = 0.001,
    save_path: Optional[str] = None,
):
    """
    Plot a 2D heatmap of -log(p) with features on the y-axis and groups on
    the x-axis, using a single shared color scale and star annotations per
    (feature, group) cell.

    This is useful when df_pvals contains repeated measurements of the same
    feature for different subgroups, e.g. MCI_Status in the MOCA analysis.

    Parameters
    ----------
    df_pvals:
        DataFrame with at least [feature_col, group_col, pval_col].
    group_col:
        Column indicating the subgroup (e.g. "MCI_Status").
    groups:
        Ordered list of group labels to show as columns. Only these groups
        are included and the columns appear in this order.
    feature_col, pval_col:
        Column names in df_pvals for the feature name and p-value.
    feature_order:
        Optional explicit order of features on the y-axis.
    center_p:
        p-value to use as the center of the color scale (in -log space).
    thresholds:
        List of (p_threshold, star_string) pairs, evaluated from most to
        least stringent, used for star annotations.
    cmap:
        Matplotlib colormap name.
    title:
        Title for the heatmap.
    figsize_per_row, min_fig_height:
        Figure height is max(min_fig_height, n_rows * figsize_per_row).
    vmax_p:
        Maximum p-value (smallest p) that maps to top of color scale.
    """
    # Filter to requested groups and columns, drop missing p-values
    df = df_pvals[
        df_pvals[group_col].isin(groups)
    ][[feature_col, group_col, pval_col]].dropna().copy()

    # Wide table: rows = feature, columns = group, values = p-value
    wide = df.pivot(index=feature_col, columns=group_col, values=pval_col)

    # Ensure all requested groups appear as columns in the desired order
    wide = wide.reindex(columns=list(groups))

    # Optionally reorder features
    if feature_order is not None:
        wide = wide.reindex(index=list(feature_order))

    # Convert to -log(p) for coloring
    neg_log = -np.log(wide.astype(float))

    # Build star annotations from raw p-values (not from -log space)
    def star_for_p(p_val: float) -> str:
        if pd.isna(p_val):
            return ""
        for thr, stars in thresholds:
            if p_val < thr:
                return stars
        return ""

    annot = wide.applymap(star_for_p)

    n_rows = neg_log.shape[0]
    fig_height = max(min_fig_height, n_rows * figsize_per_row)

    fig, ax = plt.subplots(
        figsize=(max(5.0, len(groups) * 1.5), fig_height)
    )

    sns.heatmap(
        data=neg_log,
        cmap=cmap,
        center=-np.log(center_p),
        annot=annot,
        fmt="s",
        vmin=0,
        vmax=-np.log(vmax_p),
        yticklabels=neg_log.index,
        cbar_kws={"label": "-log(Adjusted p-value)"},
        ax=ax,
    )

    ax.set_xlabel(group_col)
    ax.set_ylabel("Feature")
    ax.set_title(title)

    # Rotate x-tick labels to 45 degrees (was 90)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 4c. Multi-group p-value heatmap with male/female columns
# ---------------------------------------------------------------------------

def plot_multigroup_pvalue_heatmap_with_overall_sex(
    df_pvals: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
    feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    pval_col_males: str = "adjusted p-value_males",
    pval_col_females: str = "adjusted p-value_females",
    *,
    overall_group_value: Optional[str] = "All",
    feature_order: Optional[Sequence[str]] = None,
    center_p: float = 0.1,
    thresholds: Sequence[Tuple[float, str]] = (
        (0.005, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ),
    cmap: str = "Blues",
    title: str = "Adjusted p-values by group + overall males/females",
    figsize_per_row: float = 0.5,
    min_fig_height: float = 3.0,
    vmax_p: float = 0.001,
    save_path: Optional[str] = None,
):
    """
    Plot a 2D heatmap of -log(p) with features on the y-axis and columns:
        [groups..., "Males", "Females"].

    - The first len(groups) columns are exactly as in
      plot_multigroup_pvalue_heatmap: they use `pval_col` for each value in
      `groups` along `group_col` (e.g. All / Cognitively Healthy /
      Cognitively Impaired for MCI_Status).
    - The final two columns ("Males", "Females") use the rows where
      `group_col == overall_group_value` (by default "All") and the
      p-values from `pval_col_males` and `pval_col_females`.

    This keeps the same column layout as the original multigroup heatmap
    and simply appends overall male/female columns on the right.
    """
    if overall_group_value is None:
        overall_group_value = groups[0]

    # 1) Base multigroup wide table using the main p-value column
    df_groups = df_pvals[
        df_pvals[group_col].isin(groups)
    ][[feature_col, group_col, pval_col]].dropna(subset=[pval_col]).copy()

    wide_groups = df_groups.pivot(
        index=feature_col, columns=group_col, values=pval_col
    )
    wide_groups = wide_groups.reindex(columns=list(groups))

    if feature_order is not None:
        wide_groups = wide_groups.reindex(index=list(feature_order))

    # 2) Overall male/female p-values from the specified "All" group
    df_overall = df_pvals[
        df_pvals[group_col] == overall_group_value
    ][[feature_col, pval_col_males, pval_col_females]].copy()
    df_overall = df_overall.set_index(feature_col)

    features_idx = wide_groups.index
    males = df_overall[pval_col_males].reindex(features_idx).astype(float)
    females = df_overall[pval_col_females].reindex(features_idx).astype(float)

    sex_df = pd.DataFrame(
        {"Males": males, "Females": females},
        index=features_idx,
    )

    wide = pd.concat([wide_groups, sex_df], axis=1)

    # Convert to -log(p) for coloring
    neg_log = -np.log(wide.astype(float))

    # Build star annotations from raw p-values (not from -log space)
    def star_for_p(p_val: float) -> str:
        if pd.isna(p_val):
            return ""
        for thr, stars in thresholds:
            if p_val < thr:
                return stars
        return ""

    annot = wide.applymap(star_for_p)

    n_rows = neg_log.shape[0]
    fig_height = max(min_fig_height, n_rows * figsize_per_row)
    n_cols = len(groups) + 2

    fig, ax = plt.subplots(
        figsize=(max(5.0, n_cols * 1.5), fig_height)
    )

    sns.heatmap(
        data=neg_log,
        cmap=cmap,
        center=-np.log(center_p),
        annot=annot,
        fmt="s",
        vmin=0,
        vmax=-np.log(vmax_p),
        yticklabels=neg_log.index,
        cbar_kws={"label": "-log(Adjusted p-value)"},
        ax=ax,
    )

    ax.set_ylabel("Feature")
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Histogram / KDE grid for continuous features by group
# ---------------------------------------------------------------------------

def plot_hist_grid(
    df_data: pd.DataFrame,
    feature_map: Mapping[str, str],
    group_col: str,
    *,
    df_pvals: Optional[pd.DataFrame] = None,
    pval_feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    significance_label_fn: Callable[[float], str] = default_significance_label,
    kde: bool = True,
    stat: str = "density",
    common_norm: bool = False,
    multiple: str = "dodge",
    palette: Sequence[str] = DEFAULT_PALETTE,
    bins: Optional[Sequence[float]] = "auto",
    main_title: str = '',
    sharey: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIG_SIZE,
    save_path: Optional[str] = None,
    share_legend: bool = False,
):
    """
    Plot histograms (optionally with KDE) for multiple continuous features,
    colored by a grouping column (e.g. group_col='tau_217_category').

    Each feature is plotted in its own subplot (one row).

    Parameters
    ----------
    df_data:
        Dataframe with raw values.
    feature_map:
        Mapping from display name -> column name in df_data.
    group_col:
        Column used as hue (group) in histplot.
    kde, stat, common_norm, multiple:
        Passed directly to seaborn.histplot.
    palette:
        Color palette for groups.
    bins:
        Optional explicit bin edges/count.
    main_title:
        Overall figure title.
    sharey:
        If True, share y-axis across rows.
    figsize:
        (width, height).
    """
    features = list(feature_map.items())
    n = len(features)

    # Optional p-value lookup per feature (by display name)
    p_dict: Dict[str, float] = {}
    if df_pvals is not None:
        for feature_name in feature_map.keys():
            feature_row = df_pvals[df_pvals[pval_feature_col] == feature_name]
            if not feature_row.empty:
                p_dict[feature_name] = float(feature_row.iloc[0][pval_col])

    fig, axes = plt.subplots(
        1, n, figsize=figsize, sharex=False, sharey=sharey, squeeze=False
    )
    axes = axes[0, :]

    for idx, (ax, (feature_name, col_name)) in enumerate(zip(axes, features)):
        plot_data = df_data[df_data[col_name].notna()].copy()

        sns.histplot(
            data=plot_data,
            x=col_name,
            hue=group_col,
            stat=stat,
            multiple=multiple,
            common_norm=common_norm,
            kde=kde,
            ax=ax,
            palette=palette,
            bins=bins,
            # If share_legend is True, only show legend on the first subplot
            legend=(not share_legend) or (share_legend and idx == 0),
        )

        # Optional p-value annotation at top-center of each subplot
        p_val = p_dict.get(feature_name)
        if p_val is not None:
            sig_marker = significance_label_fn(p_val)

            ax.text(
                0.5,
                0.9,
                f"{sig_marker}\np = {p_val:.4g}",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
            )

        ax.set_title(feature_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel(stat.capitalize(), fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# 6. KDE grid (density only, clipped to data support)
# ---------------------------------------------------------------------------

def plot_kde_grid(
    df_data: pd.DataFrame,
    feature_map: Mapping[str, str],
    group_col: str,
    *,
    df_pvals: Optional[pd.DataFrame] = None,
    pval_feature_col: str = "Feature",
    pval_col: str = "adjusted p-value",
    significance_label_fn: Callable[[float], str] = default_significance_label,
    x_label: str = '',
    y_labels: Mapping[str, str] = '',
    group_order: Optional[Sequence[str]] = None,
    palette: Sequence[str] = DEFAULT_PALETTE,
    main_title: str = '',
    sharey: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIG_SIZE,
    save_path: Optional[str] = None,
    share_legend: bool = True,
):
    """
    Plot one KDE (kernel density estimate) per feature, in a single row.
    Each KDE is clipped to the data support (min/max of values) so the curve
    is not drawn where there are no observations.

    Parameters
    ----------
    df_data:
        Dataframe containing the raw values to plot.
    feature_map:
        Mapping from display name -> column name in df_data.
    group_col:
        Column used as hue (group) for separate KDE curves.
    df_pvals:
        Optional dataframe with per-feature p-values for significance annotations.
    pval_feature_col, pval_col:
        Column names in df_pvals to match features and read p-values from.
    significance_label_fn:
        Function p -> string (e.g. "***", "**", "*", "ns").
    x_label:
        Label for x-axis.
    y_labels:
        Optional mapping from display name -> y-axis label.
    group_order:
        Optional order of groups in the legend.
    palette:
        Colors for groups.
    main_title:
        Optional overall title for the figure.
    sharey:
        If True, share y-axis across features.
    figsize:
        (width, height) for the figure.
    share_legend:
        If True, show legend only on the first subplot.
    """
    features = list(feature_map.items())
    n = len(features)

    fig, axes = plt.subplots(
        1, n, figsize=figsize, sharey=sharey, squeeze=False
    )
    axes = axes[0]

    # Normalize so .get() is safe when caller omits y_labels (default '')
    y_labels = y_labels if isinstance(y_labels, Mapping) else {}

    p_dict: Dict[str, float] = {}
    if df_pvals is not None:
        for feature_name in feature_map.keys():
            feature_row = df_pvals[df_pvals[pval_feature_col] == feature_name]
            if not feature_row.empty:
                p_dict[feature_name] = float(feature_row.iloc[0][pval_col])

    for idx, (ax, (feature_name, col_name)) in enumerate(zip(axes, features)):
        plot_data = df_data[df_data[col_name].notna()].copy()
        if plot_data.empty:
            ax.set_title(feature_name, fontsize=12, fontweight="bold")
            ax.set_xlabel(x_label or col_name, fontsize=10)
            ax.set_ylabel(y_labels.get(feature_name, "Density"), fontsize=10)
            continue

        data_min = float(plot_data[col_name].min())
        data_max = float(plot_data[col_name].max())
        clip = (data_min, data_max)

        sns.kdeplot(
            data=plot_data,
            x=col_name,
            hue=group_col,
            ax=ax,
            palette=palette,
            hue_order=group_order,
            clip=clip,
            common_norm=False,
            fill=True,
            alpha=0.4,
            linewidth=1.5,
            legend=(not share_legend) or (share_legend and idx == 0),
        )

        # Optional p-value annotation
        p_val = p_dict.get(feature_name)
        if p_val is not None:
            sig_marker = significance_label_fn(p_val)
            ax.text(
                0.5,
                0.95,
                f"{sig_marker}\np = {p_val:.4g}",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
            )

        ax.set_title(feature_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label or y_labels.get(feature_name, feature_name), fontsize=10)
        ax.set_ylabel(y_labels.get(feature_name, "Density"), fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        # Clip y-axis at 0 so density does not show below zero
        ymin, ymax = ax.get_ylim()
        if ymin < 0:
            ax.set_ylim(bottom=0, top=ymax)

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()
    return fig, axes

# ---------------------------------------------------------------------------
# 7. Stacked 0/1 bar plot by group (percent)
# ---------------------------------------------------------------------------

def plot_binary_stacked_bar(
    df_data: pd.DataFrame,
    feature_col: str,
    group_col: str,
    *,
    x_order: Optional[Sequence] = None,
    group_order: Optional[Sequence[str]] = None,
    x_labels: Optional[Mapping] = None,
    palette: Sequence[str] = DEFAULT_PALETTE,
    title: str = "",
    x_axis_label: str = "",
    y_axis_label: str = "Percentage",
    figsize: Tuple[float, float] = (5.0, 4.0),
    save_path: Optional[str] = None,
):
    """
    Plot a single stacked bar chart for a binary (0/1, NO/YES, etc.) feature,
    where bars at each x-level are stacked by group, and the y-axis shows
    the percentage of observations at that level.

    Parameters
    ----------
    df_data:
        DataFrame containing the raw values.
    feature_col:
        Name of the binary feature column in df_data.
    group_col:
        Column indicating the group (e.g. "tau_217_category" with "High"/"Low").
    x_order:
        Optional explicit order of the binary levels on the x-axis, e.g. [0, 1]
        or ["NO", "YES"]. If None, uses the sorted unique values present.
    group_order:
        Optional order of groups for stacking. If None, uses sorted unique
        group labels from the data.
    x_labels:
        Optional mapping from raw x values -> display labels on the axis
        (e.g. {0: "No", 1: "Yes"}).
    palette:
        Colors for groups, in the order of group_order.
    title:
        Main title for the plot.
    x_axis_label:
        Label for the x-axis (e.g. "Herpes zoster vaccine").
    y_axis_label:
        Label for the y-axis. Defaults to "Percentage".
    figsize:
        (width, height) of the figure.
    save_path:
        If provided, save the figure as a PDF to this path.
    """
    plot_data = df_data[[feature_col, group_col]].dropna().copy()
    if plot_data.empty:
        raise ValueError("No data available for the requested feature / group.")

    # Determine x-levels and group order
    x_levels = (
        list(x_order)
        if x_order is not None
        else sorted(plot_data[feature_col].dropna().unique())
    )
    if group_order is None:
        group_order = sorted(plot_data[group_col].dropna().unique())

    # Counts: index = feature level, columns = group
    counts = (
        plot_data.groupby([feature_col, group_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=x_levels)
    )

    # Percentages per x-level (so each bar sums to 100%)
    perc = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros(len(x_levels), dtype=float)
    for idx, group in enumerate(group_order):
        if group not in perc.columns:
            heights = np.zeros(len(x_levels), dtype=float)
        else:
            heights = perc[group].to_numpy()

        color = palette[idx % len(palette)]
        ax.bar(
            range(len(x_levels)),
            heights,
            bottom=bottoms,
            label=group,
            color=color,
            edgecolor="black",
            linewidth=1.0,
        )
        bottoms += np.nan_to_num(heights)

    # X-axis tick labels
    if x_labels is not None:
        tick_labels = [x_labels.get(x, x) for x in x_levels]
    else:
        tick_labels = x_levels

    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels(tick_labels)

    ax.set_ylim(0, 100)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label or feature_col)
    ax.set_title(title or feature_col)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(title=group_col)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 8. 2x2 KDE: p-tau217 by tau category vs. a 2-level factor
# ---------------------------------------------------------------------------

def plot_tau217_kde_by_factor(
    df_data: pd.DataFrame,
    factor_col: str,
    *,
    factor_mode: str = "categorical",  # "categorical" or "threshold"
    factor_categories: Optional[Sequence[str]] = None,
    factor_threshold: Optional[float] = None,
    factor_category_col: Optional[str] = None,
    factor_high_label: str = "High",
    factor_low_label: str = "Low",
    tau_value_col: str = "p-tau217",
    tau_cutoff: float = 0.444,
    tau_category_col: str = "tau_217_category",
    tau_high_label: str = "High",
    tau_low_label: str = "Low",
    tau_palette: Optional[Mapping[str, str]] = None,
    factor_palette: Optional[Mapping[str, str]] = None,
    clip_kde_to_data: bool = True,
    figsize: Tuple[float, float] = (12, 8),
    main_title: Optional[str] = None,
    print_show: bool = True,
    copy_df: bool = True,
    return_df: bool = False,
    save_path: Optional[str] = None,
):
    """
    Reusable version of the repeated notebook logic.

    Produces a 2x2 grid:
      - Row 1: for each factor level, overlay KDEs of p-tau217 split by tau
        category (tau_high_label vs tau_low_label).
      - Row 2: compare tau_high_label and tau_low_label groups separately,
        each split by the factor levels.

    Each KDE is clipped to the observed value range of the data used for
    that particular curve (so density is not drawn outside the "real" range).
    
    If `copy_df` is True (default), the function works on a dataframe copy.
    If you set `return_df=True`, it will also return the dataframe with
    the added `tau_category_col` / factor category column (if they were
    missing).
    """

    if tau_palette is None:
        tau_palette = {tau_high_label: DEFAULT_BAD_COLOR, tau_low_label: DEFAULT_GOOD_COLOR}

    if save_path is not None:
        configure_illustrator_fonts()

    # Work on a copy so we don't surprise callers by adding columns.
    df = df_data.copy() if copy_df else df_data

    # 1) Tau category
    if tau_category_col not in df.columns:
        df[tau_category_col] = np.where(
            df[tau_value_col] >= tau_cutoff, tau_high_label, tau_low_label
        )

    tau_groups = [tau_high_label, tau_low_label]

    # 2) Factor level -> categorical labels
    if factor_mode not in {"categorical", "threshold"}:
        raise ValueError("factor_mode must be either 'categorical' or 'threshold'")

    if factor_mode == "threshold":
        if factor_threshold is None:
            raise ValueError("factor_threshold is required for factor_mode='threshold'")
        cat_col = factor_category_col or f"{factor_col}_category"
        df[cat_col] = np.where(df[factor_col] >= factor_threshold, factor_high_label, factor_low_label)
        factor_level_col = cat_col
        factor_levels = [factor_high_label, factor_low_label]
        factor_label_base = factor_col
    else:
        factor_level_col = factor_col
        factor_levels_raw = (
            list(factor_categories)
            if factor_categories is not None
            else list(pd.Series(df[factor_col].dropna().unique()).tolist())
        )
        if len(factor_levels_raw) != 2:
            raise ValueError(
                f"Expected exactly 2 factor levels in '{factor_col}', got {len(factor_levels_raw)}. "
                "Pass factor_categories to control which two to plot."
            )
        factor_levels = factor_levels_raw
        factor_label_base = factor_col

    # Palette for the factor levels (colors correspond to factor_levels order).
    if factor_palette is None:
        factor_palette = {
            factor_levels[0]: DEFAULT_BAD_COLOR,
            factor_levels[1]: DEFAULT_GOOD_COLOR,
        }

    def _kde_clipped(ax, data: pd.Series, *, color: str, label: str) -> None:
        data = pd.Series(data).dropna()
        if data.empty:
            return
        if data.nunique(dropna=True) < 2:
            return

        clip = None
        if clip_kde_to_data:
            data_min = float(data.min())
            data_max = float(data.max())
            if np.isfinite(data_min) and np.isfinite(data_max) and data_min < data_max:
                clip = (data_min, data_max)

        sns.kdeplot(
            data=data,
            ax=ax,
            color=color,
            label=label,
            fill=True,
            alpha=0.5,
            linewidth=1.5,
            clip=clip,
        )

    # 2x2 layout like in the notebook
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)

    # Row 1: for each factor level, overlay tau groups
    for i, factor_level in enumerate(factor_levels):
        ax = axes[0, i]
        subset = df[df[factor_level_col] == factor_level]
        for tau_group in tau_groups:
            plot_data = subset.loc[subset[tau_category_col] == tau_group, tau_value_col]
            _kde_clipped(
                ax,
                plot_data,
                color=tau_palette[tau_group],
                label=f"{tau_group} p-tau217",
            )
        ax.set_title(f"{factor_label_base}: {factor_level}")
        ax.set_xlabel(tau_value_col)
        ax.set_ylabel("Density")
        ax.legend()

    # Row 2-left: tau_high group split by factor
    ax = axes[1, 0]
    for factor_level in factor_levels:
        subset = df[
            (df[tau_category_col] == tau_high_label)
            & (df[factor_level_col] == factor_level)
        ]
        _kde_clipped(
            ax,
            subset[tau_value_col],
            color=factor_palette[factor_level],
            label=f"{factor_label_base}: {factor_level}",
        )
    ax.set_title(
        f"{tau_value_col}: {tau_high_label} group,\nby {factor_label_base} "
        f"{factor_levels[0]}/{factor_levels[1]}"
    )
    ax.set_xlabel(tau_value_col)
    ax.set_ylabel("Density")
    ax.legend()

    # Row 2-right: tau_low group split by factor
    ax = axes[1, 1]
    for factor_level in factor_levels:
        subset = df[
            (df[tau_category_col] == tau_low_label)
            & (df[factor_level_col] == factor_level)
        ]
        _kde_clipped(
            ax,
            subset[tau_value_col],
            color=factor_palette[factor_level],
            label=f"{factor_label_base}: {factor_level}",
        )
    ax.set_title(
        f"{tau_value_col}: {tau_low_label} group,\nby {factor_label_base} "
        f"{factor_levels[0]}/{factor_levels[1]}"
    )
    ax.set_xlabel(tau_value_col)
    ax.set_ylabel("Density")
    ax.legend()

    if main_title is None:
        main_title = (
            f"{tau_value_col} distribution by {factor_label_base} and tau category "
            f"(top: by {factor_label_base}; bottom: by tau group)"
        )
    fig.suptitle(main_title, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        save_fig_for_illustrator(fig, save_path)

    if print_show:
        plt.show()

    if return_df:
        return fig, axes, df
    return fig, axes