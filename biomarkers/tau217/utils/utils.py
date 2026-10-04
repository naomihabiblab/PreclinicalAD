import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data')
from clinical_data_analysis.statistical_analysis.settings import *


def add_significance_indicators(mean1, mean2, p_value, ax=None):
    """
    Add significance indicators (horizontal line with asterisks) to a plot.
    
    Parameters:
    -----------
    mean1 : float
        Mean of first group
    mean2 : float
        Mean of second group
    p_value : float
        P-value from statistical test
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    
    Returns:
    --------
    None
    """
    if ax is None:
        ax = plt.gca()
    
    # Only add indicators if significant
    if p_value >= 0.05:
        return
    
    # Determine significance level
    if p_value < 0.005:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    else:
        sig_text = "*"
    
    # Get current y-axis limits
    y_min, y_max = ax.get_ylim()
    
    # Position the significance line above the plot
    sig_y = y_max * 1.15
    
    # Draw the significance line a bit lower
    sig_y = y_max * 1.08  # Lower than before (was 1.15)
    
    # Draw horizontal line connecting the two means
    ax.plot([mean1, mean2], [sig_y, sig_y], 'k-', linewidth=1.5, alpha=0.8)
    
    # Add vertical ticks at the ends of the line
    ax.plot([mean1, mean1], [sig_y - y_max*0.02, sig_y + y_max*0.02], 'k-', linewidth=1.5, alpha=0.8)
    ax.plot([mean2, mean2], [sig_y - y_max*0.02, sig_y + y_max*0.02], 'k-', linewidth=1.5, alpha=0.8)
    
    # Add asterisks above the horizontal line
    mid_point = (mean1 + mean2) / 2
    ax.text(mid_point, sig_y + y_max*0.03, sig_text, 
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Adjust y-axis limits to accommodate the significance line
    ax.set_ylim(y_min, y_max * 1.25)


def plot_kde(df, risk, tau_217=TAU_217, title=None, high_thr_217=HIGH_TAU_217_THR, labels=None, fig_size=(10, 5)):
    fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
    risk_tau_217 = df.loc[df[risk], tau_217]
    healthy_tau_217 = df.loc[~df[risk], tau_217]

    sns.kdeplot(data=healthy_tau_217, label='Healthy', alpha=0.5, fill=True, ax=ax1, clip=(healthy_tau_217.min(), healthy_tau_217.max()))
    sns.kdeplot(data=risk_tau_217, label=risk, alpha=0.5, fill=True, ax=ax1, clip=(risk_tau_217.min(), risk_tau_217.max()))

    ax1.vlines(high_thr_217, 0, ymax=ax1.get_ylim()[1], colors='#b4cde3', linestyles='dashed', label='high p-tau217 threshold')
    if labels:
        ax1.legend(labels=labels)
    else:
        ax1.legend()
    ax1.set_title('p-tau217')
    ax1.set_xlabel('Tau 217')
    ax1.set_ylabel('Density')

    if title:
        plt.suptitle(title)
    else:
        plt.suptitle(f'{risk} vs Healthy')
    plt.show()


def plot_hist_kde(df, risk, tau_217=TAU_217, title=None, high_thr_217=HIGH_TAU_217_THR, labels=None, fig_size=(10, 5), stat='count'):
    fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
    
    # Extract data
    risk_tau_217 = df.loc[df[risk], tau_217]
    healthy_tau_217 = df.loc[~df[risk], tau_217]

    print(f"Healthy samples: {len(healthy_tau_217)}")
    print(f"MCI samples: {len(risk_tau_217)}")

    # Plot p-tau217
    sns.histplot(data=healthy_tau_217, label='Healthy', alpha=0.5, fill=True, ax=ax1, kde=True, 
                 line_kws={'linewidth': 2}, bins=20, stat=stat)
    sns.histplot(data=risk_tau_217, label=risk, alpha=0.5, fill=True, ax=ax1, kde=True, 
                 line_kws={'linewidth': 2}, bins=20, stat=stat)

    ax1.vlines(high_thr_217, 0, ymax=ax1.get_ylim()[1], colors='#b4cde3', linestyles='dashed', label='high p-tau217 threshold')
    if labels:
        ax1.legend(labels=labels)
    else:
        ax1.legend()
    ax1.set_title('p-tau217')
    ax1.set_xlabel('Tau 217')
    ax1.set_ylabel('Count' if stat == 'count' else 'Percentage')

    if title:
        plt.suptitle(title)
    else:
        plt.suptitle(f'{risk} vs Healthy')
    plt.show()


def compare_groups_statistical(df, group_col, feature_col, group1_filter, group2_filter, 
                             group1_name="Group 1", group2_name="Group 2", 
                             feature_name="Feature", test_type='mannwhitneyu', 
                             alternative='greater', plot=True, fig_dir='/ems/elsc-labs/habib-n/yuval.rom/BCG/biomarkers/final_figures'):
    """
    Compare two groups on a continuous feature using statistical tests and visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    group_col : str
        Column name containing group identifiers
    feature_col : str
        Column name containing the continuous feature to compare
    group1_filter : callable or value
        Filter for first group (e.g., lambda x: x == 0, or just 0)
    group2_filter : callable or value
        Filter for second group (e.g., lambda x: (x == 1) | (x == 2), or [1, 2])
    group1_name : str
        Display name for first group
    group2_name : str
        Display name for second group
    feature_name : str
        Display name for the feature
    test_type : str
        Statistical test: 'mannwhitneyu', 'ttest_ind', 'ks_2samp', 'ranksums', 'chi2_contingency'
    alternative : str
        Alternative hypothesis: 'greater', 'less', or 'two-sided'
    plot : bool
        Whether to create the distribution plot
    
    Returns:
    --------
    None
    """

    # Filter groups
    if callable(group1_filter):
        df_group1 = df[group1_filter(df[group_col])]
    else:
        df_group1 = df[df[group_col] == group1_filter]
    
    if callable(group2_filter):
        df_group2 = df[group2_filter(df[group_col])]
    else:
        df_group2 = df[group2_filter(df[group_col])]
    
    # Remove NaN values for the test
    feature_group1 = df_group1[feature_col].dropna()
    feature_group2 = df_group2[feature_col].dropna()
    
    # Perform statistical test based on test_type
    if test_type == 'mannwhitneyu':
        from scipy.stats import mannwhitneyu
        statistic, p_value = mannwhitneyu(feature_group2, feature_group1, alternative=alternative)
        test_name = "Mann-Whitney U test (Wilcoxon rank-sum)"
    elif test_type == 'ttest_ind':
        from scipy.stats import ttest_ind
        statistic, p_value = ttest_ind(feature_group2, feature_group1, alternative=alternative)
        test_name = "Independent t-test"
    elif test_type == 'ks_2samp':
        from scipy.stats import ks_2samp
        statistic, p_value = ks_2samp(feature_group2, feature_group1, alternative=alternative)
        test_name = "Kolmogorov-Smirnov test"
    elif test_type == 'ranksums':
        from scipy.stats import ranksums
        statistic, p_value = ranksums(feature_group2, feature_group1, alternative=alternative)
        test_name = "Wilcoxon rank-sum test"
    elif test_type == 'chi2_contingency':
        from scipy.stats import chi2_contingency
        statistic, p_value, dof, expected = chi2_contingency(feature_group2, feature_group1)
        test_name = "Chi-square test"
    else:
        raise ValueError(f"Unsupported test_type: {test_type}. Use 'mannwhitneyu', 'ttest_ind', 'ks_2samp', 'ranksums', or 'chi2_contingency'")
    
    # Print results
    print(f"{test_name} one-sided '{alternative}' test results:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"{group1_name} (N={len(feature_group1)}): mean={feature_group1.mean():.4f}, std={feature_group1.std():.4f}")
    print(f"{group2_name} (N={len(feature_group2)}): mean={feature_group2.mean():.4f}, std={feature_group2.std():.4f}")
    
    # Determine alternative hypothesis text
    if alternative == 'greater':
        alt_text = f"{group2_name} group has greater {feature_name} values than {group1_name} group"
    elif alternative == 'less':
        alt_text = f"{group2_name} group has less {feature_name} values than {group1_name} group"
    else:  # two-sided
        alt_text = f"{group2_name} group has different {feature_name} values than {group1_name} group"
    
    print(f"Alternative hypothesis: {alt_text}")
    
    # Create plot if requested
    if plot:
        # Create histograms and KDEs
        sns.histplot(feature_group1, color='#b4cde3', alpha=0.5, 
                    label=f'{group1_name} N={len(feature_group1)}', 
                    kde=True, stat='density', bins=30, line_kws={'linewidth': 3})
        sns.histplot(feature_group2, color='#fbb4ae', alpha=0.5, 
                    label=f'{group2_name} N={len(feature_group2)}', 
                    kde=True, stat='density', bins=30, line_kws={'linewidth': 3})
        # Redraw KDE curves with a black edge (outline). KDE lines have many points; bin outlines have few.
        ax = plt.gca()
        all_lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 50]
        kde_colors = ['#b4cde3', '#fbb4ae']
        for line, line_color in zip(all_lines[-2:], kde_colors):
            x, y = line.get_xdata(), line.get_ydata()
            line.remove()
            ax.plot(x, y, color='black', linewidth=4, zorder=0)
            ax.plot(x, y, color=line_color, linewidth=2, zorder=1)
        
        # Add significance indicators if p < 0.05
        if p_value < 0.05:
            mean1 = feature_group1.mean()
            mean2 = feature_group2.mean()
            add_significance_indicators(mean1, mean2, p_value)
        
        plt.title(f'{feature_name} Distribution by Group')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'{fig_dir}/{feature_name}_distribution_by_{group_col}_{test_name}.pdf', bbox_inches='tight', format='pdf')
        plt.show()
