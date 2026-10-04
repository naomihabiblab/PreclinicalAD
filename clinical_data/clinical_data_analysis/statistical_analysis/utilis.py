import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.stats import levene

import sys
sys.path.append('../')

from settings import *

# Graphs Settings
healthy_color = '#b4cde3' # blue
risk_color = '#fbb4ae' # red
inter_color = '#fed9a6'
pie_colors = [healthy_color, inter_color, risk_color] # blue, orange, red
vline_color = 'red'

TAU_217 = 'p-tau217'
TAU_CAT = 'tau_217_category'
HIGH_TAU_217_THR = 0.53
INT_TAU_217_THR = 0.445

TMT_B_COL = 'Trail_B_(second)'
MBI_C_COL = 'Mild_behavioral_Impairment_Checklist'
IST_COL = 'IST'
ANEXITY_COL = 'Anxiety/depression'

MCI_THR = 26
MOCA_COL = 'MOCA_score'

ALPHA = 0.05


def kdeplot_high_vs_low(df, x_col, hue, palette, title, save_path, alpha=0.7, figsize=(5, 3), save=True):
    df_high, df_low = df[df[hue] == 'High'], df[df[hue] == 'Low']
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df_low, x=x_col, fill=True, clip=(df_low[x_col].min(), df_low[x_col].max()), color=palette[0], label='Low', alpha=alpha)
    sns.kdeplot(data=df_high, x=x_col, fill=True, clip=(df_high[x_col].min(), df_high[x_col].max()), color=palette[1], label='High', alpha=alpha)
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(save_path)
    plt.show()
    

def plot_histplot(df, col, hue, bins, palette, title, save_path, figsize=(5, 3), save=True):
    high = df[df[hue] == 'High']
    low = df[df[hue] == 'Low']
    plt.figure(figsize=figsize)
    print(f"high: {high[col].value_counts()}, mean: {(high[col] == 'YES').mean()}")
    print(f"low: {low[col].value_counts()}, mean: {(low[col] == 'YES').mean()}")
    sns.histplot(data=df, x=col, hue=hue, multiple='stack', bins=bins, palette=palette)
    plt.legend(labels=['High Tau', 'Low Tau'])
    plt.title(title)
    if save:
        plt.savefig(save_path)
    plt.show()



def analyze_two_groups(group1, group2, alpha=ALPHA):
    """
    Perform statistical analysis on two groups with a continuous feature.
    
    Parameters:
    group1, group2 (array-like): Data for each group
    alpha (float): Significance level, default is 0.05
    
    Returns:
    None (prints results and displays plot)
    """
    
    def check_normality(data, group_name):
        stat, p = stats.shapiro(data)
        print(f"Shapiro-Wilk test for {group_name}:")
        print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
        if p > alpha:
            print(f"{group_name} is likely normally distributed.")
        else:
            print(f"{group_name} is likely not normally distributed.")
        print()

    # Check normality for both groups
    check_normality(group1, "Group 1")
    check_normality(group2, "Group 2")

    # Check for equal variances
    stat, p = levene(group1, group2)
    print("Levene's test for equal variances:")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
    if p > alpha:
        print("The two groups likely have equal variances.")
        equal_var = True
    else:
        print("The two groups likely have unequal variances.")
        equal_var = False
    print()

    # Perform independent samples t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

    # Print results
    print("Independent Samples T-Test Results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < alpha:
        print(f"There is a statistically significant difference between the two groups (p < {alpha}).")
    else:
        print(f"There is no statistically significant difference between the two groups (p > {alpha}).")

    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.hist(group1, alpha=0.5, label='Group 1')
    plt.hist(group2, alpha=0.5, label='Group 2')
    plt.legend()
    plt.title('Distribution of Data in Both Groups')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def man_whitney(group1, group2, alpha=ALPHA, alternative='two-sided'):
    """
    Perform Mann-Whitney U test.
    
    Parameters:
    group1: First group (Low biomarker group)
    group2: Second group (High biomarker group)
    alpha: Significance level
    alternative: 'two-sided', 'greater', or 'less'
        - 'two-sided': H1: group1 != group2
        - 'greater': H1: group1 > group2 (Low > High)
        - 'less': H1: group1 < group2 (Low < High)
    """
    stat, p = stats.mannwhitneyu(group1, group2, alternative=alternative)
    print(f"Mann-Whitney U test ({alternative}):")
    print(f"Statistic: {stat:.5f}, p-value: {p:.5f}")
    if p < alpha:
        print(f"There is a statistically significant difference between the two groups (p < {alpha}).")
    else:
        print(f"There is no statistically significant difference between the two groups (p > {alpha}).")

    return p


def t_test(group1, group2, alpha=ALPHA, alternative='two-sided'):
    """
    Perform independent samples t-test.
    
    Parameters:
    group1: First group (Low biomarker group)
    group2: Second group (High biomarker group)
    alpha: Significance level
    alternative: 'two-sided', 'greater', or 'less'
        - 'two-sided': H1: group1 != group2
        - 'greater': H1: group1 > group2 (Low > High)
        - 'less': H1: group1 < group2 (Low < High)
    """
    stat, p = stats.ttest_ind(group1, group2, alternative=alternative)
    print(f"Independent Samples T-Test ({alternative}):")
    print(f"Statistic: {stat:.5f}, p-value: {p:.5f}")
    if p < alpha:
        print(f"There is a statistically significant difference between the two groups (p < {alpha}).")
    else:
        print(f"There is no statistically significant difference between the two groups (p > {alpha}).")
    
    return p


def ks_test(group1, group2, alpha=ALPHA, alternative='two-sided'):
    """
    Perform two-sample Kolmogorov-Smirnov test.
    
    Parameters:
    group1: First group (Low biomarker group)
    group2: Second group (High biomarker group)
    alpha: Significance level
    alternative: 'two-sided', 'greater', or 'less'
        - 'two-sided': H1: distributions are different
        - 'greater': H1: CDF(group1) > CDF(group2)
        - 'less': H1: CDF(group1) < CDF(group2)
    """
    stat, p = stats.ks_2samp(group1, group2, alternative=alternative)
    print(f"Kolmogorov-Smirnov Test ({alternative}):")
    print(f"Statistic: {stat:.5f}, p-value: {p:.5f}")
    if p < alpha:
        print(f"There is a statistically significant difference between the two groups (p < {alpha}).")
    else:
        print(f"There is no statistically significant difference between the two groups (p > {alpha}).")
    
    return p


def chi2_test(df, features, alpha=ALPHA):
    """
    Perform chi-square test for categorical data.
    
    Parameters:
    df: DataFrame containing the data
    features: List of two feature names [biomarker_cat, col]
    alpha: Significance level
    
    Note: Chi-square tests are inherently two-sided and test for association.
    """
    # Create a contingency table
    contingency_table = pd.crosstab(df[features[0]], df[features[1]])
    print("Contingency Table:")
    print(contingency_table)
    # Perform the chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("Chi-Square Test:")
    print(f"Chi2: {chi2:.5f}")
    print(f"p-value: {p:.5f}")
    if p < alpha:
        print(f"There is a statistically significant association between the two variables (p < {alpha}).")
    else:
        print(f"There is no statistically significant association between the two variables (p > {alpha}).")
    
    return p


def stat_test(data_type, group1, group2, alpha=ALPHA, alternative='two-sided'):
    """
    Perform statistical test based on data type.
    
    Parameters:
    data_type: 'Categorical', 'Continuous', or 'Ordinal'
    group1: First group (dataframe for Categorical, arrays for others)
    group2: Second group (feature list for Categorical, arrays for others)
    alpha: Significance level
    alternative: 'two-sided', 'greater', or 'less'
        - For Categorical: Always uses two-sided chi-square test (alternative parameter ignored)
        - For Continuous/Ordinal: 'two-sided', 'greater', or 'less'
    """
    if data_type == 'Categorical':
        p = chi2_test(group1, group2, alpha)
    elif data_type == 'Continuous':
        p = man_whitney(group1, group2, alpha, alternative=alternative)
        # p = ks_test(group1, group2, alpha, alternative=alternative)
        # p = t_test(group1, group2, alpha, alternative=alternative)
    elif data_type == 'Ordinal':
        p = man_whitney(group1, group2, alpha, alternative=alternative)
    else:
        print("Invalid data type.")
        
    return p