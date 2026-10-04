"""
LogisticRegression coefficient explainability.

Loads a saved sklearn LogisticRegression model, exports a full coefficient
ranking CSV, and produces importance plots styled like the ALE/SHAP outputs.
"""

import argparse
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.grid'] = False
sns.set_palette('husl')

DEFAULT_BAD_COLOR = '#fbb4ae'   # red  — high p-tau217 (class 1)
DEFAULT_GOOD_COLOR = '#b4cde3'  # blue — low p-tau217  (class 0)

DEFAULT_SAVED_MODELS_DIR = (
    '/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/'
    'biomarker_prediction/figures/classification'
)
DEFAULT_OUTPUT_DIR = (
    '/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/'
    'biomarker_prediction/figures/classification/explainability/LogisticRegression'
)


def _mci_nci_suffix(mci: bool, nci: bool) -> str:
    return f'MCI={mci}_NCI={nci}'


def load_model_and_clinical_data(saved_models_dir, model_name, mci=False, nci=False):
    """Load LogisticRegression model and clinical_data with feature names."""
    suffix = _mci_nci_suffix(mci, nci)
    model_path = os.path.join(saved_models_dir, 'saved_models', f'{model_name}_{suffix}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if not isinstance(model, LogisticRegression):
        raise TypeError(f'Expected LogisticRegression, got {type(model).__name__}')

    clinical_data_path = os.path.join(
        saved_models_dir, 'saved_models', f'individual_models_clinical_data_{suffix}.pkl'
    )
    if not os.path.exists(clinical_data_path):
        raise FileNotFoundError(f'Clinical_data not found: {clinical_data_path}')

    with open(clinical_data_path, 'rb') as f:
        clinical_data = pickle.load(f)

    if 'feature_names' not in clinical_data:
        raise KeyError(f"'feature_names' missing from clinical_data: {clinical_data_path}")

    feature_names = list(clinical_data['feature_names'])
    coef = model.coef_.ravel()
    if len(feature_names) != len(coef):
        raise ValueError(
            f'Feature count mismatch: {len(feature_names)} names vs {len(coef)} coefficients'
        )

    return model, feature_names, coef


def build_coefficient_table(feature_names, coef):
    """Build sorted coefficient DataFrame with rank."""
    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef),
    })
    df = df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df


def save_intercept(model, output_dir):
    """Save model intercept to a one-row CSV."""
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    intercept_path = os.path.join(output_dir, 'lr_intercept.csv')
    pd.DataFrame({'intercept': [intercept]}).to_csv(intercept_path, index=False)
    return intercept_path


def plot_abs_importance(importance_df_top, output_dir, n_plot):
    """Horizontal bar plot of |coefficient| (ALE-style)."""
    plt.figure(figsize=(10, max(8, len(importance_df_top) * 0.3)))
    sns.barplot(
        data=importance_df_top,
        y='feature',
        x='abs_coefficient',
        palette='viridis',
        rasterized=False,
    )
    plt.xlabel('|Coefficient| (standardized features)')
    plt.ylabel('Feature')
    plt.title(
        'Feature Importance Based on |Logistic Regression Coefficient| (top %d)\n'
        '(Higher |coef| = stronger linear effect on scaled inputs)' % n_plot
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'lr_feature_importance.pdf')
    plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_signed_coefficients(importance_df_top, output_dir, n_plot):
    """Horizontal bar plot of signed coefficients, colored by direction."""
    colors = [
        DEFAULT_BAD_COLOR if c > 0 else DEFAULT_GOOD_COLOR
        for c in importance_df_top['coefficient']
    ]
    plt.figure(figsize=(10, max(8, len(importance_df_top) * 0.3)))
    ax = sns.barplot(
        data=importance_df_top,
        y='feature',
        x='coefficient',
        palette=colors,
        rasterized=False,
    )
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, rasterized=False)
    plt.xlabel('Coefficient (standardized features)')
    plt.ylabel('Feature')
    plt.title(
        'Logistic Regression Coefficients (top %d by |coef|)\n'
        'Positive → higher log-odds of high p-tau217 (class 1); '
        'negative → lower log-odds' % n_plot
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'lr_coefficients_signed.pdf')
    plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    return path


def run(args):
    """Load model, export CSV, and create plots."""
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 80)
    print('LOGISTIC REGRESSION COEFFICIENT EXPLAINABILITY')
    print('=' * 80)

    model, feature_names, coef = load_model_and_clinical_data(
        args.saved_models_dir, args.model_name, mci=args.mci, nci=args.nci
    )
    print(f'  Loaded model with {len(feature_names)} features')

    importance_df = build_coefficient_table(feature_names, coef)
    csv_path = os.path.join(args.output_dir, 'lr_coefficients.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f'  Saved coefficient table to: {csv_path}')

    intercept_path = save_intercept(model, args.output_dir)
    print(f'  Saved intercept to: {intercept_path}')

    n_plot = min(args.top_n_plot, len(importance_df))
    # Same order as ALE/SHAP plots: descending importance, top feature first in data
    importance_df_top = importance_df.head(n_plot).copy()

    abs_plot_path = plot_abs_importance(importance_df_top, args.output_dir, n_plot)
    print(f'  Saved |coefficient| importance plot to: {abs_plot_path}')

    signed_plot_path = plot_signed_coefficients(importance_df_top, args.output_dir, n_plot)
    print(f'  Saved signed coefficient plot to: {signed_plot_path}')

    print('=' * 80)
    print('COMPLETE')
    print('=' * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export and plot LogisticRegression coefficients from a saved model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python logistic_regression_coefficients.py \\
    --saved_models_dir figures/classification \\
    --output_dir figures/classification/explainability/LogisticRegression
        """,
    )
    parser.add_argument(
        '--saved_models_dir',
        type=str,
        default=DEFAULT_SAVED_MODELS_DIR,
        help='Directory containing saved_models subdirectory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for CSV and PDF outputs',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='LogisticRegression',
        help='Base name of the saved model file',
    )
    parser.add_argument('--mci', action='store_true', help='Use MCI=True model suffix')
    parser.add_argument('--nci', action='store_true', help='Use NCI=True model suffix')
    parser.add_argument(
        '--top_n_plot',
        type=int,
        default=15,
        help='Top N features in plots; full ranking always in CSV',
    )
    args = parser.parse_args()
    if args.mci and args.nci:
        raise ValueError('Cannot specify both --mci and --nci.')
    return args


if __name__ == '__main__':
    run(parse_args())
