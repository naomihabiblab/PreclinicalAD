"""
Within the high p-tau217 cohort only (p-tau217 > HIGH_TAU_217_THR),
compare the prevalence of binary risk factors between:
  - MCI == 1  (defined here as MOCA_score <= MCI_THR)
  - MCI == 0  (MOCA_score > MCI_THR)

For each binary risk factor:
  - Build a 2x2 contingency table (mci_status x risk_factor_present)
  - Run Chi-square test of independence
  - Compute Odds Ratio (with a small continuity correction if any cell is zero)
  - Correct p-values for multiple testing (FDR-BH), consistent with other analyses

Outputs:
  - Console: per-trait contingency table + chi2 statistic + p-value
  - CSV: per-trait counts/percentages, chi2, p, adjusted p, odds ratio
  - Figure (PDF): grouped bar chart of prevalence (%) by MCI status, colored by risk factor,
    with an annotation panel listing OR and p-values.

Run from the statistical_analysis directory, e.g.:
  python mci_prevalence_high_ptau217_risk_factors.py
  python mci_prevalence_high_ptau217_risk_factors.py --configs risk_factors reported_risk_factors genetic_risk_factors
  python mci_prevalence_high_ptau217_risk_factors.py --no_plot
"""

import argparse
import importlib
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats

from settings import HIGH_TAU_217_THR, MCI_THR, MOCA_COL, TAU_217, TAU_217_CAT


DEFAULT_SAVE_ROOT_DIRNAME = "high_p-tau217-staging_analysis"

# Resolve default paths relative to this script's directory, matching how other scripts
# in `clinical_data_analysis/statistical_analysis/` are typically executed.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default save root for this analysis (as requested).
DEFAULT_SAVE_ROOT = os.path.join(SCRIPT_DIR, DEFAULT_SAVE_ROOT_DIRNAME)
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_SAVE_ROOT, "figures")

# Local copy of the tau_217 trait-association table location used elsewhere in this repo.
# We keep this local to avoid importing `trait_association.py`, which depends on statsmodels.
TAU217_TRAIT_ASSOCIATION_DATA_PATH = os.path.join(
    SCRIPT_DIR, "Tau", "217", "trait_association_data.csv"
)
# Kept for reference; outputs for this script are saved under DEFAULT_SAVE_ROOT by default.
TAU217_TRAIT_ASSOCIATION_SAVE_DIR = os.path.join(SCRIPT_DIR, "Tau", "217", "trait_association")


def fdr_bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR adjustment.
    Implemented locally to avoid requiring statsmodels at runtime.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity from largest to smallest rank
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    out = np.empty_like(adj)
    out[order] = adj
    return out


def _to_binary_or_none(series: pd.Series) -> Optional[pd.Series]:
    """
    Try to coerce a series into a strict binary {0,1} indicator.
    Returns the mapped series (aligned to non-NA values) or None if not binary.
    """
    mapped = series.copy()

    if mapped.dtype == bool:
        mapped = mapped.astype(int)
    elif np.issubdtype(mapped.dtype, np.number):
        mapped = pd.to_numeric(mapped, errors="coerce")
    else:
        # Normalize common yes/no encodings
        str_map = {
            "yes": 1,
            "YES": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
            "in the past": 1,
            "NO": 0,
            "NO ": 0,
            "YES ": 1,
            "in _the_ past": 1,
            "in_the_past": 1,
        }
        mapped = mapped.astype(str).str.strip().str.lower().map(str_map)

    mapped = mapped.dropna()
    uniq = set(pd.unique(mapped))
    if uniq.issubset({0, 1}) and len(uniq) == 2:
        return mapped.astype(int)
    return None


def get_traits_from_configs(config_names, sex=None):
    """
    Load trait definitions from configuration modules.
    Notes:
      - For reported_risk_factors, we follow existing convention and drop 'Menopause age'
        unless analyzing females only (sex == 1). Here sex filtering is not used by default.
    """
    traits = []
    for config_name in config_names:
        if config_name == "risk_factors":
            mod = importlib.import_module("risk_factors.risk_factors_configuration")
            traits.extend(list(getattr(mod, "RISK_FACTORS_TESTS")))
        elif config_name == "reported_risk_factors":
            mod = importlib.import_module(
                "reported_risk_factors.reported_risk_factors_configuration"
            )
            tests = list(getattr(mod, "RISK_FACTORS_TESTS"))
            if sex != 1:
                tests = [t for t in tests if t.get("col") != "Menopause age"]
            traits.extend(tests)
        elif config_name == "genetic_risk_factors":
            mod = importlib.import_module(
                "genetic_risk_factors.genetic_risk_factors_configuration"
            )
            traits.extend(list(getattr(mod, "GENETIC_RISK_FACTORS_TESTS")))
        else:
            raise ValueError(
                f"Unknown config '{config_name}'. "
                f"Use: risk_factors, reported_risk_factors, genetic_risk_factors."
            )
    # de-duplicate by (label,col)
    seen = set()
    deduped = []
    for t in traits:
        key = (t.get("label"), t.get("col"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def define_high_ptau217_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Prefer the already-categorized column when available (aligned with existing preprocessing),
    # and fall back to the numeric threshold definition otherwise.
    if TAU_217_CAT in df.columns:
        # Expected convention in this repo is category strings like "High"/"Low"/...
        cat = df[TAU_217_CAT].astype(str).str.strip().str.lower()
        df = df[cat == "high"].copy()
        return df

    if TAU_217 not in df.columns:
        raise KeyError(
            f"Missing both '{TAU_217_CAT}' and '{TAU_217}' columns; cannot define high p-tau217 cohort."
        )
    df[TAU_217] = pd.to_numeric(df[TAU_217], errors="coerce")
    df = df[df[TAU_217] > HIGH_TAU_217_THR].copy()
    return df


def add_mci_status_from_moca(df: pd.DataFrame) -> pd.DataFrame:
    if MOCA_COL not in df.columns:
        raise KeyError(f"Missing column '{MOCA_COL}' needed to define MCI from MoCA.")
    df = df.copy()
    df[MOCA_COL] = pd.to_numeric(df[MOCA_COL], errors="coerce")
    df = df.dropna(subset=[MOCA_COL]).copy()
    df["mci_status"] = (df[MOCA_COL] <= MCI_THR).astype(int)
    return df


def contingency_and_stats(df: pd.DataFrame, trait_col: str, trait_label: str):
    """
    Build 2x2 contingency table for mci_status (rows) vs trait presence (cols).
    Returns:
      - contingency DataFrame with index [0,1] and columns [0,1]
      - chi2_stat, dof, p_value
      - odds_ratio (with 0.5 correction if needed)
    """
    if trait_col not in df.columns:
        return None, None, None, None, None, "column_missing"

    df_test = df[["mci_status", trait_col]].dropna().copy()
    if df_test.empty:
        return None, None, None, None, None, "no_data"

    binary = _to_binary_or_none(df_test[trait_col])
    if binary is None:
        return None, None, None, None, None, "non_binary"

    df_test = df_test.loc[binary.index].copy()
    df_test[trait_col] = binary

    # Require both MCI strata present for comparison
    if df_test["mci_status"].nunique() < 2:
        return None, None, None, None, None, "single_mci_level"
    if df_test[trait_col].nunique() < 2:
        return None, None, None, None, None, "single_trait_level"

    # Contingency table (aligned with `utilis.chi2_test` / `high_vs_low_stats.py`)
    tab = pd.crosstab(df_test["mci_status"], df_test[trait_col])
    tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    # Chi-square test of independence (same call style as `utilis.chi2_test`).
    # Note: SciPy's default uses Yates' correction for 2x2 tables (correction=True),
    # which is what `utilis.chi2_test` currently does.
    chi2_stat, p_value, dof, _expected = stats.chi2_contingency(tab.values)

    # Odds ratio: (a*d)/(b*c) with a = MCI=1 & trait=1
    a = tab.loc[1, 1]
    b = tab.loc[1, 0]
    c = tab.loc[0, 1]
    d = tab.loc[0, 0]

    # continuity correction if any cell is zero to avoid inf/0
    if min(a, b, c, d) == 0:
        a, b, c, d = (a + 0.5, b + 0.5, c + 0.5, d + 0.5)
    odds_ratio = (a * d) / (b * c)

    tab_df = pd.DataFrame(tab.values, index=[0, 1], columns=[0, 1])
    tab_df.index.name = "mci_status"
    tab_df.columns.name = f"{trait_label}_present"
    return tab_df, chi2_stat, dof, p_value, odds_ratio, None


def build_results_table(df: pd.DataFrame, traits):
    rows = []
    skipped = []

    for t in traits:
        label = t.get("label", t.get("col"))
        col = t.get("col", label)

        tab, chi2_stat, dof, p_value, odds_ratio, reason = contingency_and_stats(
            df, col, label
        )

        if reason is not None:
            skipped.append({"Feature": label, "Column": col, "Reason": reason})
            continue

        # Print requested outputs to console
        print("\n" + "=" * 80)
        print(f"Risk factor: {label} (column: {col})")
        print("Contingency table (rows=mci_status, cols=trait_present):")
        print(tab.to_string())
        print(f"Chi-square statistic: {chi2_stat:.6g}  (dof={dof})")
        print(f"p-value: {p_value:.6g}")

        n_mci0 = int(tab.loc[0].sum())
        n_mci1 = int(tab.loc[1].sum())
        n_trait_mci0 = int(tab.loc[0, 1])
        n_trait_mci1 = int(tab.loc[1, 1])

        pct_mci0 = 100.0 * n_trait_mci0 / n_mci0 if n_mci0 > 0 else np.nan
        pct_mci1 = 100.0 * n_trait_mci1 / n_mci1 if n_mci1 > 0 else np.nan

        # Cell counts for convenience (rows=mci_status, cols=trait_present)
        # d = No MCI & trait=0, c = No MCI & trait=1, b = MCI & trait=0, a = MCI & trait=1
        d = int(tab.loc[0, 0])
        c = int(tab.loc[0, 1])
        b = int(tab.loc[1, 0])
        a = int(tab.loc[1, 1])

        # Percent MCI within each trait level (for stacked bar visualization)
        n_trait0_total = d + b
        n_trait1_total = c + a
        pct_mci_within_trait0 = 100.0 * b / n_trait0_total if n_trait0_total > 0 else np.nan
        pct_mci_within_trait1 = 100.0 * a / n_trait1_total if n_trait1_total > 0 else np.nan

        rows.append(
            {
                "Feature": label,
                "Column": col,
                "n_total_mci0": n_mci0,
                "n_total_mci1": n_mci1,
                "n_trait_present_mci0": n_trait_mci0,
                "n_trait_present_mci1": n_trait_mci1,
                "pct_trait_present_mci0": pct_mci0,
                "pct_trait_present_mci1": pct_mci1,
                "n_no_mci_trait0": d,
                "n_no_mci_trait1": c,
                "n_mci_trait0": b,
                "n_mci_trait1": a,
                "pct_mci_within_trait0": pct_mci_within_trait0,
                "pct_mci_within_trait1": pct_mci_within_trait1,
                "chi2_stat": chi2_stat,
                "dof": dof,
                "p_value": p_value,
                "odds_ratio_mci1_vs_mci0": odds_ratio,
            }
        )

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df["adjusted_p_value_fdr_bh"] = fdr_bh_adjust(
            results_df["p_value"].values
        )
        results_df = results_df.sort_values("adjusted_p_value_fdr_bh").reset_index(drop=True)

    skipped_df = pd.DataFrame(skipped)
    return results_df, skipped_df


def _safe_filename(s: str) -> str:
    s = str(s).strip()
    # replace path separators and other problematic characters
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    return s[:150] if len(s) > 150 else s


def plot_grouped_prevalence(results_df: pd.DataFrame, output_path: str, title: str):
    """
    100% stacked bar chart:
      x = risk factor status (No / Yes)
      y = percentage of subjects (%)
      stacks = MCI status (No MCI vs MCI) within each risk-factor level

    Annotation:
      OR and p-values for the feature.
    """
    if results_df.empty:
        print("No results to plot (empty results table).")
        return

    # Style tuned for publication-like output
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    # Decide output mode:
    # - If output_path is a directory OR doesn't end with .pdf, save one PDF per feature inside it.
    # - If output_path ends with .pdf, treat it as a prefix and save <prefix>__<Feature>.pdf files.
    output_is_pdf = str(output_path).lower().endswith(".pdf")
    if output_is_pdf:
        out_dir = os.path.dirname(output_path) or "."
        prefix = os.path.splitext(os.path.basename(output_path))[0]
    else:
        out_dir = output_path
        prefix = None
    os.makedirs(out_dir, exist_ok=True)

    palette = {"No MCI (MoCA > 26)": "#b4cde3", "MCI (MoCA ≤ 26)": "#fbb4ae"}

    for _, r in results_df.iterrows():
        feat = r["Feature"]
        df_plot = pd.DataFrame(
            [
                {"Risk factor": "No", "mci_status_label": "No MCI (MoCA > 26)", "n": r["n_no_mci_trait0"]},
                {"Risk factor": "No", "mci_status_label": "MCI (MoCA ≤ 26)", "n": r["n_mci_trait0"]},
                {"Risk factor": "Yes", "mci_status_label": "No MCI (MoCA > 26)", "n": r["n_no_mci_trait1"]},
                {"Risk factor": "Yes", "mci_status_label": "MCI (MoCA ≤ 26)", "n": r["n_mci_trait1"]},
            ]
        )
        totals = df_plot.groupby(["Risk factor"])["n"].transform("sum")
        df_plot["Percent (%)"] = np.where(totals > 0, 100.0 * df_plot["n"] / totals, np.nan)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x_labels = ["No", "Yes"]
        x_pos = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels), dtype=float)
        for status in ["No MCI (MoCA > 26)", "MCI (MoCA ≤ 26)"]:
            heights = []
            for rf in x_labels:
                val = df_plot[
                    (df_plot["Risk factor"] == rf) & (df_plot["mci_status_label"] == status)
                ]["Percent (%)"].values
                heights.append(float(val[0]) if len(val) else 0.0)
            heights = np.array(heights, dtype=float)
            ax.bar(
                x_pos,
                heights,
                bottom=bottom,
                color=palette[status],
                edgecolor="black",
                linewidth=0.4,
                label=status,
            )
            bottom += np.nan_to_num(heights)

        ax.set_ylim(0, 100)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Percentage of subjects (%)")
        ax.set_xlabel("Risk factor")
        ax.set_title(f"{title}\n{feat}")
        ax.legend(title="MCI status", loc="upper right", frameon=True, fontsize=9, title_fontsize=10)

        annot_text = (
            f"OR={r['odds_ratio_mci1_vs_mci0']:.3g}\n"
            f"p={r['p_value']:.3g}\n"
            f"adj.p={r['adjusted_p_value_fdr_bh']:.3g}"
        )
        ax.text(
            0.98,
            0.02,
            annot_text,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.5", alpha=0.95),
        )

        fig.tight_layout()
        feat_safe = _safe_filename(feat)
        if prefix is None:
            out_file = os.path.join(out_dir, f"barplot__{feat_safe}__high_ptau217_only.pdf")
        else:
            out_file = os.path.join(out_dir, f"{prefix}__{feat_safe}.pdf")
        fig.savefig(out_file, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"Saved figure: {out_file}")


def main(
    configs,
    input_file=None,
    output_csv=None,
    output_skipped_csv=None,
    output_figure=None,
    no_plot=False,
    sex=None,
    mci=None,
):
    # Default input: use same table as tau_217 trait association data
    data_path = input_file if input_file else TAU217_TRAIT_ASSOCIATION_DATA_PATH
    if not os.path.exists(data_path):
        print(f"Input CSV not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df = define_high_ptau217_cohort(df)
    df = add_mci_status_from_moca(df)

    # Optional subgroup filters (aligned with other scripts' CLI conventions).
    if sex is not None:
        if "Gender" not in df.columns:
            raise KeyError("Missing column 'Gender' needed for --sex filtering.")
        if sex == 0:
            df = df[df["Gender"] == 0].copy()
        elif sex == 1:
            df = df[df["Gender"] == 1].copy()
        else:
            raise ValueError("Unknown --sex value. Use 0 (males) or 1 (females).")

    if mci is not None:
        if mci == 0 or mci == "0":
            df = df[df["mci_status"] == 0].copy()
        elif mci == 1 or mci == "1":
            df = df[df["mci_status"] == 1].copy()
        else:
            raise ValueError("Unknown --MCI value. Use '0' or '1'.")

    print(
        f"High p-tau217 cohort (p-tau217 > {HIGH_TAU_217_THR}) with MoCA available: n={len(df)}"
    )

    traits = get_traits_from_configs(configs)
    results_df, skipped_df = build_results_table(df, traits)

    # Output locations (default: high_p-tau217-staging_analysis/)
    save_dir = DEFAULT_SAVE_ROOT
    os.makedirs(save_dir, exist_ok=True)

    if output_csv is None:
        output_csv = os.path.join(
            save_dir,
            "mci_prevalence_risk_factors_high_ptau217_only.csv",
        )
    if output_skipped_csv is None:
        output_skipped_csv = os.path.join(
            save_dir,
            "mci_prevalence_risk_factors_high_ptau217_only_skipped.csv",
        )

    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results CSV: {output_csv}")
    if not skipped_df.empty:
        skipped_df.to_csv(output_skipped_csv, index=False)
        print(f"Saved skipped CSV: {output_skipped_csv}")

    if not no_plot:
        os.makedirs(DEFAULT_FIGURES_DIR, exist_ok=True)
        if output_figure is None:
            # Default: write one PDF per feature into the figures directory.
            output_figure = DEFAULT_FIGURES_DIR
        title = "MCI distribution within risk-factor Yes/No (high p-tau217 cohort only)"
        plot_grouped_prevalence(results_df, output_figure, title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chi-square prevalence comparison: MCI (MoCA<=26) vs risk factors, within high p-tau217 cohort."
    )
    parser.add_argument(
        "--biomarker",
        type=str,
        default="tau_217",
        help="Kept for CLI compatibility; cohort is always high p-tau217.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="risk_factors",
        choices=["risk_factors", "reported_risk_factors", "genetic_risk_factors"],
        help="Risk-factor configuration to analyze (default: risk_factors).",
    )
    # Backwards-compatible alias (if you want multiple configs at once).
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        choices=["genetic_risk_factors", "reported_risk_factors", "risk_factors"],
        help="Alias: analyze multiple configurations in one run (overrides --config).",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Optional input CSV. Default: Tau/217/trait_association_data.csv (script-relative).",
    )
    parser.add_argument(
        "--sex",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional: 0 males, 1 females.",
    )
    parser.add_argument(
        "--MCI",
        type=str,
        choices=["0", "1"],
        default=None,
        help="Optional: 0 cognitively healthy, 1 impaired (based on MoCA<=26 definition in this script).",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_skipped_csv", type=str, default=None)
    parser.add_argument(
        "--output_figure",
        type=str,
        default=None,
        help="Optional path for the PDF barplot (default: figures/barplot_*.pdf).",
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the PDF figure.")
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all configurations (risk_factors, reported_risk_factors, genetic_risk_factors).",
    )
    args = parser.parse_args()

    # Match the other script's UX: default is a single config unless explicitly running all.
    if args.run_all:
        configs = ["risk_factors", "reported_risk_factors", "genetic_risk_factors"]
    else:
        configs = args.configs if args.configs is not None else [args.config]

    main(
        configs=configs,
        input_file=args.input_file,
        output_csv=args.output_csv,
        output_skipped_csv=args.output_skipped_csv,
        output_figure=args.output_figure,
        no_plot=args.no_plot,
        sex=args.sex,
        mci=args.MCI,
    )

