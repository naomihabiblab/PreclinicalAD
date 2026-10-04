"""
Trait association (OLS: biomarker ~ trait + Age + Gender) for risk-factor traits,
restricted to participants in the high p-tau217 group.

Uses trait lists from risk_factors, reported_risk_factors, or blood_tests
configurations (--config), with the same regression logic as trait_association.py.
Also supports cognitive / behavioral / moca_tests configurations from trait_association.py.
The high group is defined as
p-tau217 > HIGH_TAU_217_THR (settings.py), consistent with the High bin in
preprocess_data.load_and_preprocess_data for tau_217.

Writes a clustered heatmap (same style as cluster_heatmap_trait_association.py):
-log10(adjusted p) * sign(coefficient), traits clustered by similarity on the outcome biomarker.

Cohort is always high p-tau217 (p-tau217 > HIGH_TAU_217_THR); --biomarker selects the
input/output paths from trait_association.BIOMARKER_SETTINGS.

Run from the statistical_analysis directory, e.g.:
  python trait_association_high_ptau217_risk_factors.py
  python trait_association_high_ptau217_risk_factors.py --biomarker nfl
  python trait_association_high_ptau217_risk_factors.py --config reported_risk_factors
  python trait_association_high_ptau217_risk_factors.py --config blood_tests
  python trait_association_high_ptau217_risk_factors.py --config moca_tests
  python trait_association_high_ptau217_risk_factors.py --config cognitive
  python trait_association_high_ptau217_risk_factors.py --config behavioral
  python trait_association_high_ptau217_risk_factors.py --sex 0 --MCI 0
  python trait_association_high_ptau217_risk_factors.py --no_plot
"""
import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns

from settings import HIGH_TAU_217_THR, TAU_217
from trait_association import (
    BIOMARKER_SETTINGS,
    correct_multiple_testing,
    run_trait_association,
    save_csv,
)

FIGURES_DIR = "figures"

# Short names for heatmap column / clustering (aligned with cluster_heatmap_trait_association biomarker keys)
BIOMARKER_HEATMAP_COL = {
    "tau_217": "tau_217",
    "nfl": "nfl",
    "gfap": "gfap",
}


def get_traits_config(config_name, sex=None):
    """
    Load trait definitions like trait_association.main / high_vs_low_stats.get_filtered_tests.
    For reported_risk_factors, drops Menopause age unless analyzing females only (sex == 1).
    """
    if config_name == "moca_tests":
        mod = importlib.import_module("moca_tests.moca_tests_configuration")
        return list(getattr(mod, "MOCA_TESTS"))
    if config_name == "cognitive":
        mod = importlib.import_module("cognitive.cognitive_configuration")
        return list(getattr(mod, "COGNITIVE_TESTS"))
    if config_name == "behavioral":
        mod = importlib.import_module("behavioral.behavioral_configuration")
        return list(getattr(mod, "BEHAVIORAL_TESTS"))
    if config_name == "systemic_factors":
        mod = importlib.import_module("systemic_factors.systemic_factors_configuration")
        return list(getattr(mod, "SYSTEMIC_FACTORS_TESTS"))
    if config_name == "risk_factors":
        mod = importlib.import_module("risk_factors.risk_factors_configuration")
        return list(getattr(mod, "RISK_FACTORS_TESTS"))
    if config_name == "reported_risk_factors":
        mod = importlib.import_module(
            "reported_risk_factors.reported_risk_factors_configuration"
        )
        tests = list(getattr(mod, "RISK_FACTORS_TESTS"))
        if sex != 1:
            tests = [t for t in tests if t["col"] != "Menopause age"]
        return tests
    if config_name == "blood_tests":
        mod = importlib.import_module("blood_tests.blood_tests_configuration")
        return list(getattr(mod, "BLOOD_TESTS"))
    print(
        f"Unknown config_name '{config_name}'. Use 'risk_factors', 'reported_risk_factors', 'blood_tests', "
        f"'systemic_factors', 'moca_tests', 'cognitive', or 'behavioral'."
    )
    sys.exit(1)


def pval_to_star(p):
    if pd.isna(p):
        return ""
    if p < 0.005:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_risk_factors_clustermap(
    results,
    output_pdf,
    title_suffix="",
    analysis_title="Risk factors",
    heatmap_col="tau_217",
    outcome_label="p-tau217",
):
    """
    Cluster heatmap aligned with cluster_heatmap_trait_association.cluster_and_plot:
    values are -log10(adj p) * sign(coef); rows clustered on the outcome biomarker column.
    """
    df = pd.DataFrame(results)
    df = df.rename(
        columns={
            "trait": "Trait",
            "coef": "Coefficient",
            "pval": "p-value",
            "adj_pval": "adjusted p-value",
        }
    )
    min_p = 1e-300
    df["adj_pval_for_log"] = df["adjusted p-value"].clip(lower=min_p)
    df["sign_coef"] = np.sign(df["Coefficient"])
    df["heatmap_value"] = -np.log10(df["adj_pval_for_log"]) * df["sign_coef"]

    heatmap_data = (
        df.set_index("Trait")[["heatmap_value"]]
        .rename(columns={"heatmap_value": heatmap_col})
        .dropna(how="any")
    )
    if heatmap_data.shape[0] < 2:
        print(
            "Not enough traits for a clustered heatmap (need at least 2). Skipping figure."
        )
        return None

    annot_series = df.set_index("Trait")["adjusted p-value"].map(pval_to_star)
    annot_df = pd.DataFrame({heatmap_col: annot_series})
    annot_df = annot_df.reindex(index=heatmap_data.index, columns=heatmap_data.columns)

    tau_vals = heatmap_data[heatmap_col].values.reshape(-1, 1)
    dists = ssd.pdist(tau_vals)
    linkage = sch.linkage(dists, method="average")

    max_abs_val = max(
        abs(heatmap_data.min().min()), abs(heatmap_data.max().max())
    )
    vmin = max(-max_abs_val, np.log10(0.005))
    vmax = min(max_abs_val, -np.log10(0.005))

    g = sns.clustermap(
        heatmap_data,
        cmap="vlag",
        linewidths=0.5,
        annot=annot_df,
        fmt="s",
        figsize=(max(8, heatmap_data.shape[1] * 0.7), max(8, heatmap_data.shape[0] * 0.4)),
        row_linkage=linkage,
        col_cluster=False,
        vmin=vmin,
        vmax=vmax,
    )
    g.cax.set_title("-log10(adj p) * sign(coef)", fontsize=10)
    title = (
        f"{analysis_title} vs {outcome_label} (high p-tau217 cohort only)\n"
        f"Cluster heatmap (traits clustered by {heatmap_col}){title_suffix}"
    )
    g.figure.suptitle(title, y=1.02)
    g.figure.subplots_adjust(right=0.85, top=0.93)
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Heatmap saved to {output_pdf}")
    plt.close(g.figure)
    return g


def _filter_title_suffix(sex_str, mci_str):
    parts = []
    if sex_str == "_males":
        parts.append("Males")
    elif sex_str == "_females":
        parts.append("Females")
    if mci_str == "_cognitively_healthy":
        parts.append("Cognitively healthy")
    elif mci_str == "_cognitively_impaired":
        parts.append("Cognitively impaired")
    return ("\n" + ", ".join(parts)) if parts else ""


def main(
    input_file=None,
    sex=None,
    mci=None,
    no_plot=False,
    output_figure=None,
    config_name="risk_factors",
    biomarker="tau_217",
):
    if biomarker not in BIOMARKER_SETTINGS:
        print(
            f"Unknown biomarker '{biomarker}'. "
            f"Choose from: {', '.join(BIOMARKER_SETTINGS.keys())}."
        )
        sys.exit(1)
    biomarker_info = BIOMARKER_SETTINGS[biomarker]
    heatmap_col = BIOMARKER_HEATMAP_COL[biomarker]
    outcome_label = biomarker_info["col"]

    if input_file is not None and input_file != "":
        data_path = input_file
    else:
        data_path = biomarker_info.get("data_path")
        if not data_path or not os.path.exists(data_path):
            print(
                f"No input file provided and default data_path does not exist: {data_path}"
            )
            sys.exit(1)

    df = pd.read_csv(data_path)
    if TAU_217 not in df.columns:
        print(
            f"Column '{TAU_217}' not found in {data_path}; cannot define high p-tau217 cohort."
        )
        sys.exit(1)
    df[TAU_217] = pd.to_numeric(df[TAU_217], errors="coerce")
    df = df[df[TAU_217] > HIGH_TAU_217_THR].copy()  # cohort: high p-tau217
    print(
        f"High p-tau217 cohort (p-tau217 > {HIGH_TAU_217_THR}): n={len(df)}"
    )

    sex_str = ""
    if sex is not None:
        if sex == 0:
            df = df[df["Gender"] == 0]
            sex_str = "_males"
        elif sex == 1:
            df = df[df["Gender"] == 1]
            sex_str = "_females"
        else:
            print(f"Unknown sex argument '{sex}'. Use 0 (males) or 1 (females).")
            sys.exit(1)
        print(f"Filtered by sex: {sex_str[1:].capitalize()} (n={len(df)})")

    mci_str = ""
    if mci is not None:
        if mci == 0 or mci == "0":
            df = df[df["MCI"] == 0]
            mci_str = "_cognitively_healthy"
            print(f"Filtered by MCI: Cognitively Healthy (n={len(df)})")
        elif mci == 1 or mci == "1":
            df = df[df["MCI"] == 1]
            mci_str = "_cognitively_impaired"
            print(f"Filtered by MCI: Cognitively Impaired (n={len(df)})")
        else:
            print(f"Unknown MCI argument '{mci}'. Use '0' for healthy or '1' for impaired.")
            sys.exit(1)

    save_dir = biomarker_info["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    biomarker_suffix = "" if biomarker == "tau_217" else f"_{biomarker}"
    output_file = (
        save_dir
        + f"trait_association_{config_name}_high_ptau217_only{biomarker_suffix}{sex_str}{mci_str}.csv"
    )

    traits_config = get_traits_config(config_name, sex=sex)
    results = run_trait_association(df, biomarker_info["col"], traits_config)
    pvals = [r["pval"] for r in results]
    adj_pvals = correct_multiple_testing(pvals)
    for i, adj_pval in enumerate(adj_pvals):
        results[i]["adj_pval"] = adj_pval
    save_csv(results, output_file)
    print(f"Wrote {output_file}")

    if not no_plot:
        if output_figure:
            fig_path = output_figure
        else:
            fig_name = (
                f"heatmap_{config_name}_high_ptau217_only{biomarker_suffix}{sex_str}{mci_str}.pdf"
            )
            os.makedirs(FIGURES_DIR, exist_ok=True)
            fig_path = os.path.join(FIGURES_DIR, fig_name)
        title_suffix = _filter_title_suffix(sex_str, mci_str)
        analysis_title_map = {
            "risk_factors": "Risk factors",
            "reported_risk_factors": "Reported risk factors",
            "blood_tests": "Blood tests",
            "systemic_factors": "Systemic factors",
            "moca_tests": "MoCA tests",
            "cognitive": "Cognitive tests",
            "behavioral": "Behavioral tests",
        }
        analysis_title = analysis_title_map.get(config_name, config_name)
        plot_risk_factors_clustermap(
            results,
            fig_path,
            title_suffix=title_suffix,
            analysis_title=analysis_title,
            heatmap_col=heatmap_col,
            outcome_label=outcome_label,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Risk-factor trait association vs a plasma biomarker, within the high p-tau217 cohort."
    )
    parser.add_argument(
        "--biomarker",
        type=str,
        default="tau_217",
        choices=list(BIOMARKER_SETTINGS.keys()),
        help="Outcome biomarker (default: tau_217). Sets default CSV path and save_dir.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="risk_factors",
        choices=[
            "risk_factors",
            "reported_risk_factors",
            "blood_tests",
            "systemic_factors",
            "moca_tests",
            "cognitive",
            "behavioral",
        ],
        help="Trait list (default: risk_factors). Also supports moca_tests, cognitive, behavioral.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input CSV (default: from --biomarker, e.g. Tau/217/... or NFL/...).",
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
        help="Optional: 0 cognitively healthy, 1 impaired.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip saving the clustered heatmap PDF.",
    )
    parser.add_argument(
        "--output_figure",
        type=str,
        default=None,
        help="Optional path for the heatmap PDF (default: figures/heatmap_<config>_high_ptau217_only_<biomarker>*.pdf).",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all configurations (and all sex/MCI subgroup combinations).",
    )
    args = parser.parse_args()

    if args.run_all:
        config_names = [
            "risk_factors",
            "reported_risk_factors",
            "blood_tests",
            "systemic_factors",
            "moca_tests",
            "cognitive",
            "behavioral",
        ]
        sexes = [None]#, 0, 1]
        mcis = [None]#, "0", "1"]
        for config_name in config_names:
            for sex in sexes:
                for mci in mcis:
                    print(
                        f"######## Running trait association high p-tau217: "
                        f"biomarker={args.biomarker}, config={config_name}, sex={sex}, MCI={mci} ########"
                    )
                    main(
                        args.input_file,
                        sex,
                        mci,
                        no_plot=args.no_plot,
                        output_figure=None,
                        config_name=config_name,
                        biomarker=args.biomarker,
                    )
    else:
        main(
            args.input_file,
            args.sex,
            args.MCI,
            no_plot=args.no_plot,
            output_figure=args.output_figure,
            config_name=args.config,
            biomarker=args.biomarker,
        )
