import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm

from high_vs_low_stats import BIOMARKER_SETTINGS
from settings import TAU_217_CAT, GFAP_CAT, NFL_CAT
from utilis import stat_test
from visualizations.statistics_visualization_utils import plot_multigroup_pvalue_heatmap


BIOMARKER_CAT_COL = {
    "tau_217": TAU_217_CAT,
    "gfap": GFAP_CAT,
    "nfl": NFL_CAT,
}

DEFAULT_BIOMARKERS = ["tau_217", "gfap", "nfl"]

def _sex_mci_suffix(sex=None, mci=None):
    sex_str = ""
    if sex == "m":
        sex_str = "_males"
    elif sex == "f":
        sex_str = "_females"

    mci_str = ""
    if mci == "0":
        mci_str = "_cognitively_healthy"
    elif mci == "1":
        mci_str = "_cognitively_impaired"
    return sex_str, mci_str


def get_filtered_tests(config_name, sex=None):
    if config_name == "risk_factors":
        config_module = importlib.import_module("risk_factors.risk_factors_configuration")
        tests = list(getattr(config_module, "RISK_FACTORS_TESTS"))
    elif config_name == "reported_risk_factors":
        config_module = importlib.import_module(
            "reported_risk_factors.reported_risk_factors_configuration"
        )
        tests = list(getattr(config_module, "RISK_FACTORS_TESTS"))
        # Keep same convention as high_vs_low_stats.py
        if sex != "f":
            tests = [test for test in tests if test["col"] != "Menopause age"]
    else:
        print(
            f"Unknown config_name '{config_name}'. Use 'risk_factors' or 'reported_risk_factors'."
        )
        sys.exit(1)
    return tests


def _to_binary_or_none(series):
    mapped = series.copy()

    if mapped.dtype == bool:
        mapped = mapped.astype(int)
    elif np.issubdtype(mapped.dtype, np.number):
        mapped = pd.to_numeric(mapped, errors="coerce")
    else:
        str_map = {
            "yes": 1,
            "YES": 1,
            "in the past": 1,
            "no": 0,
            "NO": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
        mapped = (
            mapped.astype(str)
            .str.strip()
            .str.lower()
            .map(str_map)
        )

    mapped = mapped.dropna()
    unique_vals = set(pd.unique(mapped))
    if unique_vals.issubset({0, 1}) and len(unique_vals) == 2:
        return mapped.astype(int)
    return None


def run_binary_trait_tests(df_high_group, tests_to_run, biomarker_col):
    rows = []
    skipped = []

    for test in tests_to_run:
        col = test["col"]
        label = test["label"]
        alternative = test.get("alternative", "two-sided")

        if col not in df_high_group.columns:
            skipped.append((label, col, "column_missing"))
            continue

        df_test = df_high_group[[biomarker_col, col]].dropna().copy()
        if df_test.empty:
            skipped.append((label, col, "no_data"))
            continue

        df_test[biomarker_col] = pd.to_numeric(df_test[biomarker_col], errors="coerce")
        df_test = df_test.dropna(subset=[biomarker_col])
        if df_test.empty:
            skipped.append((label, col, "biomarker_not_numeric"))
            continue

        binary_trait = _to_binary_or_none(df_test[col])
        if binary_trait is None:
            skipped.append((label, col, "non_binary"))
            continue

        # Re-align after binary conversion/dropna
        df_test = df_test.loc[binary_trait.index].copy()
        df_test[col] = binary_trait

        group0 = df_test[df_test[col] == 0][biomarker_col]
        group1 = df_test[df_test[col] == 1][biomarker_col]
        if len(group0) == 0 or len(group1) == 0:
            skipped.append((label, col, "single_level_after_cleaning"))
            continue

        p_val = stat_test("Continuous", group0, group1, alternative=alternative)
        rows.append(
            {
                "Feature": label,
                "Column": col,
                "n_trait_0": len(group0),
                "n_trait_1": len(group1),
                "mean_biomarker_trait_0": group0.mean(),
                "mean_biomarker_trait_1": group1.mean(),
                "p-value": p_val,
            }
        )

    return rows, skipped


def _build_paths(save_dir, config_name, split_biomarker_type, sex_str="", mci_str=""):
    split_suffix = f"_high_{split_biomarker_type}"
    out_file = os.path.join(
        save_dir,
        f"high_group_binary_trait_stats_{config_name}{split_suffix}{sex_str}{mci_str}.csv",
    )
    skipped_file = os.path.join(
        save_dir,
        f"high_group_binary_trait_stats_{config_name}{split_suffix}{sex_str}{mci_str}_skipped.csv",
    )
    return out_file, skipped_file


def get_high_group_ids(split_biomarker_type):
    """
    Build cohort IDs using the split biomarker table/category only.
    This is the source of truth for the high-group split in plotting mode.
    """
    split_info = BIOMARKER_SETTINGS[split_biomarker_type]
    split_cat_col = BIOMARKER_CAT_COL[split_biomarker_type]
    split_path = split_info["data_path"]

    df_split = pd.read_csv(split_path)
    if "ID" not in df_split.columns:
        print(f"Missing 'ID' column in split table: {split_path}")
        sys.exit(1)
    if split_cat_col not in df_split.columns:
        print(
            f"Missing split biomarker category column '{split_cat_col}' in split table: {split_path}."
        )
        sys.exit(1)

    high_ids = set(df_split.loc[df_split[split_cat_col] == "High", "ID"].dropna().astype(str))
    return high_ids


def run_single_analysis(
    biomarker_type,
    config_name,
    sex=None,
    mci=None,
    split_biomarker_type=None,
):
    if split_biomarker_type is None:
        split_biomarker_type = biomarker_type

    biomarker = BIOMARKER_SETTINGS[biomarker_type]
    biomarker_col = {"tau_217": "p-tau217", "gfap": "GFAP", "nfl": "NFL"}[
        biomarker_type
    ]
    tests = get_filtered_tests(config_name, sex=sex)

    save_dir = os.path.join(biomarker["save_dir"], config_name)
    os.makedirs(save_dir, exist_ok=True)

    # Read outcome-biomarker table and keep only IDs belonging to the split biomarker HIGH group.
    df = pd.read_csv(biomarker["data_path"])
    if "ID" not in df.columns:
        print(f"Missing 'ID' column in outcome table: {biomarker['data_path']}.")
        sys.exit(1)

    high_ids = get_high_group_ids(split_biomarker_type)
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"].isin(high_ids)].copy()
    print(
        f"Selected high {split_biomarker_type} group only (outcome biomarker: {biomarker_type}): n={len(df)}"
    )

    sex_str, mci_str = _sex_mci_suffix(sex=sex, mci=mci)
    if sex is not None:
        if sex.lower() == "m":
            df = df[df["Gender"] == 0]
        elif sex.lower() == "f":
            df = df[df["Gender"] == 1]
        else:
            print(f"Unknown sex argument '{sex}'. Use 'm' or 'f'.")
            sys.exit(1)
        print(f"Filtered by sex: {sex_str[1:].capitalize()} (n={len(df)})")

    if mci is not None:
        if mci in (0, "0"):
            df = df[df["MCI"] == 0]
            print(f"Filtered by MCI: Cognitively Healthy (n={len(df)})")
        elif mci in (1, "1"):
            df = df[df["MCI"] == 1]
            print(f"Filtered by MCI: Cognitively Impaired (n={len(df)})")
        else:
            print(f"Unknown MCI argument '{mci}'. Use '0' for healthy or '1' for impaired.")
            sys.exit(1)

    rows, skipped = run_binary_trait_tests(df, tests, biomarker_col)
    out_file, skipped_file = _build_paths(
        save_dir, config_name, split_biomarker_type, sex_str, mci_str
    )

    if rows:
        p_values = [r["p-value"] for r in rows]
        _, adj_pvals, _, _ = smm.multipletests(p_values, alpha=0.05, method="fdr_bh")
        for i, adj in enumerate(adj_pvals):
            rows[i]["adjusted p-value"] = adj
        out_df = pd.DataFrame(rows)
        out_df = out_df[
            [
                "Feature",
                "Column",
                "n_trait_0",
                "n_trait_1",
                "mean_biomarker_trait_0",
                "mean_biomarker_trait_1",
                "p-value",
                "adjusted p-value",
            ]
        ]
        out_df.to_csv(out_file, index=False)
        print(f"Saved {out_file}")
    else:
        pd.DataFrame(
            columns=[
                "Feature",
                "Column",
                "n_trait_0",
                "n_trait_1",
                "mean_biomarker_trait_0",
                "mean_biomarker_trait_1",
                "p-value",
                "adjusted p-value",
            ]
        ).to_csv(out_file, index=False)
        print(f"No binary traits available. Wrote empty results file: {out_file}")

    if skipped:
        pd.DataFrame(skipped, columns=["Feature", "Column", "Reason"]).to_csv(
            skipped_file, index=False
        )
        print(f"Saved skipped-traits report: {skipped_file}")


def plot_heatmap_from_outputs(
    config_name,
    biomarker_types=None,
    sex=None,
    mci=None,
    output=None,
    split_biomarker_type="tau_217",
):
    if biomarker_types is None or len(biomarker_types) == 0:
        biomarker_types = DEFAULT_BIOMARKERS
    bad = [b for b in biomarker_types if b not in BIOMARKER_SETTINGS]
    if bad:
        print(f"Unknown biomarker(s): {bad}. Valid: {list(BIOMARKER_SETTINGS.keys())}")
        sys.exit(1)

    sex_str, mci_str = _sex_mci_suffix(sex=sex, mci=mci)
    rows = []
    missing = []
    for biomarker in biomarker_types:
        save_dir = os.path.join(BIOMARKER_SETTINGS[biomarker]["save_dir"], config_name)
        out_file, _ = _build_paths(
            save_dir, config_name, split_biomarker_type, sex_str, mci_str
        )
        if not os.path.exists(out_file):
            missing.append(out_file)
            continue
        df = pd.read_csv(out_file)
        if df.empty:
            continue
        part = df[["Feature", "adjusted p-value"]].copy()
        part["Biomarker"] = biomarker
        rows.append(part)

    if not rows:
        print("No output CSVs found to plot.")
        if missing:
            print("Missing files:")
            for p in missing:
                print(f"  - {p}")
        sys.exit(1)

    df_plot = pd.concat(rows, axis=0, ignore_index=True)
    if output is None:
        os.makedirs("figures", exist_ok=True)
        output = os.path.join(
            "figures",
            f"heatmap_high_group_binary_trait_stats_{config_name}_high_{split_biomarker_type}{sex_str}{mci_str}.pdf",
        )

    subgroup_bits = []
    if sex == "m":
        subgroup_bits.append("Males")
    elif sex == "f":
        subgroup_bits.append("Females")
    if mci == "0":
        subgroup_bits.append("Cognitively healthy")
    elif mci == "1":
        subgroup_bits.append("Cognitively impaired")
    subgroup_suffix = f" ({', '.join(subgroup_bits)})" if subgroup_bits else ""

    plot_multigroup_pvalue_heatmap(
        df_pvals=df_plot,
        group_col="Biomarker",
        groups=list(biomarker_types),
        feature_col="Feature",
        pval_col="adjusted p-value",
        title=(
            f"High-{split_biomarker_type} binary-trait stats: {config_name}"
            f"{subgroup_suffix}"
        ),
        save_path=output,
    )
    print(f"Saved heatmap: {output}")
    if missing:
        print(f"Note: biomarkers {', '.join(missing)} files were missing and skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Within selected biomarker HIGH group only, compare biomarker values between "
            "trait=0 vs trait=1 for binary traits (non-binary traits are skipped)."
        )
    )
    parser.add_argument(
        "--biomarker_type",
        type=str,
        required=True,
        help="Biomarker type",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        choices=["risk_factors", "reported_risk_factors"],
        help="Configuration name",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all configurations (and all sex/MCI subgroup combinations).",
    )
    parser.add_argument("--sex", type=str, choices=["m", "f"], default=None)
    parser.add_argument("--MCI", type=str, choices=["0", "1"], default=None)
    parser.add_argument(
        "--plot_heatmap",
        action="store_true",
        help="Plot heatmap from existing output CSVs of this script.",
    )
    parser.add_argument(
        "--plot_biomarker_types",
        type=str,
        nargs="+",
        choices=DEFAULT_BIOMARKERS,
        default=DEFAULT_BIOMARKERS,
        help="Biomarkers to include in --plot_heatmap.",
    )
    parser.add_argument(
        "--biomarker_types",
        dest="plot_biomarker_types",
        type=str,
        nargs="+",
        choices=DEFAULT_BIOMARKERS,
        help="Alias for --plot_biomarker_types.",
    )
    parser.add_argument(
        "--output_figure",
        type=str,
        default=None,
        help="Custom PDF path for --plot_heatmap.",
    )
    args = parser.parse_args()

    if args.run_all and args.plot_heatmap:
        print("Error: --run_all cannot be combined with --plot_heatmap.")
        sys.exit(1)

    if args.plot_heatmap:
        # In plotting mode:
        #   - biomarker_type => cohort split biomarker (High group)
        #   - plot_biomarker_types => outcome biomarkers used for stats/plot columns
        run_order = list(dict.fromkeys(list(args.plot_biomarker_types)))
        for outcome_biomarker_type in run_order:
            run_single_analysis(
                outcome_biomarker_type,
                args.config_name,
                args.sex,
                args.MCI,
                split_biomarker_type=args.biomarker_type,
            )
        plot_heatmap_from_outputs(
            config_name=args.config_name,
            biomarker_types=args.plot_biomarker_types,
            sex=args.sex,
            mci=args.MCI,
            output=args.output_figure,
            split_biomarker_type=args.biomarker_type,
        )
    else:
        if args.run_all:
            config_names = ["risk_factors", "reported_risk_factors"]
            sexes = [None, "m", "f"]
            mcis = [None, "0", "1"]
            for config_name in config_names:
                for sex in sexes:
                    for mci in mcis:
                        print(
                            f"######## Running high-group binary trait stats: "
                            f"biomarker_type={args.biomarker_type}, config={config_name}, sex={sex}, MCI={mci} ########"
                        )
                        run_single_analysis(args.biomarker_type, config_name, sex, mci)
        else:
            run_single_analysis(args.biomarker_type, args.config_name, args.sex, args.MCI)
