import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import numpy as np

CSV_PATH = '/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/clinical_data_analysis/statistical_analysis/all_trait_association_results.csv'
CSV_PATH = './all_trait_association_results.csv'
FIGURES_DIR = 'figures'
CONFIGS = [
    'behavioral',
    'cognitive',
    'moca_tests',
    'blood_tests',
    'categorical_blood_tests',
    'risk_factors',
    'genetic_risk_factors',
    'reported_risk_factors'

]

def parse_args():
    parser = argparse.ArgumentParser(description='Create a cluster heatmap from all_trait_association_results.csv')
    parser.add_argument('--config', type=str, default=None, help='Config to filter (e.g., behavioral, cognitive, etc.)')
    parser.add_argument('--sex', type=str, default=None, help='Sex to filter (All, Males, Females)')
    parser.add_argument('--mci_status', type=str, default=None, help='MCI_Status to filter (All, Cognitively Healthy, Cognitively Impaired)')
    parser.add_argument('--output', type=str, default=None, help='Output pdf file (optional, will be placed in figures/)')
    parser.add_argument('--run_all_configs', action='store_true', help='Run and save heatmaps for all configs')
    parser.add_argument('--stacked', action='store_true', help='Stack all configs vertically in one figure')
    return parser.parse_args()

def pval_to_star(p):
    if p < 0.005:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def cluster_and_plot(df, config, sex, mci_status, output=None, show=False, ax=None, title_prefix=None):
    # Filter by arguments (except biomarker)
    if config:
        df = df[df['Config'] == config]
    
    # Default behavior: if no sex/mci_status specified, use only "All" values
    # This prevents mixing data from different subgroups when not explicitly requested
    if sex is None:
        df = df[df['Sex'] == 'All']
    else:
        df = df[df['Sex'] == sex]
        
    if mci_status is None:
        df = df[df['MCI_Status'] == 'All']
    else:
        df = df[df['MCI_Status'] == mci_status]
        
    if df.empty:
        print(f'No data matches the selected filters for config={config}, sex={sex}, mci_status={mci_status}.')
        return None
    
    # Debug: Print filter info
    print(f"Filters applied: Config={config}, Sex={sex or 'All (default)'}, MCI_Status={mci_status or 'All (default)'}")
    print(f"Data shape after filtering: {df.shape}")
    
    # Calculate -log10(adjusted p-value) * sign(coefficient)
    min_p = 1e-300
    df['adj_pval_for_log'] = df['adjusted p-value'].clip(lower=min_p)
    df['sign_coef'] = np.sign(df['Coefficient'])
    df['heatmap_value'] = -np.log10(df['adj_pval_for_log']) * df['sign_coef']
    # Pivot: Traits as rows, Biomarker as columns, values as heatmap_value (aggregate by mean)
    heatmap_data = df.groupby(['Trait', 'Biomarker'])['heatmap_value'].mean().unstack()
    # Cluster based on tau_217 values only
    if 'tau_217' not in heatmap_data.columns:
        print('tau_217 not found in Biomarker column. Cannot cluster based on tau_217.')
        return None
    clustering_vector = heatmap_data['tau_217'].dropna()
    heatmap_data = heatmap_data.loc[clustering_vector.index]
    if heatmap_data.shape[0] < 2 or heatmap_data.shape[1] < 2:
        print(f'Not enough data to create a clustered heatmap for config={config} (need at least 2 rows and 2 columns).')
        return None
    # Create annotation DataFrame for asterisks (aggregate by min p-value)
    # Only use the filtered data for annotations to avoid cross-contamination from other groups
    # This ensures that asterisks only appear for significant p-values within the specified subgroup
    annot_df = df.groupby(['Trait', 'Biomarker'])['adjusted p-value'].min().unstack().applymap(pval_to_star)
    annot_df = annot_df.loc[heatmap_data.index, heatmap_data.columns]
    
    tau217_values = heatmap_data['tau_217'].values.reshape(-1, 1)
    if len(tau217_values) < 2:
        print(f'Not enough traits with tau_217 data to cluster for config={config}.')
        return None
    dists = ssd.pdist(tau217_values)
    linkage = sch.linkage(dists, method='average')
    # Calculate symmetric color scale limits centered around 0
    max_abs_val = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    vmin, vmax = max(-max_abs_val, np.log10(0.005)), min(max_abs_val, -np.log10(0.005))
    
    g = sns.clustermap(
        heatmap_data,
        cmap='vlag',
        linewidths=0.5,
        annot=annot_df,
        fmt='s',
        figsize=(max(8, heatmap_data.shape[1]*0.7), max(8, heatmap_data.shape[0]*0.4)),
        row_linkage=linkage,
        col_cluster=False,
        vmin=vmin,
        vmax=vmax
    )
    g.cax.set_title('-log10(adj p) * sign(coef)', fontsize=10)
    title = f'{title_prefix + ": " if title_prefix else ""}Cluster Heatmap (traits clustered by tau_217)\nConfig: {config}'
    g.figure.suptitle(title, y=1.02)
    g.figure.subplots_adjust(right=0.85, top=0.93)
    if output:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(FIGURES_DIR, output)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f'Heatmap saved to {out_path}')
    if show:
        plt.show()
    plt.close(g.figure)
    return g, heatmap_data, annot_df

def main():
    args = parse_args()
    df = pd.read_csv(CSV_PATH)
    if args.run_all_configs or args.stacked:
        # Loop over all configs
        all_heatmaps = []
        for config in CONFIGS:
            result = cluster_and_plot(
                df.copy(),
                config=config,
                sex=args.sex,
                mci_status=args.mci_status,
                output=f'heatmap_{config}.pdf' if not args.stacked else None,
                show=False,
                ax=None,
                title_prefix=None
            )
            if result is not None:
                g, heatmap_data, annot_df = result
                all_heatmaps.append((config, heatmap_data, annot_df, g))
        if args.stacked and all_heatmaps:
            # Stack all heatmaps vertically
            n = len(all_heatmaps)
            fig_height = sum([max(8, h[1].shape[0]*0.4) for h in all_heatmaps])
            fig, axes = plt.subplots(n, 1, figsize=(max(8, all_heatmaps[0][1].shape[1]*0.7), fig_height), squeeze=False)
            # Calculate symmetric color scale limits for all heatmaps
            all_max_abs_val = 0
            for config, heatmap_data, annot_df, g in all_heatmaps:
                max_abs_val = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
                all_max_abs_val = max(all_max_abs_val, max_abs_val)
            
            vmin, vmax = -all_max_abs_val, all_max_abs_val
            
            for i, (config, heatmap_data, annot_df, g) in enumerate(all_heatmaps):
                ax = axes[i, 0]
                sns.heatmap(
                    heatmap_data,
                    cmap='vlag',
                    linewidths=0.5,
                    annot=annot_df,
                    fmt='s',
                    ax=ax,
                    cbar=(i == n-1),
                    cbar_kws={'label': '-log10(adj p) * sign(coef)'} if i == n-1 else None,
                    vmin=vmin,
                    vmax=vmax
                )
                ax.set_title(f'{config}', fontsize=12, loc='left')
                if i < n-1:
                    ax.set_xlabel('')
                else:
                    ax.set_xlabel('Biomarker')
                if i == n//2:
                    ax.set_ylabel('Trait')
                else:
                    ax.set_ylabel('')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.suptitle('Stacked Cluster Heatmaps (traits clustered by tau_217)', y=1.02)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            os.makedirs(FIGURES_DIR, exist_ok=True)
            out_path = os.path.join(FIGURES_DIR, 'stacked_heatmaps.pdf')
            plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
            print(f'Stacked heatmap saved to {out_path}')
            plt.show()
    else:
        # Single config
        output_name = args.output or (f'heatmap_{args.config or "all"}.pdf')
        cluster_and_plot(
            df,
            config=args.config,
            sex=args.sex,
            mci_status=args.mci_status,
            output=output_name,
            show=True,
            ax=None,
            title_prefix=None
        )

if __name__ == '__main__':
    main() 