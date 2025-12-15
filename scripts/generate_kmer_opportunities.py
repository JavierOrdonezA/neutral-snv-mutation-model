
# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pathlib
from typing import Dict, Iterable, Tuple, Mapping, Union, List
from pathlib import Path
# %%
REPO_ROOT = Path(__file__).resolve().parents[1]

data_dir = REPO_ROOT / "data"

maf_path = (data_dir / "example_maf"
            / "MC3_TCGA_bed_with_maf38_filtered_hg38_annotated.parquet")


def compute_mutation_opportunities(
    df: pd.DataFrame,
    kmer_col: str,
    mutated_base_col: str = "mutated_base",
    out_col: str = "nchance",
) -> pd.DataFrame:
    """
    Compute mutation opportunities by k-mer and mutated base.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing k-mer context and mutated base.
    kmer_col : str
        Column name for the k-mer context (e.g. '5_mer', '3_mer').
    mutated_base_col : str, default='mutated_base'
        Column name for the mutated base.
    out_col : str, default='nchance'
        Name of the output column with counts.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [kmer_col, mutated_base_col, out_col],
        where out_col is the number of mutation opportunities.
    """
    df_group = (df
                .groupby([kmer_col, mutated_base_col], as_index=False)
                .size()
                .rename(columns={"size": out_col}))
    return df_group


def build_density_tables(df_study: pd.DataFrame,
                         df_sim: pd.DataFrame) -> dict:
    """
    Compute k-mer density tables for both 5-mer and 3-mer contexts.

    Parameters
    ----------
    df_study : pd.DataFrame
        Mutations observed for one study (real data).
    df_sim : pd.DataFrame
        Simulated mutations.

    Returns
    -------
    dict:
        {
            "5mer": pd.DataFrame,  # with columns [k_mer, mutated_base, nmut, nchance, density]
            "3mer": pd.DataFrame   # same as above for 3-mers
        }
    """
    # --- 5-mer
    # df_study = tcga_maf_h38_filtered_df
    obs_5 = (
        df_study
        .groupby(['5_mer_h38', 'AltAllele_stranded'], as_index=False)
        .size()
        .rename(columns={
            '5_mer_h38': 'k_mer',
            'AltAllele_stranded': 'mutated_base',
            'size': 'nmut'
        })
    )
    opp_5 = (
        df_sim
        .groupby(['5_mer', 'mutated_base'], as_index=False)
        .size()
        .rename(columns={'size': 'nchance',
                         '5_mer': 'k_mer', })
    )

    merged_5 = (
        obs_5.merge(opp_5, on=['k_mer', 'mutated_base'], how='outer')
        .fillna({'nmut': 0, 'nchance': 0})
    )

    merged_5['density'] = np.where(
        merged_5['nchance'] > 0,
        merged_5['nmut'] / merged_5['nchance'],
        np.nan
    )

    # --- 3-mer
    obs_3 = (
        df_obs
        .groupby(['3_mer_h38', 'AltAllele_stranded'], as_index=False)
        .size()
        .rename(columns={
            '3_mer_h38': 'k_mer',
            'AltAllele_stranded': 'mutated_base',
            'size': 'nmut'
        })
    )
    opp_3 = (
        df_sim
        .groupby(['3_mer', 'mutated_base'], as_index=False)
        .size()
        .rename(columns={'size': 'nchance',
                         '3_mer': 'k_mer'})
    )

    merged_3 = (
        obs_3.merge(opp_3, on=['k_mer', 'mutated_base'], how='outer')
        .fillna({'nmut': 0, 'nchance': 0})
    )
    merged_3['density'] = np.where(
        merged_3['nchance'] > 0,
        merged_3['nmut'] / merged_3['nchance'],
        np.nan
    )

    return {"5mer": merged_5, "3mer": merged_3}


def analyze_reverse_complement_density_vectorized(density_matrix, cancer_type_key):
    """
    Analyzes genomic k-mer data by computing reverse complements and density differences.

    Parameters:
    -----------
    density_matrix : dict
        Dictionary containing genomic data matrices
    cancer_type_key : str
        Key to access specific cancer type data (e.g., "LUAD")

    Returns:
    --------
    pandas.DataFrame
        DataFrame sorted by density differences in descending order
    """
    cancer_matrix = density_matrix[cancer_type_key].copy()
    complement_trans = str.maketrans("ACGTacgt", "TGCAtgca")

    # Generate reverse complements
    cancer_matrix["k_mer_revers"] = [
        seq.translate(complement_trans)[::-1]
        for seq in cancer_matrix["k_mer"].values
    ]
    cancer_matrix["mutated_base_revers"] = [
        seq.translate(complement_trans)[::-1]
        for seq in cancer_matrix["mutated_base"].values
    ]

    # Create lookup dictionary for faster matching
    lookup_dict = {}
    for idx, row in cancer_matrix.iterrows():
        key = (row["k_mer"], row["mutated_base"])
        lookup_dict[key] = {
            'nmut': row['nmut'],
            'nchance': row['nchance'],
            'density': row['density']
        }

    # Use lookup for reverse complement data
    reverse_data = []
    for idx, row in cancer_matrix.iterrows():
        key = (row["k_mer_revers"], row["mutated_base_revers"])
        if key in lookup_dict:
            reverse_data.append(lookup_dict[key])
        else:
            reverse_data.append(
                {'nmut': np.nan, 'nchance': np.nan, 'density': np.nan})

    # Add columns efficiently
    reverse_df = pd.DataFrame(reverse_data)
    cancer_matrix["nmut_revers"] = reverse_df['nmut'].values
    cancer_matrix["nchance_revers"] = reverse_df['nchance'].values
    cancer_matrix["density_revers"] = reverse_df['density'].values

    cancer_matrix["density_diff"] = np.abs(
        cancer_matrix["density_revers"] - cancer_matrix["density"]
    )

    return cancer_matrix.sort_values(by="density_diff", ascending=False)


tcga_maf_h38_filtered_df = pd.read_parquet(maf_path)
driver_genes_path = (
    data_dir / "driver_genes_2024_06_18_IntOGenCompendium_Cancer_Genes.tsv")
driver_genes = pd.read_csv(driver_genes_path, sep="\t")

# %%
path_all_mutation_no_overlaped = "/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/data/all_chromosomes_bases_MUT_NO_dupli_k_mer.parquet"
path_all_mutation = "/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/data/all_chromosomes_bases_MUT_with_dupli_k_mer.parquet"

simulated_all_mutation_no_overlaped = pd.read_parquet(
    path_all_mutation_no_overlaped, dtype_backend="pyarrow")

# simulated_all_mutation_all = pd.read_parquet(
#    path_all_mutation, dtype_backend="pyarrow")
# %%
mutations_per_study = tcga_maf_h38_filtered_df["Study Abbreviation"].value_counts(
)
mutation_types_used = ["Missense_Mutation", "Silent", "Nonsense_Mutation"]
density_mats = {}
# %%
for study in mutations_per_study.index[0:10]:
    # Select driver genes specific to this tumor type
    driver_genes_for_study = (
        driver_genes[driver_genes["CANCER_TYPE"] == study]["SYMBOL"]
        .unique()
    )

    # Filter observed mutations (TCGA)
    df_obs = tcga_maf_h38_filtered_df[
        (tcga_maf_h38_filtered_df["Study Abbreviation"] == study) &
        (~tcga_maf_h38_filtered_df["gene_name"].isin(driver_genes_for_study)) &
        (tcga_maf_h38_filtered_df["mutation_type"].isin(mutation_types_used))
    ].copy()

    # Filter simulated mutations
    df_sim = simulated_all_mutation_no_overlaped[
        (~simulated_all_mutation_no_overlaped["gene_name"].isin(driver_genes_for_study)) &
        (simulated_all_mutation_no_overlaped["mutation_type"].isin(
            mutation_types_used))
    ].copy()

    print(
        f"Processing {study}: "
        f"Observed n = {len(df_obs)}, Simulated n = {len(df_sim)}, "
        f"Excluded driver genes = {len(driver_genes_for_study)}"
    )

    # Build density/immunogenicity tables for this tumor type
    density_mats[study] = build_density_tables(
        df_study=df_obs,
        df_sim=df_sim
    )
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# pd.read_parquet("/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/scripts/selection/mutation_matrix_density_MC3/LUAD_5mer.parquet")
 # %%
# %%
# %%
# %%
# %%
# %%

# --- Paths ---
base_dir = Path(data_dir) / "kmer_opportunities"
path_no_over = base_dir / "non_overlapping_cds"
path_all = base_dir / "all_cds"

path_no_over.mkdir(parents=True, exist_ok=True)
path_all.mkdir(parents=True, exist_ok=True)

# %%

# out_dir = pathlib.Path(
#    "/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/scripts/selection/mutation_matrix_density_MC3")
path_no_over.mkdir(exist_ok=True)

for study, mat_dict in density_mats.items():
    for kmer_type, df in mat_dict.items():  # kmer_type: '5mer' or '3mer'
        file_path = path_no_over / f"{study}_{kmer_type}.parquet"
        print(f"Saving {file_path.name} ({len(df)} rows)")
        df[["k_mer", "mutated_base", "nchance"]
           ].to_parquet(file_path, index=False)
# %%
# %%
# %%

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%


# %%
for study in mutations_per_study.index[0::]:
    print("==================================")
    print(f"-------------{study}-------------")
    print("==================================")

    # --- Driver genes for this study ---
    driver_genes_for_study = (
        driver_genes.loc[driver_genes["CANCER_TYPE"] == study, "SYMBOL"]
        .dropna()
        .unique()
    )

    # --- Filter simulated mutations (NO OVERLAP) ---
    df_sim_no_overlap = simulated_all_mutation_no_overlaped.loc[
        (~simulated_all_mutation_no_overlaped["gene_name"].isin(
            driver_genes_for_study))
        & (simulated_all_mutation_no_overlaped["mutation_type"].isin(mutation_types_used))
    ].copy()

    # --- Filter simulated mutations (ALL CDS) ---
    df_sim_all = simulated_all_mutation_all.loc[
        (~simulated_all_mutation_all["gene_name"].isin(driver_genes_for_study))
        & (simulated_all_mutation_all["mutation_type"].isin(mutation_types_used))
    ].copy()

    # --- Compute opportunities ---
    opp_3_no_overlap = compute_mutation_opportunities(
        df_sim_no_overlap,
        kmer_col="3_mer",
        mutated_base_col="mutated_base",
        out_col="nchance",
    )
    opp_5_no_overlap = compute_mutation_opportunities(
        df_sim_no_overlap,
        kmer_col="5_mer",
        mutated_base_col="mutated_base",
        out_col="nchance",
    )

    opp_3_all = compute_mutation_opportunities(
        df_sim_all,
        kmer_col="3_mer",
        mutated_base_col="mutated_base",
        out_col="nchance",
    )
    opp_5_all = compute_mutation_opportunities(
        df_sim_all,
        kmer_col="5_mer",
        mutated_base_col="mutated_base",
        out_col="nchance",
    )


# %%
# %%
