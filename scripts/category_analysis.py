
# %%
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from scipy.stats import ks_2samp
from scipy.stats import binomtest
import seaborn as sns
import os
import argparse
from tqdm import tqdm
import pyarrow.parquet as pq
import warnings
import dask.dataframe as dd
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Mapping, Union, List, Optional
from typing import Optional
import sys
from statsmodels.stats.multitest import multipletests
from pathlib import Path


warnings.filterwarnings('ignore')

date_tag = datetime.now().strftime("%Y%m%d")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# %%


class Config:
    """Configuration parameters for the analysis."""

    def __init__(
        self,
        base_dir: Path = Path(
            "/Users/fordonez/CRGCluster/jordonez/Task_16_dN_dS/"),
        cohort: str = "LUAD",
        chromosome: str = None,
        windows=None
        # chromosome: str = "chr21",
        # windows=(100_000,)
    ):
        self.base_dir = base_dir
        self.cohort = cohort
        self.chromosome = chromosome
        self.windows = windows

    def set_chromosome(self, chrom: str) -> None:
        """Update chromosome dynamically (useful in loops)."""
        self.chromosome = chrom

    @property
    def paths(self) -> Dict[str, Path]:
        """Return dictionary of file paths for the current chromosome."""
        return {
            "maf38_mc3_filtered": self.base_dir / "scripts/selection/MC3_TCGA_bed_with_maf38_filtered_hg38_annotated.parquet",
            "mut_matrix_3": self.base_dir / f"scripts/selection/mutation_matrix_density_MC3/{self.cohort}_3mer.parquet",
            "mut_matrix_5": self.base_dir / f"scripts/selection/mutation_matrix_density_MC3/{self.cohort}_5mer.parquet",
            "cds_no_overlap": self.base_dir / "data/all_chromosomes_bases_NO_mut_k_mer.parquet",
            "cds_mut_no_overlap": self.base_dir / "data/all_chromosomes_Filtered_mutations_no_overlap.parquet",
            "peptides": self.base_dir / f"result_mutations/{self.chromosome}/{self.chromosome}_peptides.parquet",
            'chr_list': self.base_dir / f"data/chromosomes_list.txt",
            "binding_mut_pept": self.base_dir / f"data/unique_peptide_MERGE_binding_all_chr/{self.chromosome}_unique_peptide_binding.parquet",
            "binding_wild_pept": self.base_dir / f"data/unique_WILD_peptide_MERGE_binding_all_chr/{self.chromosome}_unique_peptide_binding.parquet",
            "peptides_binding_simple": self.base_dir / "data/Immunogenecity_x_axes_1_6_hlas" / f"immunogenicity_{self.chromosome}_simple.parquet",
            "peptides_binding_sigmoid": self.base_dir / "data/Immunogenecity_x_axes_1_6_hlas" / f"immunogenicity_{self.chromosome}_sigmoid.parquet",
            "immune_window_non_driver": self.base_dir / "scripts/dnds_mc3/Immuno_x_axis_hla_1_and_6/results_Immunogenecity_1_6_hlas_no_driver/cds_unique_with_windows.parquet",
            "immune_window_result_simple": self.base_dir / "scripts/dnds_mc3/Immuno_x_axis_hla_1_and_6/results_Immunogenecity_1_6_hlas_no_driver/LUAD_results_simple.pkl.gz",
            "immune_window_result_sigmoid": self.base_dir / "scripts/dnds_mc3/Immuno_x_axis_hla_1_and_6/results_Immunogenecity_1_6_hlas_no_driver/LUAD_results_sigmoid.pkl.gz",
            "dn_ds_expected_counts": self.base_dir / "scripts/dnds_mc3/Immuno_x_axis_hla_1_and_6/results_Immunogenecity_1_6_hlas_no_driver/expected_densities_LUAD_20251203.pkl.gz",
            "dn_ds_observed_counts": self.base_dir / "scripts/dnds_mc3/Immuno_x_axis_hla_1_and_6/results_Immunogenecity_1_6_hlas_no_driver/observed_counts_LUAD_20251203.pkl.gz",
        }


base_comp = str.maketrans("ACGT", "TGCA")


def analyze_reverse_complement_density_vectorized(cancer_matrix):
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
# ================================================================
# Convert 3-mer matrix into SBS96
# ================================================================


def compute_sbs96(mut_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 3-mer mutation matrix into canonical pyrimidine-centered SBS96 format.

    Steps:
    1. Reverse-complement k-mer and mutated base when the central reference base
       is A or G (canonicalization to pyrimidine space).
    2. Aggregate counts by corrected 3-mer and mutation.
    3. Compute mutation density = nmut / nchance.
    4. Build SBS-96 labels: X[Ref>Alt]Y.

    Parameters
    ----------
    mut_df : pd.DataFrame
        Must include columns: ["k_mer", "mutated_base", "nmut", "nchance"].

    Returns
    -------
    pd.DataFrame
        Tidy SBS96 matrix with columns:
        ['kmer', 'mutated_base', 'nmut', 'nchance', 'density', 'sbs96']
    """

    df = mut_df.copy()

    # --- Complement map ---
    comp: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def reverse_complement(kmer: str) -> str:
        return "".join(comp[b] for b in kmer)[::-1]

    def complement_base(base: str) -> str:
        return comp[base]

    # --- Canonicalization step ---
    def process_row(row):
        kmer = row["k_mer"]
        base = row["mutated_base"]
        center = kmer[1]

        # Default case: already C or T
        new_kmer = kmer
        new_base = base

        # Convert purines to complementary pyrimidine context
        if center in {"A", "G"}:
            new_kmer = reverse_complement(kmer)
            new_base = complement_base(base)

        return pd.Series([new_kmer, new_base])

    df[["kmer_corrected", "mut_base_corrected"]
       ] = df.apply(process_row, axis=1)

    # --- Aggregate collapsed matrix ---
    collapsed = (
        df.groupby(["kmer_corrected", "mut_base_corrected"], as_index=False)
        .agg({"nmut": "sum", "nchance": "sum"})
    )

    collapsed["density"] = collapsed["nmut"] / collapsed["nchance"]

    collapsed = collapsed.rename(
        columns={"kmer_corrected": "k_mer",
                 "mut_base_corrected": "mutated_base"}
    )

    # --- Create SBS-96 notation ---
    base_5p = collapsed["k_mer"].str[0]
    base_mut = collapsed["k_mer"].str[1]
    base_3p = collapsed["k_mer"].str[2]

    collapsed["sbs96"] = (
        base_5p + "[" + base_mut + ">" +
        collapsed["mutated_base"] + "]" + base_3p
    )

    # Sort final table
    collapsed = collapsed.sort_values("sbs96").reset_index(drop=True)

    def extract_category(s):
        return s.split("[")[1].split(">")[0] + ">" + s.split(">")[1].split("]")[0]

    collapsed["category"] = collapsed["sbs96"].apply(extract_category)

    ordered_categories = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    collapsed["category"] = pd.Categorical(
        collapsed["category"], categories=ordered_categories, ordered=True)

    return collapsed


def compute_sbs1536(mut_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 5-mer mutation matrix into canonical pyrimidine-centered SBS1536 format.

    Steps:
    1. Reverse-complement k-mer and mutated base when the central reference base 
       is A or G (canonicalization to pyrimidine space).
    2. Aggregate counts by corrected 5-mer and mutation.
    3. Compute mutation density = nmut / nchance.
    4. build SBS1536 labels: XX[Ref>Alt]YY.

    Parameters
    ----------
    mut_df : pd.DataFrame
        Must include columns: ["k_mer", "mutated_base", "nmut", "nchance"].
        k_mer must be length 5.

    Returns
    -------
    pd.DataFrame
        Tidy SBS1536 matrix with columns:
        ['kmer', 'mutated_base', 'nmut', 'nchance', 'density', 'sbs1536']
    """

    df = mut_df.copy()

    # --- Complement map ---
    comp: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def reverse_complement(kmer: str) -> str:
        return "".join(comp[b] for b in kmer)[::-1]

    def complement_base(base: str) -> str:
        return comp[base]

    # --- Canonicalization step ---
    def process_row(row):
        kmer = row["k_mer"]
        base = row["mutated_base"]
        center = kmer[2]  # <-- 5-mer center

        new_kmer = kmer
        new_base = base

        # Convert purine-centered mutations to pyrimidine-centered space
        if center in {"A", "G"}:
            new_kmer = reverse_complement(kmer)
            new_base = complement_base(base)

        return pd.Series([new_kmer, new_base])

    df[["kmer_corrected", "mut_base_corrected"]
       ] = df.apply(process_row, axis=1)

    # --- Aggregate collapsed matrix ---
    collapsed = (
        df.groupby(["kmer_corrected", "mut_base_corrected"], as_index=False)
          .agg({"nmut": "sum", "nchance": "sum"})
    )

    collapsed["density"] = collapsed["nmut"] / collapsed["nchance"]

    collapsed = collapsed.rename(
        columns={"kmer_corrected": "k_mer",
                 "mut_base_corrected": "mutated_base"}
    )

    # --- Create SBS-1536 notation ---
    kmer = collapsed["k_mer"]

    # 5-mer = X1 X2 [Ref>Alt] Y1 Y2
    X1 = kmer.str[0]
    X2 = kmer.str[1]
    Ref = kmer.str[2]
    Y1 = kmer.str[3]
    Y2 = kmer.str[4]

    collapsed["sbs1536"] = (
        X1 + X2 + "[" + Ref + ">" + collapsed["mutated_base"] + "]" + Y1 + Y2
    )

    # Sort final table
    collapsed = collapsed.sort_values("sbs1536").reset_index(drop=True)

    def extract_category(s):
        return s.split("[")[1].split(">")[0] + ">" + s.split(">")[1].split("]")[0]
    collapsed["category"] = collapsed["sbs1536"].apply(extract_category)

    ordered_categories = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    collapsed["category"] = pd.Categorical(
        collapsed["category"], categories=ordered_categories, ordered=True)

    return collapsed


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(base_comp)[::-1]


def complement_base(base: str) -> str:
    """Return complementary DNA base."""
    return base.translate(base_comp)


def plot_sbs96_forward_reverse(
    df_forward: pd.DataFrame,
    df_reverse: pd.DataFrame,
    value_col: str = "density",
    sbs_col: str = "sbs96",
    title: str = "SBS-96 Mutational Profile — Forward vs Reverse",
    figsize: tuple[int, int] = (22, 10),
) -> None:
    """
    Plot SBS-96 mutational profiles for forward and reverse strands,
    grouped and colored by COSMIC categories.

    Parameters
    ----------
    df_forward : pd.DataFrame
        Forward-strand SBS96 dataframe.
    df_reverse : pd.DataFrame
        Reverse-strand SBS96 dataframe.
    value_col : str, default="density"
        Column to plot on Y-axis.
    sbs_col : str, default="sbs96"
        SBS96 context column.
    title : str
        Figure title.
    figsize : tuple[int, int]
        Figure size.
    """

    # ======================================================
    # COSMIC definitions
    # ======================================================
    CATEGORY_COLORS: Dict[str, str] = {
        "C>A": "#1f77b4",
        "C>G": "#000000",
        "C>T": "#d62728",
        "T>A": "#7f7f7f",
        "T>C": "#2ca02c",
        "T>G": "#e377c2",
    }

    ORDERED_CATEGORIES: List[str] = list(CATEGORY_COLORS.keys())

    # ======================================================
    # Helpers
    # ======================================================
    def extract_category(s: str) -> str:
        ref = s.split("[")[1].split(">")[0]
        alt = s.split(">")[1].split("]")[0]
        return f"{ref}>{alt}"

    def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["category"] = out[sbs_col].apply(extract_category)
        out["category"] = pd.Categorical(
            out["category"],
            categories=ORDERED_CATEGORIES,
            ordered=True,
        )
        return out.sort_values(["category", sbs_col]).reset_index(drop=True)

    def add_category_spans(ax, df: pd.DataFrame) -> None:
        current_x = 0
        for cat in ORDERED_CATEGORIES:
            n = (df["category"] == cat).sum()
            ax.axvspan(
                current_x,
                current_x + n,
                color=CATEGORY_COLORS[cat],
                alpha=0.15,
                zorder=0,
            )
            current_x += n

    def draw_top_category_bar(ax, df: pd.DataFrame) -> None:
        ax.set_ylim(0, 1)
        ax.axis("off")

        current_x = 0
        for cat in ORDERED_CATEGORIES:
            n = (df["category"] == cat).sum()
            ax.barh(
                y=0.3,
                width=n,
                left=current_x,
                height=0.55,
                color=CATEGORY_COLORS[cat],
            )
            ax.text(
                current_x + n / 2,
                0.85,
                cat,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
            current_x += n

    # ======================================================
    # Prepare data
    # ======================================================
    fwd = prepare_df(df_forward)
    rev = prepare_df(df_reverse)

    bar_positions = range(len(fwd))
    colors = fwd["category"].map(CATEGORY_COLORS).tolist()

    # ======================================================
    # Plot
    # ======================================================
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.25, 4, 4])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_fwd = fig.add_subplot(gs[1, 0])
    ax_rev = fig.add_subplot(gs[2, 0], sharex=ax_fwd)

    # Forward
    ax_fwd.bar(bar_positions, fwd[value_col], color=colors, width=0.8)
    ax_fwd.set_ylabel("Forward\nMutation density",
                      fontsize=13, fontweight="bold")
    ax_fwd.tick_params(axis="x", bottom=False, labelbottom=False)

    # Reverse
    ax_rev.bar(bar_positions, rev[value_col], color=colors, width=0.8)
    ax_rev.set_ylabel("Reverse\nMutation density",
                      fontsize=13, fontweight="bold")
    ax_rev.set_xticks(bar_positions)
    ax_rev.set_xticklabels(
        fwd[sbs_col],
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )

    # Shared X limits
    xlim = (-0.5, len(fwd) - 0.5)
    for ax in (ax_fwd, ax_rev, ax_top):
        ax.set_xlim(xlim)

    # COSMIC decorations
    add_category_spans(ax_fwd, fwd)
    add_category_spans(ax_rev, rev)
    draw_top_category_bar(ax_top, fwd)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)
    plt.tight_layout()
    plt.show()


def plot_sbs96_profile(
    mut_matrix: pd.DataFrame,
    value_col: str = "density",
    title: str = "SBS-96 Profile",
    figsize: tuple[int, int] = (22, 6),
    compute_fn=None,
) -> None:
    """
    Compute and plot an SBS-96 mutational profile with COSMIC categories.

    Parameters
    ----------
    mut_matrix : pd.DataFrame
        Input mutation matrix used by `compute_sbs96`.
    value_col : str, default="density"
        Column to plot (e.g. "density", "nmut", etc.).
    title : str
        Figure title.
    figsize : tuple[int, int]
        Figure size.
    compute_fn : callable
        Function that computes SBS96 (e.g. compute_sbs96).

    Returns
    -------
    None
    """

    if compute_fn is None:
        raise ValueError("You must provide compute_fn (e.g. compute_sbs96).")

    # --- COSMIC categories ---
    CATEGORY_COLORS = {
        "C>A": "#1f77b4",
        "C>G": "#000000",
        "C>T": "#d62728",
        "T>A": "#7f7f7f",
        "T>C": "#2ca02c",
        "T>G": "#e377c2",
    }

    ORDERED_CATEGORIES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]

    # --- Helpers ---
    def extract_category(s: str) -> str:
        # "ACA[C>T]TGA" → "C>T"
        return s.split("[")[1].split(">")[0] + ">" + s.split(">")[1].split("]")[0]

    # --- Compute SBS96 ---
    sbs96_df = compute_fn(mut_matrix).copy()
    sbs96_df["category"] = sbs96_df["sbs96"].apply(extract_category)

    sbs96_df["category"] = pd.Categorical(
        sbs96_df["category"],
        categories=ORDERED_CATEGORIES,
        ordered=True,
    )

    plot_df = sbs96_df.sort_values(
        ["category", "sbs96"]).reset_index(drop=True)

    # --- Plot ---
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.25, 4])

    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    bar_positions = range(len(plot_df))
    colors = plot_df["category"].map(CATEGORY_COLORS).tolist()

    # Main bars
    ax.bar(bar_positions, plot_df[value_col], color=colors, width=0.8)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(
        plot_df["sbs96"],
        rotation=90,
        fontsize=10,
        fontweight="bold",
    )
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize=12)

    # Category background shading
    current_x = 0
    for cat in ORDERED_CATEGORIES:
        n = (plot_df["category"] == cat).sum()
        ax.axvspan(
            current_x,
            current_x + n,
            color=CATEGORY_COLORS[cat],
            alpha=0.15,
        )
        current_x += n

    ax.set_ylabel("Mutation density", fontsize=14, fontweight="bold")
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)

    # --- Top category bar ---
    ax_top.set_xlim(-0.5, len(plot_df) - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    bar_height = 0.55
    current_x = 0

    for cat in ORDERED_CATEGORIES:
        n = (plot_df["category"] == cat).sum()

        ax_top.barh(
            y=0.3,
            width=n,
            left=current_x,
            height=bar_height,
            color=CATEGORY_COLORS[cat],
        )

        ax_top.text(
            current_x + n / 2,
            0.9,
            cat,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        current_x += n

    plt.xlim(-0.5, len(plot_df) - 0.5)
    plt.tight_layout()
    plt.show()


# %%


def split_kmer_reverse_complements(
    df: pd.DataFrame,
    kmer_col: str = "k_mer",
    mut_col: str = "mutated_base",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a mutation k-mer matrix into forward (canonical) and
    reverse-complement rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input matrix with at least:
        - k_mer
        - mutated_base
    kmer_col : str
        Column name containing k-mers.
    mut_col : str
        Column name containing mutated base.

    Returns
    -------
    df_forward : pd.DataFrame
        Rows corresponding to canonical k-mers.
    df_reverse : pd.DataFrame
        Rows corresponding to reverse complements.
    """

    out = df.copy()

    # Reverse complements
    out["kmer_rc"] = out[kmer_col].apply(reverse_complement)
    out["mutated_base_rc"] = out[mut_col].apply(complement_base)

    # Canonical k-mer (lexicographically smallest)
    out["kmer_canonical"] = out.apply(
        lambda r: min(r[kmer_col], r["kmer_rc"]),
        axis=1
    )

    # Strand assignment
    out["strand"] = out.apply(
        lambda r: "forward" if r[kmer_col] == r["kmer_canonical"] else "reverse",
        axis=1
    )

    # Split
    df_forward = (
        out[out["strand"] == "forward"]
        .copy()
    )
    df_reverse = (
        out[out["strand"] == "reverse"]
        .copy()
    )

    df_forward = df_forward[[
        "k_mer", "mutated_base", "nmut", "nchance", "density"]].copy()

    df_reverse = df_reverse[[
        "k_mer", "mutated_base", "nmut", "nchance", "density"]]

    df_reverse["k_mer"] = df_reverse["k_mer"].apply(reverse_complement)
    df_reverse["mutated_base"] = df_reverse["mutated_base"].apply(
        complement_base)

    df_forward = (
        df_forward
        .sort_values(by=["k_mer", "mutated_base"])
        .reset_index(drop=True)
    )
    df_reverse = (
        df_reverse
        .sort_values(by=["k_mer", "mutated_base"])
        .reset_index(drop=True)
    )

    return out, df_forward, df_reverse


# %%
# Config & paths
config = Config(
    cohort="LUAD",
    base_dir=Path("/Users/fordonez/CRGCluster/jordonez/Task_16_dN_dS/")
    # base_dir=Path("/users/dweghorn/jordonez/Task_16_dN_dS/")
)
paths = config.paths

mut_matrix_3 = pd.read_parquet(paths["mut_matrix_3"])

mut_matrix_5 = pd.read_parquet(paths["mut_matrix_5"])
mut_matrix_5 = analyze_reverse_complement_density_vectorized(
    mut_matrix_5).sort_values(by="k_mer").reset_index(drop=True)

mut_matrix_3 = analyze_reverse_complement_density_vectorized(
    mut_matrix_3).sort_values(by="k_mer").reset_index(drop=True)

# %%
mut_matrix_3_sbs96 = compute_sbs96(mut_matrix_3)


all_out_3, df_forward_3, df_reverse_3 = split_kmer_reverse_complements(
    mut_matrix_3)

all_out_5, df_forward_5, df_reverse_5 = split_kmer_reverse_complements(
    mut_matrix_5)

df_forward_3_sbs96, df_reverse_3_sbs96 = compute_sbs96(
    df_forward_3), compute_sbs96(df_reverse_3)

df_forward_5_sbs1536, df_reverse_5_sbs1536 = compute_sbs1536(
    df_forward_5), compute_sbs1536(df_reverse_5)
# %%
# %%
plot_sbs96_forward_reverse(
    df_forward_3_sbs96,
    df_reverse_3_sbs96,
    value_col="density",
)

plot_sbs96_profile(
    mut_matrix_3,
    value_col="density",
    compute_fn=compute_sbs96,
)
# %%


def poisson_rate_ratio_test(
    count_group_a: int,
    exposure_group_a: float,
    count_group_b: int,
    exposure_group_b: float,
):
    """
    Exact Poisson rate ratio test using conditioning on the total number
    of observed events.

    This function tests whether two event rates are equal when counts are
    assumed to arise from independent Poisson processes with different
    exposures.

    Statistical model
    -----------------
    Let

        X ~ Poisson(λ_A * E_A)
        Y ~ Poisson(λ_B * E_B)

    where:
        - X, Y are observed event counts,
        - λ_A, λ_B are the underlying event rates,
        - E_A, E_B are known exposures (e.g. number of mutation opportunities,
          genomic length, or time at risk).

    Hypotheses
    ----------
    H0: λ_A = λ_B   (equal rates)
    H1: λ_A ≠ λ_B   (different rates)

    Conditioning argument (key idea)
    --------------------------------
    Under H0, conditioning on the total number of observed events

        T = X + Y

    removes the nuisance parameter λ and yields:

        X | (X + Y = T) ~ Binomial(
            T,
            p = E_A / (E_A + E_B)
        )

    This allows an *exact* test using a binomial distribution, avoiding
    asymptotic approximations.

    Interpretation
    --------------
    - The p-value tests whether the observed allocation of events between
      groups A and B is compatible with equal Poisson rates, given their
      relative exposures.
    - The rate ratio estimates the relative mutation/event intensity between
      groups.

    This test is particularly appropriate when:
        - Counts are low or sparse
        - Exposures differ between groups
        - An exact (non-asymptotic) test is desired

    References
    ----------
    - Sahai & Khurshid (1993), Statistics in Epidemiology
    - rateratio.test R vignette:
      https://cran.r-project.org/web/packages/rateratio.test/vignettes/rateratio.test.pdf

    Parameters
    ----------
    count_group_a : int
        Number of observed events in group A.
    exposure_group_a : float
        Total exposure (e.g. opportunities, time at risk) for group A.
    count_group_b : int
        Number of observed events in group B.
    exposure_group_b : float
        Total exposure (e.g. opportunities, time at risk) for group B.

    Returns
    -------
    dict
        Dictionary containing:
        - rate_group_a : estimated rate in group A (count / exposure)
        - rate_group_b : estimated rate in group B (count / exposure)
        - rate_ratio   : rate_group_a / rate_group_b
        - p_value      : exact two-sided p-value for H0: λ_A = λ_B


    Under the Poisson model, the equality of rates is equivalent to the conditional distribution of events following a binomial with

    #-------"Given what I observed, is there evidence that λ_f ≠ λ_r?"------

    """

    "https://cran.r-project.org/web/packages/rateratio.test/vignettes/rateratio.test.pdf"
    total_events = count_group_a + count_group_b

    # Under H0: equal Poisson rates
    prob_event_in_a_h0 = (
        exposure_group_a / (exposure_group_a + exposure_group_b)
    )

    # Exact conditional binomial test
    test_result = binomtest(
        count_group_a,
        total_events,
        p=prob_event_in_a_h0,
        alternative="two-sided",
    )

    rate_group_a = count_group_a / exposure_group_a
    rate_group_b = count_group_b / exposure_group_b

    # rate_ratio = rate_group_a / rate_group_b

    # Handle zero-rate cases explicitly
    if rate_group_b == 0 and rate_group_a > 0:
        rate_ratio = np.inf
    elif rate_group_a == 0 and rate_group_b > 0:
        rate_ratio = 0.0
    elif rate_group_a == 0 and rate_group_b == 0:
        rate_ratio = np.nan
    else:
        rate_ratio = rate_group_a / rate_group_b

    return {
        "rate_group_a": rate_group_a,
        "rate_group_b": rate_group_b,
        "rate_ratio": rate_ratio,
        "p_value": test_result.pvalue,
    }


def run_rate_ratio_by_category(
    df_forward: pd.DataFrame,
    df_reverse: pd.DataFrame,
    categories: list[str],
) -> pd.DataFrame:
    """
    Run exact Poisson rate ratio test for each SBS category.

    Returns
    -------
    pd.DataFrame
        One row per category with rates, rate ratio and p-value.
    """
    results = []

    for category in categories:
        fwd = df_forward.query("category == @category")
        rev = df_reverse.query("category == @category")

        count_forward = int(fwd["nmut"].sum())
        exposure_forward = fwd["nchance"].sum()

        count_reverse = int(rev["nmut"].sum())
        exposure_reverse = rev["nchance"].sum()

        res = poisson_rate_ratio_test(
            count_forward,
            exposure_forward,
            count_reverse,
            exposure_reverse,
        )

        results.append({
            "category": category,
            "count_forward": count_forward,
            "count_reverse": count_reverse,
            "exposure_forward": exposure_forward,
            "exposure_reverse": exposure_reverse,
            **res,
        })
    results = pd.DataFrame(results) .sort_values(
        by="p_value").reset_index(drop=True)
    return results


category_tests = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

df_results_3_mers = run_rate_ratio_by_category(
    df_forward_3_sbs96,
    df_reverse_3_sbs96,
    category_tests,
)
df_results_5_mers = run_rate_ratio_by_category(
    df_forward_5_sbs1536,
    df_reverse_5_sbs1536,
    category_tests,
)
# %%
df_results_3_mers
# %%
df_results_5_mers
# %%


def run_rate_ratio_by_kmer_aligned(
    df_forward: pd.DataFrame,
    df_reverse: pd.DataFrame,
    min_total_events: int = 1,
) -> pd.DataFrame:
    """
    Compare forward and reverse mutation rates for each (kmer, mutated_base)
    assuming both dataframes contain the same contexts.

    Forward and reverse tables are aligned by (kmer, mutated_base), which
    represent reverse-complementary contexts of the same mutational process.
    For each context, mutation counts are modeled as Poisson processes with
    unequal exposure, and an exact rate ratio test is applied by conditioning
    on the total number of observed mutations.

    Parameters
    ----------
    df_forward : pd.DataFrame
        Forward-strand mutation matrix.
    df_reverse : pd.DataFrame
        Reverse-strand mutation matrix.
    min_total_events : int
        Minimum total number of mutations required to run the test.

    Returns
    -------
    pd.DataFrame
        One row per (k_mer, mutated_base) with rate ratio statistics.
    """
    merged = (
        df_forward[["k_mer", "mutated_base", "nmut", "nchance"]]
        .rename(columns={
            "nmut": "nmut_forward",
            "nchance": "nchance_forward",
        })
        .merge(
            df_reverse[["k_mer", "mutated_base", "nmut", "nchance"]]
            .rename(columns={
                "nmut": "nmut_reverse",
                "nchance": "nchance_reverse",
            }),
            on=["k_mer", "mutated_base"],
            how="inner",
            validate="one_to_one",
        )
    )

    results = []

    for _, row in merged.iterrows():
        total_events = row["nmut_forward"] + row["nmut_reverse"]

        if (
            total_events < min_total_events
            or row["nchance_forward"] == 0
            or row["nchance_reverse"] == 0
        ):
            continue

        res = poisson_rate_ratio_test(
            int(row["nmut_forward"]),
            row["nchance_forward"],
            int(row["nmut_reverse"]),
            row["nchance_reverse"],
        )

        results.append({
            "k_mer": row["k_mer"],
            "mutated_base": row["mutated_base"],
            "obs_forward": int(row["nmut_forward"]),
            "obs_reverse": int(row["nmut_reverse"]),
            "nchance_forward": row["nchance_forward"],
            "nchance_reverse": row["nchance_reverse"],
            **res,
        })

    results = pd.DataFrame(results)
    results["p_value_fdr"] = multipletests(
        results["p_value"],
        alpha=0.05,
        method="fdr_bh",
    )[1]
    results["k_mer_reverse"] = results["k_mer"].apply(reverse_complement)
    results["mutated_base_reverse"] = results["mutated_base"].apply(
        complement_base)
    columns = ['k_mer', 'mutated_base', 'k_mer_reverse',
               'mutated_base_reverse', 'obs_forward', 'obs_reverse',
               'nchance_forward', 'nchance_reverse', 'rate_group_a', 'rate_group_b',
               'rate_ratio', 'p_value', 'p_value_fdr']
    results = results[columns]
    return results


df_kmer_3 = run_rate_ratio_by_kmer_aligned(
    df_forward_3,
    df_reverse_3,
    min_total_events=1,
)
df_kmer_5 = run_rate_ratio_by_kmer_aligned(
    df_forward_5,
    df_reverse_5,
    min_total_events=1,
)
# %%


def apply_strand_bias_collapse(
    all_out_3: pd.DataFrame,
    df_kmer_3: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Fill nmut and nchance for each (k_mer, mutated_base) by collapsing or
    separating forward/reverse strands depending on strand-bias test.

    Logic
    -----
    - If p_value_fdr > alpha:
        No strand bias → collapse forward + reverse
    - Else:
        Strand bias → keep forward and reverse separated

    Parameters
    ----------
    all_out_3 : pd.DataFrame
        Must contain columns ['k_mer', 'mutated_base', 'strand'].
    df_kmer_3 : pd.DataFrame
        Must contain columns:
        ['k_mer', 'mutated_base',
         'obs_forward', 'nchance_forward',
         'k_mer_reverse', 'mutated_base_reverse',
         'obs_reverse', 'nchance_reverse',
         'p_value_fdr']
    alpha : float, optional
        Significance threshold for strand-bias test (default = 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['k_mer', 'mutated_base', 'strand', 'nmut', 'nchance', 'density'].
    """

    # --------------------------------------------------
    # Initialize output table
    # --------------------------------------------------
    df_1 = (
        all_out_3[['k_mer', 'mutated_base', 'strand']]
        .copy()
        .sort_values(by=['k_mer', 'mutated_base'])
    )

    df_1[['nmut', 'nchance']] = np.nan
    df_1 = df_1.set_index(['k_mer', 'mutated_base'])

    # --------------------------------------------------
    # Fill values according to strand-bias test
    # --------------------------------------------------
    l = 0
    for _, row in df_kmer_3.iterrows():

        kmer_cano = row['k_mer']
        mut_base_cano = row['mutated_base']

        obs_cano = row['obs_forward']
        nchance_cano = row['nchance_forward']

        kmer_rev = row['k_mer_reverse']
        mut_base_rev = row['mutated_base_reverse']

        obs_rev = row['obs_reverse']
        nchance_rev = row['nchance_reverse']

        p_value_fdr = row['p_value_fdr']

        # ----------------------------------------------
        # No strand bias → collapse
        # ----------------------------------------------
        if p_value_fdr > alpha:
            n_obs_total = obs_cano + obs_rev
            n_chance_total = nchance_cano + nchance_rev

            df_1.loc[(kmer_cano, mut_base_cano),
                     ['nmut', 'nchance']] = [
                n_obs_total, n_chance_total
            ]

            df_1.loc[(kmer_rev, mut_base_rev),
                     ['nmut', 'nchance']] = [
                n_obs_total, n_chance_total
            ]
            l = l+1

        # ----------------------------------------------
        # Strand bias → keep separated
        # ----------------------------------------------
        else:
            df_1.loc[(kmer_cano, mut_base_cano),
                     ['nmut', 'nchance']] = [
                obs_cano, nchance_cano
            ]

            df_1.loc[(kmer_rev, mut_base_rev),
                     ['nmut', 'nchance']] = [
                obs_rev, nchance_rev
            ]
    print(l)
    # --------------------------------------------------
    # Final formatting
    # --------------------------------------------------
    df_1 = df_1.reset_index()
    df_1['density'] = df_1['nmut'] / df_1['nchance']

    return df_1


# %%
matrix_3_collapse_k_mer = apply_strand_bias_collapse(
    all_out_3, df_kmer_3, alpha=0.05)
matrix_5_collapse_k_mer = apply_strand_bias_collapse(
    all_out_5, df_kmer_5, alpha=0.05)


matrix_3_no_strand_bias = apply_strand_bias_collapse(
    all_out_3, df_kmer_3, alpha=0)
matrix_5_no_strand_bias = apply_strand_bias_collapse(
    all_out_5, df_kmer_5, alpha=0)

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
# %%
# %%
"""

def canonical_kmer_mut_with_fallback(
    kmer: str,
    mut: str,
    df: pd.DataFrame,
) -> tuple[str, str]:
    
    #Try (k_mer, mut). If not present in df, try reverse-complement.
    

    # 1. Try direct
    mask = (df["k_mer"] == kmer) & (df["mutated_base"] == mut)
    if mask.any():
        return kmer, mut

    # 2. Try reverse complement
    kmer_rc = reverse_complement(kmer)
    mut_rc = complement_base(mut)

    mask_rc = (df["k_mer"] == kmer_rc) & (df["mutated_base"] == mut_rc)
    if mask_rc.any():
        return kmer_rc, mut_rc

    # 3. Nothing found
    raise KeyError(
        f"Mutation not found in SBS96: ({kmer}, {mut}) nor RC ({kmer_rc}, {mut_rc})"
    )

kmer_q = "AAA"
mut_q = "T"

kmer_c, mut_c = canonical_kmer_mut_with_fallback(
    kmer_q, mut_q,
    df_forward_3_sbs96)

df_forward_3_sbs96.query("k_mer == @kmer_c and mutated_base == @mut_c")

df_reverse_3_sbs96.query("k_mer == @kmer_c and mutated_base == @mut_c")
"""
# %%

# df_forward_3, df_reverse_3
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
