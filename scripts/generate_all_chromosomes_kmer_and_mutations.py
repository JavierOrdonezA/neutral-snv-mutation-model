# %%
"""
Build canonical k-mer and mutation tables ordered by genomic coordinates.

This script constructs genome-wide, canonical k-mer and mutation tables
used as the basis for a neutral SNV opportunity model. For each chromosome,
it loads precomputed k-mer annotations and observed mutations, removes
ambiguous genomic coordinates arising from overlapping coding sequences,
and enforces a unique, reproducible ordering of bases.

Two ordering strategies are supported:
1. Absolute genomic coordinate ordering (chromosome, coord_abs).
2. Gene-wise ordering with 5'→3' mRNA base order within each gene.

The script iterates over all autosomal chromosomes, concatenates the
per-chromosome results, and writes consolidated Parquet files that are
ready for downstream statistical analysis.

Inputs
------
- Per-chromosome k-mer tables:
  {base_dir}/{chromosome}/{chromosome}_kmers.parquet

- Per-chromosome mutation tables:
  {base_dir}/{chromosome}/{chromosome}_mutations.parquet

- Chromosome list file (TSV):
  A single-column file listing chromosome identifiers to process.

Outputs
-------
- Genome-wide canonical k-mer table (Parquet):
  All coding bases ordered by genomic coordinates, with optional removal
  of overlapping (ambiguous) loci.

- Genome-wide canonical mutation table (Parquet):
  Observed mutations filtered to the canonical k-mer coordinate set and
  ordered consistently with the k-mer table.

Notes
-----
- Genomic coordinates mapping to multiple genes (e.g., overlapping CDS on
  opposite strands) can be optionally removed to avoid ambiguous mutation
  opportunities.
- The resulting tables define a unique coordinate space suitable for
  neutral mutation modeling and redistribution-based statistical tests.
"""
# %%
from pathlib import Path
import gc
from typing import Iterable, Optional, List, Dict, Any, Tuple, Union
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
# %%


def reorder_kmer_table_for_chr_and_rna(
    chr_number: str,
    base_dir: str = "/Users/fordonez/CRGCluster/jordonez/Task_16_dN_dS/result_mutations"
) -> pd.DataFrame:
    """
    For a given chromosome, loads its k-mer table and returns the DataFrame
    in reading order:
        1. Genes appear in genomic order (chromosome, coord_abs)
        2. Bases inside each gene block appear in 5'→3' mRNA order (coord_rna ascending)
        3. Removes overlapping coordinates (those mapping to >1 gene)

    Parameters
    ----------
    chr_number : str
        Chromosome number/name (e.g., "1", "2", "X").
    base_dir : str
        Directory where k-mer tables are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted and filtered as described.
    """
    # Columns to keep, in desired order
    keep_cols = [
        'chromosome', 'gene_id', 'transcript_id', 'gene_name', 'k_mer',
        'central_base', 'codon', 'exon_index', 'coord_abs', 'coord_rna',
        'mod3', 'strand_h38'
    ]

    # 1. Build file path and load the DataFrame
    kmers_path = f"{base_dir}/{chr_number}/{chr_number}_kmers.parquet"
    df_kmers = pd.read_parquet(kmers_path)
    df_kmers['chromosome'] = chr_number

    # 2. Reduce to required columns
    df_kmers = df_kmers[keep_cols].copy()

    # 3. Remove coordinates that appear in >1 gene (overlaps)
    df = df_kmers[~df_kmers.duplicated(
        subset=["chromosome", "coord_abs"], keep=False)].copy()

    # 4. Global genomic ordering (chromosome, coord_abs)
    df = df.sort_values(["chromosome", "coord_abs"], kind="mergesort")

    # 5. Assign contiguous block id (breaks when chr OR gene changes)
    block_break = (
        (df["chromosome"] != df["chromosome"].shift()) |
        (df["gene_id"] != df["gene_id"].shift())
    )
    df["block_id"] = block_break.cumsum()

    # 6. Sort within each block by mRNA order (coord_rna)
    df_final = (
        df.sort_values(["block_id", "coord_rna"], kind="mergesort")
          .drop(columns="block_id")
          .reset_index(drop=True)
    )

    return df_final


def reorder_kmer_table_abs_coord_sin_rna(
    chr_number: str,
    base_dir: str = "/Users/fordonez/CRGCluster/jordonez/Task_16_dN_dS/result_mutations",
    drop_dupli: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load k-mer and mutation tables for a chromosome and return them ordered
    by absolute genomic coordinate.

    Returns
    -------
    df_mers : pd.DataFrame
        K-mer table ordered by (chromosome, coord_abs)
    df_mut : pd.DataFrame
        Mutations filtered to coordinates present in df_mers
    """

    # Columns to keep, in desired order
    keep_cols = [
        'gene_id', 'transcript_id', 'gene_name', 'k_mer',
        'central_base', 'codon', 'exon_index', 'coord_abs', 'coord_rna',
        'mod3', 'strand_h38'
    ]

    # ---- Load k-mers ----
    kmers_path = f"{base_dir}/{chr_number}/{chr_number}_kmers.parquet"
    print(kmers_path)
    df_mers = pd.read_parquet(kmers_path,
                              columns=keep_cols)

    df_mers['chromosome'] = chr_number

    # Remove absolute genomic coordinates that map to more than one gene.
    # This avoids ambiguous k-mers arising from overlapping CDS on opposite strands.
    if drop_dupli:
        df_mers = df_mers[~df_mers.duplicated(
            subset=["chromosome", "coord_abs"], keep=False)]

    df_mers = (df_mers.
               sort_values(["chromosome", "coord_abs"], ascending=[True, True])
               .copy()
               .reset_index(drop=True))

    # ---- Load mutations ----
    mut_path = f"{base_dir}/{chr_number}/{chr_number}_mutations.parquet"
    df_mut = pd.read_parquet(mut_path)
    df_mut['chromosome'] = chr_number

    # Keep only mutations consistent with filtered genomic coordinates
    df_mut = df_mut.merge(df_mers[["gene_id", "chromosome", "coord_abs"]],
                          on=["gene_id", "chromosome", "coord_abs"],
                          how="inner",
                          )

    df_mers = df_mers.rename(columns={"k_mer": "5_mer"})
    df_mut = df_mut.rename(columns={"k_mer": "5_mer"})
    df_mers["3_mer"] = df_mers["5_mer"].str[1:4]
    df_mut["3_mer"] = df_mut["5_mer"].str[1:4]

    df_mer_order = [
        "chromosome", "gene_id", "transcript_id", "gene_name",
        "5_mer", "3_mer", "central_base", "codon", "exon_index",
        "coord_abs", "coord_rna", "mod3", "strand_h38",
    ]

    df_mut_order = [
        "chromosome", "gene_id", "transcript_id", "gene_name",
        "5_mer", "3_mer", "central_base", "mutated_base",
        "codon", "mutated_codon", "mutation_type", "wild_aa", "mutated_aa",
        "is_nonsynonymous", "is_nonsynonymous_nonstop",
        "exon_index", "coord_abs", "coord_rna", "mod3", "strand_h38",
    ]

    conditions = [
        # Silent
        df_mut["wild_aa"] == df_mut["mutated_aa"],
        # Stop-gained
        (df_mut["wild_aa"] != df_mut["mutated_aa"]) &
        (df_mut["mutated_aa"] == "*"),
        # Missense
        (df_mut["wild_aa"] != df_mut["mutated_aa"]) &
        (df_mut["mutated_aa"] != "*")
    ]
    choices = ["Silent", "Nonsense_Mutation", "Missense_Mutation"]

    df_mut["mutation_type"] = np.select(
        conditions, choices, default="Undefined")

    df_mut = df_mut[df_mut_order]
    df_mers = df_mers[df_mer_order]

    return df_mers, df_mut


# %%
chr_list = pd.read_csv(
    # "/users/dweghorn/jordonez/Task_16_dN_dS/chromosomes_list.txt",
    "/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/data/chromosomes_list.tsv",
    sep="\t",
    header=None,
    names=["chromosomes"]
)
chroms = chr_list["chromosomes"].tolist()
base_dir = "/Users/fordonez/CRGCluster/jordonez/Task_16_dN_dS/result_mutations"
# %%
results_coord = []
results_mut = []
drop_dupli = False

for chr_number in (c for c in chroms if c not in {"chrX", "chrY"}):
    print("="*40)
    print(f"Processing chromosome {chr_number}")
    # df_mers, _ = reorder_kmer_table_abs_coord_sin_rna(
    #    chr_number, base_dir,  drop_dupli=drop_dupli)

    df_mers, df_mut = reorder_kmer_table_abs_coord_sin_rna(
        chr_number, base_dir,  drop_dupli=drop_dupli)

    results_coord.append(df_mers)
    results_mut.append(df_mut)

df_cds_coords = pd.concat(results_coord, ignore_index=True)
df_mut_coords = pd.concat(results_mut, ignore_index=True)
# %%
del results_coord
gc.collect()
del results_mut
gc.collect()
# %%
suffix = "NO_dupli" if drop_dupli else "with_dupli"
output_path_cds = (
    f"/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/data/"
    f"all_chromosomes_bases_{suffix}_k_mer.parquet"
)
output_path_mut = (
    f"/Users/fordonez/Documents/PhD_Thesis/Task_16_dN_dS/data/"
    f"all_chromosomes_bases_MUT_{suffix}_k_mer.parquet"
)

df_cds_coords.to_parquet(output_path_cds, index=False)
df_mut_coords.to_parquet(output_path_mut, index=False)

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
base_dir: str = "/Users/fordonez/CRGCluster/jordonez/Task_16_dN_dS/result_mutations"
chr_number = "chr21"

"""
kmers_path = f"{base_dir}/{chr_number}/{chr_number}_kmers.parquet"
mut_path = f"{base_dir}/{chr_number}/{chr_number}_mutations.parquet"
peptides_path = f"{base_dir}/{chr_number}/{chr_number}_peptides.parquet"
seq_path = f"{base_dir}/{chr_number}/{chr_number}_sequences_table.parquet"
summary_filter_annot_path = f"{base_dir}/{chr_number}/{chr_number}_summary_filter_annot.parquet"
unique_mutated_nonsyn_nonstop_path = f"{base_dir}/{chr_number}/{chr_number}_unique_mutated_nonsyn_nonstop.parquet"

df_kmers = pd.read_parquet(kmers_path)
mut = pd.read_parquet(mut_path)
peptides = pd.read_parquet(peptides_path)
seq = pd.read_parquet(seq_path)
summary_filter_annot = pd.read_parquet(summary_filter_annot_path)
unique_mutated_nonsyn_nonstop = pd.read_parquet(
    unique_mutated_nonsyn_nonstop_path)

df_kmers = pd.DataFrame({
    "chromosome": [
        "chr1", "chr1", "chr1", "chr1", "chr1",
        "chr2", "chr2", "chr2", "chr2", "chr2",
    ],
    "gene_id": [
        "G1", "G1", "G1", "G1", "G1",
        "G2", "G2", "G2", "G2", "G2",
    ],
    "transcript_id": ["T1"] * 10,
    "gene_name": [
        "GENE1"] * 5 + ["GENE2"] * 5,
    "k_mer": ["AAAAA"] * 10,
    "central_base": ["A"] * 10,
    "codon": ["ATG"] * 10,
    "exon_index": [1] * 10,

    # 👇 coord_abs DESORDENADAS a propósito
    "coord_abs": [
        150, 100, 300, 200, 250,     # chr1
        900, 700, 800, 600, 1000     # chr2
    ],

    # coord_rna no importa aquí, pero la ponemos
    "coord_rna": [
        3, 1, 5, 2, 4,
        4, 2, 3, 1, 5
    ],
    "mod3": [0] * 10,
    "strand_h38": ["+"] * 10
})
df = (
    df_kmers[
        ~df_kmers.duplicated(subset=["chromosome", "coord_abs"], keep=False)
    ]
    .sort_values(["chromosome", "coord_abs"], ascending=[True, True])
    .copy()
    .reset_index(drop=True)
)
df
"""
