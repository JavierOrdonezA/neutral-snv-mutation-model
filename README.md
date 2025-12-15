# neutral-snv-mutation-model

### Neutral mutation opportunities

The repository includes precomputed SNV mutation opportunity tables based on
3-mer and 5-mer sequence context.

Two versions are provided:

- `non_overlapping_cds/`: mutation opportunities computed only from coding
  regions that do not overlap on opposite strands. This is the default and
  recommended option for neutral modeling.

- `all_cds/`: mutation opportunities computed from all coding regions,
  including overlapping genes on opposite strands. This dataset is provided
  for sensitivity analyses.