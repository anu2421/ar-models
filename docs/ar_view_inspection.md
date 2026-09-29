# AR view inspection

Source directory: /content/AMP/data/processed/views
Rows: 27509
Columns: ['boman_index', 'similarity_group_id', 'data_version', 'eisenberg_hydrophobic_moment', 'length', 'net_charge_ph7', 'sequence', 'sequence_id', 'split']

## Schema check
```json
{
  "missing_required": [],
  "applied_aliases": {
    "similarity_group_id": "cluster_id"
  },
  "extra_columns": [
    "boman_index",
    "eisenberg_hydrophobic_moment",
    "net_charge_ph7",
    "sequence_id"
  ],
  "n_rows": 27509,
  "source_path": "/content/AMP/data/processed/views/autoregressive_view.parquet"
}
```

## Sequence validity
```json
{
  "n_generated": 27509,
  "n_valid": 27509,
  "validity_rate": 1.0,
  "reason_counts": {
    "valid": 27509
  },
  "n_duplicates": 0,
  "n_unique": 27509
}
```