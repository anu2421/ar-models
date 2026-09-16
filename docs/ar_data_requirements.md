# AR data view — requirements for Data Engineering

Per the Data Engineering guide, Section 8, the autoregressive view should contain:

- `sequence` — clean, uppercase, standard-alphabet only
- `length`
- `split` — train / validation / test, respecting similarity-group boundaries
- `similarity_group_id`
- `data_version` — e.g. "1.0" — so every AR run can record which frozen version it used
- (optional) `activity_condition` — only if we decide to try conditioned generation

## Open questions for Data Engineering
- Format preference: parquet (matches the guide's stated deliverable format) — confirm.
- Row count expected for the training split, so we can size a fine-tuning run appropriately.
- Whether HydrAMP-derived sequences are flagged separately (they overlap with DBAASP/APD/
  DRAMP and shouldn't be treated as independent examples if we do any overlap analysis).

Status: DRAFT — send to Data Engineering lead for approval (Day 5 deliverable).
