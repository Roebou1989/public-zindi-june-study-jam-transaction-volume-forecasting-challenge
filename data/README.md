# Data directory

The competition data is NOT included (it is competition-licensed and not redistributable).
Download it from the Zindi competition page and place these files here before running with
`RUN_TRAINING = True`:

- Train.csv
- Test.csv
- SampleSubmission.csv
- transactions_features.parquet
- financials_features.parquet
- demographics_clean.parquet

The notebook's default fast path (`RUN_TRAINING = False`) only reads the committed submission in
`submissions/` and does not need these files.
