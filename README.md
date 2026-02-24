# cb-text-analytics

Topic modeling of central bank communications, tracking how themes and language
evolve across FOMC and RBNZ statements over time.

## What This Does

Applies LDA topic modeling to identify recurring themes in central bank statements
and track how those themes shift across policy eras.

## Data

- **Federal Reserve FOMC statements**: 2014–2017, 2023–2025
- **Reserve Bank of New Zealand OCR decisions**: 2006–2012

Both datasets are included under `usa-central-bank/` and `nz-central-bank/`.

## Analysis

- `fomc_analysis.ipynb` — Topic modeling of Fed statements
- `rbnz_analysis.ipynb` — Topic modeling of RBNZ statements

## Methods

Built with Python. Uses LDA for topic extraction and time-series analysis
to track topic prevalence across statement dates.

## Status

Active research project. FOMC analysis complete, RBNZ analysis in progress.

## License

CC0 — data and code are public domain.