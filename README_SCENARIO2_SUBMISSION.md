# Scenario 2 Submission Pipeline

Scenario 2 only. The submitted uncertainty parameters are generated once by
`scripts/generate_scenario2_uncertainty.py` and are reused unchanged during
optimization, final evaluation, and CSV export.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Uncertainty Parameters

```bash
python scripts/generate_scenario2_uncertainty.py
```

This writes:

```text
config/scenario2_uncertainty.json
```

## Run Scenario 2

```bash
python scripts/run_scenario2_submission.py --resume
```

Useful short test:

```bash
python scripts/run_scenario2_submission.py --instances 6_Fattahi_20_workers.fjs --n-runs 2 --limit-runs 2 --allow-failed-runs
```

## Validate

```bash
python scripts/validate_scenario2_submission.py
```

Successful full validation prints:

```text
Validation OK: 300 Scenario-2 submission rows are valid.
```

## Important Outputs

```text
results/scenario2_submission/submission_scenario2.csv
results/scenario2_submission/submission_scenario2_with_metadata.csv
results/scenario2_submission/submission_manifest.json
config/scenario2_uncertainty.json
```

`results/scenario2_submission/` is ignored by Git. For the final submission,
package the generated files into the ZIP directly, or add specific result files
intentionally with `git add -f` if they must be versioned.

## Hardware

```text
CPU:
RAM:
OS:
Python:
```
