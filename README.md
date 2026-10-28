# Project 9 - Predictive Maintenance for Business Operations

## Goal
Predict which machines may fail soon using sensor readings, maintenance history, and production conditions. The project also recommends preventive maintenance actions.

## Files
- `generate_data.py` — creates a synthetic machine sensor dataset.
- `machine_data.csv` — generated dataset (created by running the first command).
- `predictive_maintenance.py` — trains a model, evaluates risk, creates charts and a report.
- `outputs/` — charts and `maintenance_risk_report.txt` (created by the analysis script).
- `requirements.txt` — Python packages needed.

## Run in GitHub Codespaces
Open the terminal in this folder and run these commands one at a time:

```bash
pip install -r requirements.txt
python generate_data.py
python predictive_maintenance.py
```

## Outputs
The script creates:
1. `outputs/1_failure_rate_by_machine_type.png`
2. `outputs/2_sensor_patterns.png`
3. `outputs/3_model_performance.png`
4. `outputs/4_highest_risk_machines.png`
5. `outputs/5_maintenance_recommendations.png`
6. `outputs/maintenance_risk_report.txt`

The data is synthetic and for demonstration/educational use only. Model performance on synthetic data is not evidence of real-world performance.
