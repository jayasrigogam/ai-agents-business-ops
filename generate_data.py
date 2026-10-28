"""Generate synthetic machine sensor and maintenance data for Project 9."""
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(9)
    n = 1800
    machine_types = np.array(["Pump", "Conveyor", "Compressor", "Motor", "Hydraulic Press"])
    types = rng.choice(machine_types, size=n, p=[0.24, 0.22, 0.18, 0.24, 0.12])
    age = rng.uniform(0.3, 15, n).round(2)
    operating_hours = rng.integers(300, 24000, n)
    hours_since_service = rng.integers(0, 2400, n)
    temperature = rng.normal(68, 9, n) + age * 0.45
    vibration = np.maximum(0.1, rng.normal(3.2, 1.1, n) + age * 0.07)
    pressure = rng.normal(100, 12, n) - age * 0.25
    load_pct = np.clip(rng.normal(72, 16, n), 20, 120)
    recent_faults = rng.poisson(0.45, n)
    downtime_last_month = np.maximum(0, rng.gamma(1.6, 1.8, n) + recent_faults * 1.8)
    production_criticality = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.45, 0.30])

    # Build a synthetic risk score with sensor/maintenance factors.
    z = (-5.1
         + 0.035 * (temperature - 65)
         + 0.42 * (vibration - 3)
         + 0.012 * (hours_since_service / 10)
         + 0.085 * recent_faults
         + 0.025 * (load_pct - 70)
         + 0.10 * age
         + 0.12 * downtime_last_month)
    prob = 1 / (1 + np.exp(-z))
    failure = rng.binomial(1, prob)

    df = pd.DataFrame({
        "machine_id": [f"M{i:04d}" for i in range(1, n + 1)],
        "machine_type": types,
        "age_years": age,
        "operating_hours": operating_hours,
        "hours_since_maintenance": hours_since_service,
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(2),
        "pressure_kpa": pressure.round(2),
        "load_pct": load_pct.round(1),
        "recent_faults_90d": recent_faults,
        "downtime_hours_last_month": downtime_last_month.round(2),
        "production_criticality": production_criticality,
        "failure_next_30d": failure
    })
    df.to_csv("machine_data.csv", index=False)
    print(f"machine_data.csv created: {len(df)} machines, {int(df['failure_next_30d'].sum())} labeled failures")

if __name__ == "__main__":
    main()
