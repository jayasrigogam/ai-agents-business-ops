"""Step 1: Create a synthetic HR dataset (1,500 employees) -> hr_data.csv
Attrition is generated from realistic drivers (overtime, low pay, low engagement,
little training, high absenteeism) so the AI model has real patterns to discover."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

dept = rng.choice(["Sales", "Engineering", "HR", "Operations", "Finance", "Marketing"],
                  N, p=[0.25, 0.25, 0.08, 0.22, 0.10, 0.10])
age = rng.integers(21, 58, N)
tenure = np.clip(rng.normal(4.5, 3.2, N), 0.2, 20).round(1)
salary = (np.clip(rng.normal(55000, 16000, N), 25000, 140000)
          + tenure * 1200).round(-2)
overtime_hrs = np.clip(rng.normal(8, 6, N), 0, 35).round(1)          # per month
workload = np.clip(rng.normal(3.2, 0.9, N) + overtime_hrs / 30, 1, 5).round(1)  # 1-5
absent_days = np.clip(rng.poisson(4, N) + (overtime_hrs > 15) * rng.poisson(3, N), 0, 30)
engagement = np.clip(rng.normal(3.4, 0.9, N) - (overtime_hrs - 8) / 25, 1, 5).round(1)
performance = np.clip(rng.normal(3.3, 0.8, N), 1, 5).round(1)
training_hrs = np.clip(rng.normal(22, 12, N), 0, 80).round(0)
last_promo_yrs = np.clip(rng.exponential(2.2, N), 0, 10).round(1)

# attrition probability (hidden "truth" the model should rediscover)
sal_pct = pd.Series(salary).groupby(dept).transform(lambda s: s / s.median()).values
z = (-2.6
     + 0.09 * (overtime_hrs - 8)
     - 1.6 * (sal_pct - 1)
     - 0.75 * (engagement - 3.4)
     - 0.025 * (training_hrs - 22)
     + 0.07 * (absent_days - 4)
     + 0.22 * (last_promo_yrs - 2.2)
     + 0.35 * (dept == "Sales") + 0.2 * (dept == "Operations")
     - 0.05 * tenure)
p = 1 / (1 + np.exp(-z))
attrition = (rng.random(N) < p).astype(int)

df = pd.DataFrame({
    "EmployeeID": np.arange(1001, 1001 + N), "Department": dept, "Age": age,
    "TenureYears": tenure, "MonthlySalary_INR_equiv": salary, "OvertimeHrsPerMonth": overtime_hrs,
    "WorkloadScore": workload, "AbsentDays": absent_days, "EngagementScore": engagement,
    "PerformanceRating": performance, "TrainingHrsPerYear": training_hrs,
    "YearsSincePromotion": last_promo_yrs,
    "Attrition": np.where(attrition == 1, "Yes", "No")})
df.to_csv("hr_data.csv", index=False)
print(f"hr_data.csv created: {len(df)} employees, attrition rate = {attrition.mean():.1%}")
print(df.head())
