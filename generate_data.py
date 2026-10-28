"""Step 1: synthetic data -> workload_data.csv  (300 employees x 12 months = 3,600 records)
Story: working hours keep rising, but output per hour falls once people pass ~45 hrs/week."""
import numpy as np, pandas as pd
rng = np.random.default_rng(11)
TEAMS = ["Support", "Operations", "Sales", "Engineering", "Finance", "Marketing", "HR", "Logistics"]
BASE_HOURS = {"Support": 47, "Operations": 46, "Sales": 43, "Engineering": 42, "Finance": 40, "Marketing": 40, "HR": 38, "Logistics": 44}
E = 300
emp = pd.DataFrame({"EmployeeID": np.arange(1, E + 1),
                    "Team": rng.choice(TEAMS, E, p=[.17, .15, .13, .17, .08, .1, .06, .14]),
                    "TenureYears": np.clip(rng.normal(4, 3, E), .3, 15).round(1),
                    "SkillRate": np.clip(rng.normal(1.0, .15, E), .6, 1.5)})   # individual ability
rows = []
for _, e in emp.iterrows():
    persist = rng.normal(0, 2)
    for m in range(1, 13):
        hours = np.clip(BASE_HOURS[e.Team] + 0.35 * m + persist + rng.normal(0, 2.5), 30, 62)   # hours creep up
        overtime = max(hours - 40, 0)
        fatigue = max(hours - 45, 0)
        engagement = np.clip(3.8 - 0.06 * overtime - 0.02 * m + rng.normal(0, .45), 1, 5)
        absent = np.clip(rng.poisson(0.5 + 0.09 * overtime + 0.25 * (engagement < 2.8)), 0, 8)
        meetings = np.clip(rng.normal(6 + 0.15 * m + (e.Team in ("Support", "Operations")) * 2, 2), 1, 20)
        eff_hours = max(hours - absent * 8 / 4.3, 20)
        # output per hour: efficient up to 45h, then strongly diminishing; engagement & meetings matter
        rate = e.SkillRate * (1 - 0.022 * fatigue) * (0.85 + 0.07 * engagement) * (1 - 0.006 * meetings)
        tasks = max(rng.normal(eff_hours * rate * 0.55, 1.2), 1)
        assigned = tasks + max(rng.normal(2 + 0.15 * overtime, 1.5), 0)
        rows.append((e.EmployeeID, e.Team, e.TenureYears, m, round(hours, 1), round(overtime, 1), int(absent),
                     round(engagement, 2), round(meetings, 1), round(assigned), round(tasks),
                     round(np.clip(rng.normal(3.3 + 0.2 * (engagement - 3.3), .5), 1, 5), 1)))
df = pd.DataFrame(rows, columns=["EmployeeID", "Team", "TenureYears", "Month", "WeeklyHours", "OvertimeHrs",
                                 "AbsentDays", "EngagementScore", "MeetingHrsPerWeek", "TasksAssigned",
                                 "TasksCompleted", "PerformanceRating"])
df.to_csv("workload_data.csv", index=False)
print(f"workload_data.csv created: {len(df)} records, {E} employees, 12 months")
