"""Step 1: synthetic data -> employees.csv, employee_skills.csv, projects.csv, project_demand.csv
Story: plenty of people on paper, but a few skills (Cloud, ML, Cybersecurity, DevOps) are scarce."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)

SKILLS = ["Python", "SQL", "Data Analysis", "Project Management", "UI/UX", "QA Testing",
          "Cloud", "DevOps", "Cybersecurity", "Machine Learning"]
# how common each skill is among employees (scarce skills have low probability)
P = dict(zip(SKILLS, [0.30, 0.32, 0.28, 0.25, 0.15, 0.22, 0.07, 0.06, 0.05, 0.05]))
DEPTS = ["Engineering", "Data", "Product", "IT Ops", "QA"]
N = 220

emp = pd.DataFrame({
    "EmployeeID": np.arange(1, N + 1),
    "Department": rng.choice(DEPTS, N, p=[0.35, 0.15, 0.15, 0.2, 0.15]),
    "CommittedLoad": np.clip(rng.normal(0.72, 0.2, N), 0.2, 1.1).round(2),   # share of time already booked
})
rows = []
for e in emp.EmployeeID:
    for s in SKILLS:
        if rng.random() < P[s]:
            rows.append((e, s, int(rng.choice([1, 2, 3, 4, 5], p=[.15, .25, .3, .2, .1]))))
skills = pd.DataFrame(rows, columns=["EmployeeID", "Skill", "Proficiency"])

# 20 projects over 12 months; each needs a few skills (FTE, min proficiency)
dem = []
proj = []
for i in range(1, 21):
    start = int(rng.integers(1, 11)); dur = int(rng.integers(2, 5))
    proj.append((f"P{i:02d}", f"Project {i}", start, min(12, start + dur - 1)))
    need = rng.choice(SKILLS, size=int(rng.integers(2, 5)), replace=False,
                      p=np.array([.09, .09, .09, .09, .07, .09, .13, .12, .11, .12]) / 1.0)
    for s in need:
        dem.append((f"P{i:02d}", s, round(float(rng.choice([0.5, 1, 1, 1.5])), 1), 3))
projects = pd.DataFrame(proj, columns=["ProjectID", "ProjectName", "StartMonth", "EndMonth"])
demand = pd.DataFrame(dem, columns=["ProjectID", "Skill", "FTE_Needed", "MinProficiency"])

emp.to_csv("employees.csv", index=False); skills.to_csv("employee_skills.csv", index=False)
projects.to_csv("projects.csv", index=False); demand.to_csv("project_demand.csv", index=False)
print(f"{N} employees | {len(skills)} skill records | {len(projects)} projects | {len(demand)} skill requirements")
print(skills.Skill.value_counts())
