"""Step 2: Workforce Capacity & Skill Gap Intelligence
Run: python skillgap_analysis.py   -> terminal results + charts in ./outputs/ + outputs/workforce_plan.txt
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
os.makedirs("outputs", exist_ok=True); sns.set_theme(style="whitegrid")

emp = pd.read_csv("employees.csv"); sk = pd.read_csv("employee_skills.csv")
proj = pd.read_csv("projects.csv"); dem = pd.read_csv("project_demand.csv")
MONTHS = list(range(1, 13)); SKILLS = sorted(sk.Skill.unique())
QUALIFIED = 3        # proficiency >= 3 counts as "can do the work"
TARGET_UTIL = 1.0    # planned up to full utilisation
# skills that are close neighbours -> good candidates for upskilling
ADJACENT = {"Cloud": ["DevOps", "Python"], "DevOps": ["Cloud", "Python"],
            "Cybersecurity": ["DevOps", "Cloud"], "Machine Learning": ["Data Analysis", "Python"],
            "Data Analysis": ["SQL", "Python"], "Python": ["SQL", "Data Analysis"], "SQL": ["Data Analysis"],
            "QA Testing": ["Python"], "UI/UX": ["Project Management"], "Project Management": ["UI/UX"]}

# ---------------------------------------------------------- 1. ON PAPER
dem = dem.merge(proj, on="ProjectID")
monthly_total = {m: dem[(dem.StartMonth <= m) & (dem.EndMonth >= m)].FTE_Needed.sum() for m in MONTHS}
free_fte = float((TARGET_UTIL - emp.CommittedLoad).clip(lower=0).sum())
print("=" * 70, "\n1. CAPACITY ON PAPER\n" + "=" * 70)
print(f"Employees: {len(emp)} | Total free capacity (FTE): {free_fte:.1f}")
print(f"Peak monthly project demand (FTE): {max(monthly_total.values()):.1f}")
print("-> On paper, the company has enough people. The problem is WHICH skills.")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(MONTHS, [monthly_total[m] for m in MONTHS], color="#4C78A8", label="Project demand (FTE)")
ax.axhline(free_fte, color="green", ls="--", label=f"Free capacity ({free_fte:.0f} FTE)")
ax.set_xlabel("Month"); ax.set_ylabel("FTE"); ax.set_title("Capacity vs Demand 'on paper'"); ax.legend()
plt.tight_layout(); plt.savefig("outputs/1_capacity_vs_demand.png", dpi=150); plt.close()

# ---------------------------------------------------------- 2. SKILL SUPPLY vs DEMAND
q = sk[sk.Proficiency >= QUALIFIED].merge(emp, on="EmployeeID")
q["Free"] = (TARGET_UTIL - q.CommittedLoad).clip(lower=0)
# an employee with several skills can only be used once -> split free time across their skills
q["Free"] = q.Free / q.groupby("EmployeeID").Skill.transform("count")
supply = q.groupby("Skill").Free.sum().reindex(SKILLS).fillna(0)
need = pd.DataFrame(0.0, index=SKILLS, columns=MONTHS)
for m in MONTHS:
    a = dem[(dem.StartMonth <= m) & (dem.EndMonth >= m)].groupby("Skill").FTE_Needed.sum()
    need[m] = a.reindex(SKILLS).fillna(0)
gap = need.sub(supply, axis=0)                      # positive = shortage

print("\n" + "=" * 70, "\n2. SKILL SUPPLY vs DEMAND (FTE)\n" + "=" * 70)
tbl = pd.DataFrame({"Qualified_People": q.groupby("Skill").EmployeeID.nunique().reindex(SKILLS).fillna(0).astype(int),
                    "Free_FTE_Supply": supply.round(1), "Peak_Demand": need.max(axis=1).round(1),
                    "Peak_Gap": gap.max(axis=1).round(1),
                    "Months_in_Deficit": (gap > 0).sum(axis=1)}).sort_values("Peak_Gap", ascending=False)
print(tbl)

plt.figure(figsize=(10, 5.5))
sns.heatmap(gap, annot=True, fmt=".1f", cmap="RdYlGn_r", center=0, cbar_kws={"label": "FTE gap (red = shortage)"})
plt.xlabel("Month"); plt.title("Skill Gap Heatmap (Demand - Free Supply)"); plt.tight_layout()
plt.savefig("outputs/2_skill_gap_heatmap.png", dpi=150); plt.close()

t = tbl.sort_values("Peak_Demand")
fig, ax = plt.subplots(figsize=(9, 5)); y = np.arange(len(t))
ax.barh(y - 0.2, t.Free_FTE_Supply, 0.4, label="Free supply", color="#4C9F70")
ax.barh(y + 0.2, t.Peak_Demand, 0.4, label="Peak demand", color="#D9534F")
ax.set_yticks(y); ax.set_yticklabels(t.index); ax.set_xlabel("FTE"); ax.legend()
ax.set_title("Skill Supply vs Peak Demand"); plt.tight_layout()
plt.savefig("outputs/3_supply_vs_demand_by_skill.png", dpi=150); plt.close()

# ---------------------------------------------------------- 3. PROJECTS AT RISK
print("\n" + "=" * 70, "\n3. PROJECTS AT RISK OF MISSING DEADLINES\n" + "=" * 70)
risk = []
for _, r in dem.iterrows():
    months = range(r.StartMonth, r.EndMonth + 1)
    worst = max(gap.loc[r.Skill, m] for m in months)
    if worst > 0:
        risk.append((r.ProjectID, r.ProjectName, r.Skill, r.StartMonth, round(worst, 1)))
rk = pd.DataFrame(risk, columns=["ProjectID", "Project", "ShortSkill", "StartMonth", "SkillGap_FTE"])
by_proj = rk.groupby(["ProjectID", "Project"]).agg(ShortSkills=("ShortSkill", lambda s: ", ".join(sorted(set(s)))),
                                                    Worst_Gap=("SkillGap_FTE", "max")).reset_index().sort_values("Worst_Gap", ascending=False)
print(f"{by_proj.ProjectID.nunique()} of {len(proj)} projects depend on a skill that is short during their run.")
print(by_proj.head(10).to_string(index=False))
plt.figure(figsize=(8, 4.5))
top = by_proj.head(10).iloc[::-1]
plt.barh(top.Project, top.Worst_Gap, color="#D9534F"); plt.xlabel("Worst skill gap (FTE)")
plt.title("Top Projects at Risk (skill shortage)"); plt.tight_layout()
plt.savefig("outputs/4_projects_at_risk.png", dpi=150); plt.close()

# ---------------------------------------------------------- 4. RECOMMENDATIONS
print("\n" + "=" * 70, "\n4. RECOMMENDATIONS: ALLOCATE / TRAIN / HIRE\n" + "=" * 70)
plan = []
for s in tbl[tbl.Peak_Gap > 0].index:
    g = tbl.loc[s, "Peak_Gap"]
    # (a) INTERNAL ALLOCATION: qualified people sitting on spare capacity (load < 60%)
    ids = q[(q.Skill == s) & (q.CommittedLoad < 0.6)].EmployeeID.unique()
    alloc = min(g, 0.3 * len(ids))                       # redeploy ~30% of their time
    # (b) TRAINING: people with a related skill (prof>=3) or this skill at prof 1-2, who have free time
    cand = sk[((sk.Skill == s) & (sk.Proficiency.between(1, 2))) |
              ((sk.Skill.isin(ADJACENT.get(s, []))) & (sk.Proficiency >= 3))].EmployeeID.unique()
    cand = set(cand) - set(sk[(sk.Skill == s) & (sk.Proficiency >= QUALIFIED)].EmployeeID)
    cand = emp[emp.EmployeeID.isin(cand) & (emp.CommittedLoad < 0.8)]
    # a trainee contributes ~0.25 FTE once trained; the training programme can run at most 6 people per skill
    train_people = min(len(cand), 6, int(np.ceil(max(g - alloc, 0) / 0.25)))
    train_fte = min(max(g - alloc, 0), train_people * 0.25)
    hire = max(g - alloc - train_fte, 0)
    plan.append((s, round(g, 1), round(alloc, 1), train_people, round(train_fte, 1), int(np.ceil(hire))))
pl = pd.DataFrame(plan, columns=["Skill", "Peak_Gap_FTE", "Reallocate_FTE", "People_To_Train", "Train_FTE", "Hire_Headcount"])
print(pl.to_string(index=False))

pp = pl.set_index("Skill")[["Reallocate_FTE", "Train_FTE"]].copy(); pp["Hire_FTE"] = pl.set_index("Skill").Hire_Headcount
pp.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#4C78A8", "#F0AD4E", "#D9534F"])
plt.ylabel("FTE gap covered"); plt.title("How to close each skill gap"); plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.savefig("outputs/5_gap_closing_plan.png", dpi=150); plt.close()

lines = ["WORKFORCE CAPACITY & SKILL GAP PLAN", "=" * 40,
         f"Employees: {len(emp)} | Free capacity: {free_fte:.0f} FTE | Peak demand: {max(monthly_total.values()):.0f} FTE",
         f"Critical skills in deficit: {', '.join(pl.Skill)}",
         f"Projects at risk: {by_proj.ProjectID.nunique()} of {len(proj)}", "", "ACTIONS:"]
for _, r in pl.iterrows():
    lines.append(f"- {r.Skill}: gap {r.Peak_Gap_FTE} FTE -> reallocate {r.Reallocate_FTE} FTE, "
                 f"train {r.People_To_Train} people, hire {r.Hire_Headcount}")
rep = "\n".join(lines); open("outputs/workforce_plan.txt", "w").write(rep)
print("\n" + rep)

# ---------------------------------------------------------- 5. OPTIONAL LLM
if os.getenv("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        m = anthropic.Anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=500,
            messages=[{"role": "user", "content": "As a workforce-planning advisor write a short executive summary "
                       "with 3 prioritised actions for this plan:\n\n" + rep}])
        print("\n" + "=" * 70, "\n5. LLM EXECUTIVE SUMMARY\n" + "=" * 70); print(m.content[0].text)
    except Exception as e:
        print("(LLM step skipped:", e, ")")
else:
    print("\n(Optional: set ANTHROPIC_API_KEY for an LLM-written executive summary.)")
print("Done. Charts saved in ./outputs/")
