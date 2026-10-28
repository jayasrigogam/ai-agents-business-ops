"""Step 2: Employee Productivity vs Workload Intelligence
Run: python productivity_analysis.py -> terminal results + charts + outputs/workload_report.txt
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); os.makedirs("outputs", exist_ok=True); sns.set_theme(style="whitegrid")

df = pd.read_csv("workload_data.csv")
df["CompletionRate"] = df.TasksCompleted / df.TasksAssigned
df["OutputPerHour"] = df.TasksCompleted / df.WeeklyHours            # productivity

# ------------------------------------------------ 1. THE PROBLEM: hours up, output flat
print("=" * 70, "\n1. IS PRODUCTIVITY REALLY DECLINING?\n" + "=" * 70)
trend = df.groupby("Month").agg(Hours=("WeeklyHours", "mean"), Tasks=("TasksCompleted", "mean"),
                                OutputPerHour=("OutputPerHour", "mean"), Overtime=("OvertimeHrs", "mean"),
                                Engagement=("EngagementScore", "mean")).round(3)
print(trend)
h0, h1 = trend.Hours.iloc[0], trend.Hours.iloc[-1]; p0, p1 = trend.OutputPerHour.iloc[0], trend.OutputPerHour.iloc[-1]
t0, t1 = trend.Tasks.iloc[0], trend.Tasks.iloc[-1]
print(f"\nHours/week: {h0:.1f} -> {h1:.1f} ({(h1/h0-1)*100:+.1f}%) | Tasks done: {t0:.1f} -> {t1:.1f} ({(t1/t0-1)*100:+.1f}%)"
      f" | Output per hour: {p0:.3f} -> {p1:.3f} ({(p1/p0-1)*100:+.1f}%)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(trend.index, trend.Hours, "o-", color="#D9534F"); ax[0].set_ylabel("Avg weekly hours", color="#D9534F")
a2 = ax[0].twinx(); a2.plot(trend.index, trend.Tasks, "s-", color="#4C78A8"); a2.set_ylabel("Avg tasks completed", color="#4C78A8")
ax[0].set_title("Hours rising, output not following"); ax[0].set_xlabel("Month")
ax[1].plot(trend.index, trend.OutputPerHour, "o-", color="#4C9F70"); ax[1].set_title("Output per hour (productivity)")
ax[1].set_xlabel("Month"); plt.tight_layout(); plt.savefig("outputs/1_hours_vs_output_trend.png", dpi=150); plt.close()

# ------------------------------------------------ 2. WORKLOAD CURVE & SWEET SPOT
print("\n" + "=" * 70, "\n2. WORKLOAD vs PRODUCTIVITY CURVE\n" + "=" * 70)
df["HoursBand"] = pd.cut(df.WeeklyHours, [0, 38, 42, 45, 48, 52, 70], labels=["<38", "38-42", "42-45", "45-48", "48-52", "52+"])
band = df.groupby("HoursBand", observed=True).agg(Employees=("EmployeeID", "count"), Tasks=("TasksCompleted", "mean"),
                                                 OutputPerHour=("OutputPerHour", "mean"), Absent=("AbsentDays", "mean"),
                                                 Engagement=("EngagementScore", "mean")).round(2)
print(band)
coef = np.polyfit(df.WeeklyHours, df.TasksCompleted, 3); xs = np.linspace(df.WeeklyHours.min(), df.WeeklyHours.max(), 200)
ys = np.polyval(coef, xs); sweet = xs[ys.argmax()]
print(f"\nTasks completed peak at about {sweet:.0f} hours/week - extra hours beyond this add no output.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(df.WeeklyHours, df.TasksCompleted, s=6, alpha=.25, color="grey"); ax[0].plot(xs, ys, color="#D9534F", lw=3)
ax[0].axvline(sweet, color="green", ls="--", label=f"Sweet spot ~{sweet:.0f} h"); ax[0].legend()
ax[0].set_xlabel("Weekly hours"); ax[0].set_ylabel("Tasks completed"); ax[0].set_title("Diminishing returns on hours")
sns.barplot(data=band.reset_index(), x="HoursBand", y="OutputPerHour", palette="RdYlGn_r", ax=ax[1])
ax[1].set_title("Output per hour by weekly hours"); plt.tight_layout()
plt.savefig("outputs/2_workload_curve.png", dpi=150); plt.close()

# ------------------------------------------------ 3. BOTTLENECKS BY TEAM
print("\n" + "=" * 70, "\n3. OPERATIONAL BOTTLENECKS BY TEAM\n" + "=" * 70)
team = df.groupby("Team").agg(Hours=("WeeklyHours", "mean"), Overtime=("OvertimeHrs", "mean"), Absent=("AbsentDays", "mean"),
                             Meetings=("MeetingHrsPerWeek", "mean"), Engagement=("EngagementScore", "mean"),
                             CompletionRate=("CompletionRate", "mean"), OutputPerHour=("OutputPerHour", "mean")).round(2)
team["BottleneckScore"] = ((team.Overtime - team.Overtime.mean()) / team.Overtime.std()
                           - (team.OutputPerHour - team.OutputPerHour.mean()) / team.OutputPerHour.std()
                           + (team.Absent - team.Absent.mean()) / team.Absent.std()).round(2)
team = team.sort_values("BottleneckScore", ascending=False); print(team)
plt.figure(figsize=(8, 4.5)); sns.barplot(x=team.BottleneckScore, y=team.index, palette="Reds_r")
plt.title("Bottleneck score by team (higher = worse)"); plt.tight_layout(); plt.savefig("outputs/3_team_bottlenecks.png", dpi=150); plt.close()

# ------------------------------------------------ 4. DRIVERS (ML)
print("\n" + "=" * 70, "\n4. WHAT DRIVES PRODUCTIVITY? (Random Forest)\n" + "=" * 70)
feat = ["WeeklyHours", "OvertimeHrs", "AbsentDays", "EngagementScore", "MeetingHrsPerWeek", "TenureYears", "PerformanceRating"]
Xtr, Xte, ytr, yte = train_test_split(df[feat], df.OutputPerHour, test_size=.25, random_state=1)
rf = RandomForestRegressor(300, max_depth=8, random_state=1).fit(Xtr, ytr)
print(f"Model R2 on unseen data: {r2_score(yte, rf.predict(Xte)):.2f}")
imp = pd.Series(permutation_importance(rf, Xte, yte, n_repeats=10, random_state=1).importances_mean, index=feat).sort_values(ascending=False)
print(imp.round(4))
plt.figure(figsize=(7, 4)); imp[::-1].plot.barh(color="#4C78A8"); plt.title("Drivers of productivity (permutation importance)")
plt.tight_layout(); plt.savefig("outputs/4_productivity_drivers.png", dpi=150); plt.close()

# ------------------------------------------------ 5. EMPLOYEE SEGMENTS
print("\n" + "=" * 70, "\n5. EMPLOYEE WORKLOAD SEGMENTS (K-Means)\n" + "=" * 70)
last = df[df.Month >= 10].groupby("EmployeeID")[["WeeklyHours", "AbsentDays", "EngagementScore", "OutputPerHour"]].mean()
Z = StandardScaler().fit_transform(last); km = KMeans(4, n_init=10, random_state=1).fit(Z); last["Cluster"] = km.labels_
prof = last.groupby("Cluster").mean().round(2); prof["Employees"] = last.Cluster.value_counts().sort_index()
names = {}
for c, r in prof.iterrows():
    names[c] = ("Overloaded / burning out" if r.WeeklyHours >= prof.WeeklyHours.quantile(.75) and r.EngagementScore <= prof.EngagementScore.median()
                else "Healthy high performers" if r.OutputPerHour >= prof.OutputPerHour.quantile(.75)
                else "Absence-prone / low output" if r.AbsentDays >= prof.AbsentDays.quantile(.75)
                else "Steady")
prof["Segment"] = prof.index.map(names); last["Segment"] = last.Cluster.map(names); print(prof.to_string())
plt.figure(figsize=(7.5, 5)); sns.scatterplot(data=last, x="WeeklyHours", y="OutputPerHour", hue="Segment", palette="Set1", s=40)
plt.title("Employee segments: hours vs productivity"); plt.tight_layout(); plt.savefig("outputs/5_segments.png", dpi=150); plt.close()

# ------------------------------------------------ 6. WHAT-IF + RECOMMENDATIONS
print("\n" + "=" * 70, "\n6. WORKLOAD INTERVENTIONS & EXPECTED IMPACT\n" + "=" * 70)
cap = 45; over = df[df.WeeklyHours > cap]
sim = df.copy(); sim.loc[sim.WeeklyHours > cap, "WeeklyHours"] = cap; sim.loc[sim.WeeklyHours <= cap, "OvertimeHrs"] = sim.OvertimeHrs.clip(upper=cap - 40)
before = df.TasksCompleted.sum(); after = (rf.predict(sim[feat]) * sim.WeeklyHours).sum()
saved = (over.WeeklyHours - cap).sum() / df.WeeklyHours.sum() * 100
print(f"{len(over)/len(df)*100:.0f}% of employee-months exceed {cap} hrs/week.")
print(f"Capping at {cap} h would cut total hours by {saved:.1f}% while predicted output changes {(after/before-1)*100:+.1f}%.")
worst = team.index[:2].tolist()
lines = ["WORKLOAD & PRODUCTIVITY REPORT", "=" * 40,
         f"Hours {(h1/h0-1)*100:+.1f}% but tasks completed {(t1/t0-1)*100:+.1f}% and output per hour {(p1/p0-1)*100:+.1f}% over 12 months.",
         f"Productivity peaks near {sweet:.0f} h/week; beyond ~45 h output per hour drops sharply.",
         f"Biggest bottleneck teams: {', '.join(worst)}.", f"Top productivity drivers: {', '.join(imp.index[:3])}.", "",
         "RECOMMENDED INTERVENTIONS:",
         f"1. Cap planned hours at {cap}/week; rebalance work in {worst[0]} and {worst[1]} (hire or redistribute).",
         "2. Treat rising absenteeism as an early-warning signal (top productivity driver): HR check-ins and wellbeing support.",
         "3. Run engagement check-ins for the 'Overloaded / burning out' segment before absenteeism rises.",
         "4. Track output per hour (not hours worked) as the headline productivity KPI."]
rep = "\n".join(lines); open("outputs/workload_report.txt", "w").write(rep); print("\n" + rep)

if os.getenv("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        m = anthropic.Anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=500, messages=[{"role": "user",
            "content": "Write a short executive summary with 3 prioritised actions for this workload analysis:\n\n" + rep}])
        print("\n" + "=" * 70, "\n7. LLM EXECUTIVE SUMMARY\n" + "=" * 70); print(m.content[0].text)
    except Exception as e:
        print("(LLM step skipped:", e, ")")
else:
    print("\n(Optional: set ANTHROPIC_API_KEY for an LLM-written executive summary.)")
print("Done. Charts saved in ./outputs/")
