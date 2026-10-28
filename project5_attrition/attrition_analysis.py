"""Step 2: Employee Attrition Root-Cause Intelligence
Run:  python attrition_analysis.py
Outputs: terminal results + PNG charts in ./outputs/ + outputs/retention_report.txt
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("hr_data.csv")
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
print("=" * 70, "\n1. DATA OVERVIEW\n" + "=" * 70)
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print(f"Overall attrition rate: {df.AttritionFlag.mean():.1%}")

# ---------------------------------------------------------------- 2. EDA
print("\n" + "=" * 70, "\n2. ATTRITION BY DEPARTMENT\n" + "=" * 70)
dept_rate = df.groupby("Department")["AttritionFlag"].mean().sort_values(ascending=False)
print((dept_rate * 100).round(1).astype(str) + " %")

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=dept_rate.values * 100, y=dept_rate.index, palette="Reds_r", ax=ax)
ax.set_xlabel("Attrition rate (%)"); ax.set_title("Attrition Rate by Department")
plt.tight_layout(); plt.savefig("outputs/1_attrition_by_department.png", dpi=150); plt.close()

num_cols = ["Age", "TenureYears", "MonthlySalary_INR_equiv", "OvertimeHrsPerMonth", "WorkloadScore",
            "AbsentDays", "EngagementScore", "PerformanceRating", "TrainingHrsPerYear", "YearsSincePromotion"]

print("\nAverage values: Stayed vs Left")
comp = df.groupby("Attrition")[num_cols].mean().T.round(2)
comp["Diff_%"] = ((comp["Yes"] - comp["No"]) / comp["No"] * 100).round(1)
print(comp)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), ["OvertimeHrsPerMonth", "EngagementScore", "MonthlySalary_INR_equiv",
                                "TrainingHrsPerYear", "AbsentDays", "YearsSincePromotion"]):
    sns.boxplot(data=df, x="Attrition", y=c, palette={"No": "#4C9F70", "Yes": "#D9534F"}, ax=ax)
    ax.set_title(c)
plt.suptitle("Stayed vs Left - key factors", fontsize=14); plt.tight_layout()
plt.savefig("outputs/2_factors_stayed_vs_left.png", dpi=150); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols + ["AttritionFlag"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("outputs/3_correlation_heatmap.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 3. MODEL
print("\n" + "=" * 70, "\n3. MACHINE LEARNING MODELS\n" + "=" * 70)
X = pd.get_dummies(df[num_cols + ["Department"]], drop_first=True)
y = df["AttritionFlag"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
rf.fit(Xtr, ytr)
rf_prob = rf.predict_proba(Xte)[:, 1]; rf_pred = rf.predict(Xte)

sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
lr_prob = lr.predict_proba(sc.transform(Xte))[:, 1]

print(f"Random Forest      -> Accuracy {accuracy_score(yte, rf_pred):.3f} | ROC-AUC {roc_auc_score(yte, rf_prob):.3f}")
print(f"Logistic Regression-> Accuracy {accuracy_score(yte, lr.predict(sc.transform(Xte))):.3f} | ROC-AUC {roc_auc_score(yte, lr_prob):.3f}")
print("\nClassification report (Random Forest):")
print(classification_report(yte, rf_pred, target_names=["Stayed", "Left"]))
print("Confusion matrix:\n", confusion_matrix(yte, rf_pred))

fig, ax = plt.subplots(figsize=(6, 5))
for name, pr in [("Random Forest", rf_prob), ("Logistic Regression", lr_prob)]:
    fpr, tpr, _ = roc_curve(yte, pr); ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, pr):.2f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(); plt.tight_layout()
plt.savefig("outputs/4_roc_curve.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 4. ROOT CAUSES
print("\n" + "=" * 70, "\n4. ROOT-CAUSE ANALYSIS (what drives attrition?)\n" + "=" * 70)
perm = permutation_importance(rf, Xte, yte, n_repeats=15, random_state=42, scoring="roc_auc")
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("Permutation importance (drop in AUC when feature is shuffled):")
print(imp.head(8).round(4))

lr_coef = pd.Series(lr.coef_[0], index=X.columns).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
imp.head(10)[::-1].plot.barh(color="#D9534F", ax=axes[0]); axes[0].set_title("Top Drivers (Random Forest)")
lr_coef.plot.barh(color=["#4C9F70" if v < 0 else "#D9534F" for v in lr_coef], ax=axes[1])
axes[1].set_title("Direction of Effect (Logistic Regression)\nred = raises attrition, green = lowers")
plt.tight_layout(); plt.savefig("outputs/5_root_causes.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. RISK SCORING
df["AttritionRisk"] = rf.predict_proba(X)[:, 1]
df["RiskBand"] = pd.cut(df["AttritionRisk"], [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
print("\n" + "=" * 70, "\n5. EMPLOYEE RISK SEGMENTS\n" + "=" * 70)
print(df["RiskBand"].value_counts())
cur = df[df.Attrition == "No"]
top_risk = cur.sort_values("AttritionRisk", ascending=False).head(10)[
    ["EmployeeID", "Department", "OvertimeHrsPerMonth", "EngagementScore", "TrainingHrsPerYear", "AttritionRisk"]]
print("\nTop 10 current employees most at risk of leaving:")
print(top_risk.round(2).to_string(index=False))

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Department", hue="RiskBand", palette=["#4C9F70", "#F0AD4E", "#D9534F"])
plt.title("Risk Bands by Department"); plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("outputs/6_risk_by_department.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. RECOMMENDATIONS
print("\n" + "=" * 70, "\n6. EVIDENCE-BASED RETENTION RECOMMENDATIONS\n" + "=" * 70)
rules = {
    "OvertimeHrsPerMonth": "Cap overtime (e.g. max 10 hrs/month), hire/redistribute work in overloaded teams.",
    "MonthlySalary_INR_equiv": "Run a pay-benchmarking review; correct pay below department median.",
    "EngagementScore": "Launch pulse surveys, manager 1:1s and recognition programmes for low-engagement staff.",
    "TrainingHrsPerYear": "Guarantee a minimum training budget (e.g. 30 hrs/yr) and personal development plans.",
    "AbsentDays": "Treat rising absenteeism as an early-warning sign; HR check-ins and wellbeing support.",
    "YearsSincePromotion": "Define clear promotion/career-path timelines; review staff waiting > 3 years.",
    "TenureYears": "Strengthen onboarding & mentoring for new joiners (early-tenure exits).",
    "WorkloadScore": "Rebalance task allocation; use workload dashboards to spot bottlenecks.",
    "Age": "Offer life-stage-aware benefits (flexibility, learning, career coaching).",
    "PerformanceRating": "Link performance reviews to growth conversations and rewards.",
}
lines = ["EMPLOYEE ATTRITION - ROOT-CAUSE & RETENTION REPORT", "=" * 55,
         f"Employees analysed: {len(df)} | Attrition rate: {df.AttritionFlag.mean():.1%}",
         f"Highest-attrition department: {dept_rate.index[0]} ({dept_rate.iloc[0]:.1%})",
         f"Model quality (Random Forest ROC-AUC): {roc_auc_score(yte, rf_prob):.2f}", "",
         "TOP ATTRITION DRIVERS & ACTIONS:"]
rank = 0
for feat in imp.index:
    if feat in rules and rank < 5:
        rank += 1
        g = comp.loc[feat] if feat in comp.index else None
        ev = f" (leavers avg {g['Yes']} vs stayers {g['No']})" if g is not None else ""
        lines.append(f"{rank}. {feat}{ev}\n   -> {rules[feat]}")
n_high = int((cur["RiskBand"] == "High").sum())
lines += ["", f"IMMEDIATE ACTION: {n_high} current employees are in the HIGH-risk band. Prioritise stay-interviews for them."]
report = "\n".join(lines)
print(report)
open("outputs/retention_report.txt", "w").write(report)

# ------------------------------------------------------------- 7. OPTIONAL LLM
key = os.getenv("ANTHROPIC_API_KEY")
if key:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=600,
            messages=[{"role": "user", "content": "You are an HR analytics advisor. Turn this analysis into a "
                       "short executive summary with 3 prioritised actions:\n\n" + report}])
        print("\n" + "=" * 70, "\n7. LLM EXECUTIVE SUMMARY\n" + "=" * 70)
        print(msg.content[0].text)
    except Exception as e:
        print("\n(LLM step skipped:", e, ")")
else:
    print("\n(Optional: set ANTHROPIC_API_KEY to add an LLM-written executive summary.)")
print("\nDone. Charts saved in ./outputs/")
