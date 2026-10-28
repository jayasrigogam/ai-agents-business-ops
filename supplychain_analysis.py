"""Step 2: Supply Chain Disruption Prediction & Mitigation
Run: python supplychain_analysis.py -> terminal results + charts in ./outputs/ + outputs/supply_risk_report.txt
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
warnings.filterwarnings("ignore"); os.makedirs("outputs", exist_ok=True); sns.set_theme(style="whitegrid")

df = pd.read_csv("supply_data.csv")
df = df.sort_values(["SupplierID", "Month"])
base = df.groupby("SupplierID").BaseLeadTime.transform("first")
df["LeadTimeRatio"] = df.LeadTimeDays / base                                     # >1 = slower than normal
df["LeadTimeTrend3m"] = df.groupby("SupplierID").LeadTimeRatio.diff(3).fillna(0)  # early-warning: rising lead time
df["CoverTrend3m"] = df.groupby("SupplierID").InventoryCoverDays.diff(3).fillna(0)
df["SingleSourceFlag"] = df.SingleSource.astype(int)
FEATS = ["LeadTimeRatio", "LeadTimeTrend3m", "TransportDelayDays", "OnTimeRate", "DefectRate", "InventoryCoverDays",
         "CoverTrend3m", "ProductionLoad", "FinancialRisk", "SingleSourceFlag"]
hist = df.dropna(subset=["DisruptionNextMonth"]).copy(); hist["y"] = hist.DisruptionNextMonth.astype(int)

# ---------------------------------------------------- 1. OVERVIEW
print("=" * 70, "\n1. DISRUPTION OVERVIEW\n" + "=" * 70)
print(f"Supplier-months analysed: {len(hist)} | Disruption rate: {hist.y.mean():.1%}")
reg = hist.groupby("Region").y.mean().sort_values(ascending=False) * 100
cat = hist.groupby("Category").y.mean().sort_values(ascending=False) * 100
print("\nBy region (%):\n", reg.round(1).to_string()); print("\nBy category (%):\n", cat.round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sns.barplot(x=reg.values, y=reg.index, palette="Reds_r", ax=ax[0]); ax[0].set_title("Disruption rate by region (%)")
sns.barplot(x=cat.values, y=cat.index, palette="Oranges_r", ax=ax[1]); ax[1].set_title("Disruption rate by category (%)")
plt.tight_layout(); plt.savefig("outputs/1_disruption_by_region_category.png", dpi=150); plt.close()

# ---------------------------------------------------- 2. EARLY-WARNING SIGNALS
print("\n" + "=" * 70, "\n2. WHAT DO SUPPLIERS LOOK LIKE BEFORE A DISRUPTION?\n" + "=" * 70)
sig = hist.groupby("y")[FEATS].mean().T.round(3); sig.columns = ["Normal", "Before_Disruption"]
sig["Change_%"] = ((sig.Before_Disruption / sig.Normal - 1) * 100).round(1)
sig["Change_%"] = sig["Change_%"].astype(object).where(sig.Normal.abs() >= 0.1, "n/a (trend)"); print(sig)
show = ["TransportDelayDays", "InventoryCoverDays", "OnTimeRate", "LeadTimeRatio"]
fig, ax = plt.subplots(1, 4, figsize=(14, 4))
for a, c in zip(ax, show):
    sns.boxplot(data=hist, x="y", y=c, palette=["#4C9F70", "#D9534F"], ax=a); a.set_xticklabels(["Normal", "Disrupted"]); a.set_xlabel(""); a.set_title(c)
plt.suptitle("Early-warning signals: the month BEFORE a disruption", fontsize=13); plt.tight_layout()
plt.savefig("outputs/2_early_warning_signals.png", dpi=150); plt.close()

# ---------------------------------------------------- 3. PREDICTION MODEL (time-based split)
print("\n" + "=" * 70, "\n3. PREDICTION MODEL (train on months 1-17, test on months 18-23)\n" + "=" * 70)
tr, te = hist[hist.Month <= 17], hist[hist.Month >= 18]
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05, random_state=3).fit(tr[FEATS], tr.y)
p = gb.predict_proba(te[FEATS])[:, 1]; THR = 0.30; pred = (p >= THR).astype(int)
print(f"ROC-AUC on unseen future months: {roc_auc_score(te.y, p):.3f}")
print(f"Alert threshold: risk >= {THR:.0%}"); print(classification_report(te.y, pred, target_names=["No disruption", "Disruption"]))
print("Confusion matrix:\n", confusion_matrix(te.y, pred))
imp = pd.Series(permutation_importance(gb, te[FEATS], te.y, scoring="roc_auc", n_repeats=10, random_state=3).importances_mean,
                index=FEATS).sort_values(ascending=False)
print("\nTop risk drivers (permutation importance):\n", imp.head(6).round(4).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); fpr, tpr, _ = roc_curve(te.y, p)
ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(te.y, p):.2f}"); ax[0].plot([0, 1], [0, 1], "k--"); ax[0].legend()
ax[0].set_title("ROC curve (future months)"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
imp.head(8)[::-1].plot.barh(color="#D9534F", ax=ax[1]); ax[1].set_title("Top risk drivers"); plt.tight_layout()
plt.savefig("outputs/3_model_performance.png", dpi=150); plt.close()

# ---------------------------------------------------- 4. CURRENT RISK (latest month)
print("\n" + "=" * 70, "\n4. CURRENT SUPPLIER RISK (month 24 - forecast for month 25)\n" + "=" * 70)
now = df[df.Month == 24].copy(); now["Risk"] = gb.predict_proba(now[FEATS])[:, 1]
now["Band"] = pd.cut(now.Risk, [0, .2, .4, 1.01], labels=["Low", "Medium", "High"], right=False)
print(now.Band.value_counts().to_string())
top = now.sort_values("Risk", ascending=False).head(10)
print("\nTop 10 suppliers most likely to be disrupted next month:")
print(top[["SupplierID", "Region", "Category", "InventoryCoverDays", "TransportDelayDays", "OnTimeRate", "Risk"]].round(2).to_string(index=False))
t15 = now.sort_values("Risk", ascending=False).head(15).iloc[::-1]
plt.figure(figsize=(8, 5.5)); plt.barh(t15.SupplierID, t15.Risk * 100, color=t15.Band.map({"High": "#D9534F", "Medium": "#F0AD4E", "Low": "#4C9F70"}))
plt.xlabel("Predicted disruption risk (%)"); plt.title("Top 15 at-risk suppliers"); plt.tight_layout()
plt.savefig("outputs/4_supplier_risk_ranking.png", dpi=150); plt.close()

# ---------------------------------------------------- 5. MITIGATION ENGINE
print("\n" + "=" * 70, "\n5. RECOMMENDED MITIGATION ACTIONS\n" + "=" * 70)
def actions(r):
    a = []
    if r.InventoryCoverDays < 18: a.append("Raise safety stock / expedite open orders")
    if r.SingleSourceFlag == 1 and r.Risk >= .2: a.append("Qualify a backup supplier (dual sourcing)")
    if r.TransportDelayDays > 3: a.append("Switch carrier or re-route shipments")
    if r.DefectRate > .035: a.append("Schedule supplier quality audit")
    if r.FinancialRisk > .45: a.append("Financial health review / shorter payment terms")
    if r.ProductionLoad > .9: a.append("Re-sequence production schedule")
    return a or ["Monitor"]
at_risk = now[now.Band != "Low"].copy(); at_risk["Actions"] = at_risk.apply(actions, axis=1)
cnt = pd.Series([x for l in at_risk.Actions for x in l]).value_counts(); print(cnt.to_string())
print("\nExample plans for the 5 riskiest suppliers:")
for _, r in at_risk.sort_values("Risk", ascending=False).head(5).iterrows():
    print(f"- {r.SupplierID} ({r.Region}, {r.Category}) risk {r.Risk:.0%}: " + "; ".join(r.Actions))
plt.figure(figsize=(9, 4.5)); cnt[::-1].plot.barh(color="#4C78A8"); plt.xlabel("Suppliers needing this action")
plt.title("Mitigation actions for at-risk suppliers"); plt.tight_layout(); plt.savefig("outputs/5_mitigation_actions.png", dpi=150); plt.close()

lines = ["SUPPLY CHAIN DISRUPTION REPORT", "=" * 40,
         f"Model ROC-AUC on future months: {roc_auc_score(te.y, p):.2f}",
         f"Highest-risk region: {reg.index[0]} ({reg.iloc[0]:.0f}% disruption rate); category: {cat.index[0]}",
         f"Top risk drivers: {', '.join(imp.index[:3])}",
         f"Suppliers at HIGH risk next month: {(now.Band == 'High').sum()} | MEDIUM: {(now.Band == 'Medium').sum()}", "", "PRIORITY ACTIONS:"]
lines += [f"- {a}: {n} suppliers" for a, n in cnt.head(4).items()]
rep = "\n".join(lines); open("outputs/supply_risk_report.txt", "w").write(rep); print("\n" + rep)
if os.getenv("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        m = anthropic.Anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=500, messages=[{"role": "user",
            "content": "Write a short executive summary with 3 prioritised actions for this supply-chain risk analysis:\n\n" + rep}])
        print("\n" + "=" * 70, "\n6. LLM EXECUTIVE SUMMARY\n" + "=" * 70); print(m.content[0].text)
    except Exception as e:
        print("(LLM step skipped:", e, ")")
else:
    print("\n(Optional: set ANTHROPIC_API_KEY for an LLM-written executive summary.)")
print("Done. Charts saved in ./outputs/")
