"""Analyze machine failure risk and produce charts and maintenance recommendations."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.inspection import permutation_importance

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def recommendation(row):
    actions = []
    if row["vibration_mm_s"] >= 5.0:
        actions.append("inspect bearings/alignment; check vibration")
    if row["temperature_c"] >= 82:
        actions.append("inspect cooling and lubrication")
    if row["hours_since_maintenance"] >= 1500:
        actions.append("schedule preventive service")
    if row["recent_faults_90d"] >= 2:
        actions.append("review repeat fault history")
    if row["load_pct"] >= 95:
        actions.append("review overloading and operating conditions")
    if not actions:
        actions.append("continue routine monitoring")
    if row["production_criticality"] == "High":
        actions.append("prioritize due to high production criticality")
    return "; ".join(actions)

def main():
    data_path = Path("machine_data.csv")
    if not data_path.exists():
        raise SystemExit("machine_data.csv not found. Run: python generate_data.py")

    df = pd.read_csv(data_path)
    target = "failure_next_30d"
    numeric = [
        "age_years", "operating_hours", "hours_since_maintenance",
        "temperature_c", "vibration_mm_s", "pressure_kpa", "load_pct",
        "recent_faults_90d", "downtime_hours_last_month"
    ]
    categorical = ["machine_type", "production_criticality"]
    X = df[numeric + categorical]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=9, stratify=y
    )
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([
        ("prep", pre),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)

    # Score all machines and create an action list.
    df["risk_score_pct"] = (model.predict_proba(X)[:, 1] * 100).round(1)
    df["risk_level"] = pd.cut(df["risk_score_pct"], [-1, 30, 60, 100],
                              labels=["Low", "Medium", "High"])
    df["recommended_action"] = df.apply(recommendation, axis=1)
    ranked = df.sort_values("risk_score_pct", ascending=False)
    ranked.to_csv(OUT / "machine_risk_rankings.csv", index=False)

    # 1. Failure rate by machine type
    rates = df.groupby("machine_type", observed=False)[target].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(9, 5))
    rates.plot(kind="bar")
    plt.ylabel("Failure rate (%)")
    plt.title("Observed Failure Rate by Machine Type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "1_failure_rate_by_machine_type.png", dpi=150)
    plt.close()

    # 2. Sensor patterns by label
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for label, group in df.groupby(target):
        name = "Failure in 30 days" if label == 1 else "No labeled failure"
        axes[0].hist(group["temperature_c"], bins=25, alpha=0.55, label=name)
        axes[1].hist(group["vibration_mm_s"], bins=25, alpha=0.55, label=name)
    axes[0].set_title("Temperature (°C)")
    axes[1].set_title("Vibration (mm/s)")
    for ax in axes:
        ax.legend()
        ax.set_ylabel("Machine count")
    fig.suptitle("Sensor Readings by Failure Label")
    fig.tight_layout()
    fig.savefig(OUT / "2_sensor_patterns.png", dpi=150)
    plt.close(fig)

    # 3. Model performance: ROC and confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[0])
    axes[0].set_title(f"ROC Curve (AUC = {auc:.3f})")
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels=["No failure", "Failure"]).plot(
        ax=axes[1], colorbar=False
    )
    axes[1].set_title("Confusion Matrix (threshold = 0.50)")
    fig.tight_layout()
    fig.savefig(OUT / "3_model_performance.png", dpi=150)
    plt.close(fig)

    # 4. Highest risk machines
    top = ranked.head(15).sort_values("risk_score_pct")
    plt.figure(figsize=(9, 6))
    plt.barh(top["machine_id"], top["risk_score_pct"])
    plt.xlabel("Predicted risk score (%)")
    plt.title("15 Highest-Risk Machines")
    plt.tight_layout()
    plt.savefig(OUT / "4_highest_risk_machines.png", dpi=150)
    plt.close()

    # 5. Action distribution among high risk machines
    high = ranked.head(100)
    counts = high["recommended_action"].str.split("; ").explode().value_counts().head(10).sort_values()
    plt.figure(figsize=(10, 6))
    counts.plot(kind="barh")
    plt.xlabel("Count among top 100 risk-ranked machines")
    plt.title("Common Recommended Actions for Highest-Risk Machines")
    plt.tight_layout()
    plt.savefig(OUT / "5_maintenance_recommendations.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, preds)
    high_count = int((df["risk_level"] == "High").sum())
    med_count = int((df["risk_level"] == "Medium").sum())
    low_count = int((df["risk_level"] == "Low").sum())
    lines = [
        "PROJECT 9 - PREDICTIVE MAINTENANCE REPORT",
        "=" * 46,
        f"Machines analyzed: {len(df)}",
        f"Labeled failures in synthetic dataset: {int(y.sum())} ({y.mean()*100:.1f}%)",
        f"Test ROC-AUC: {auc:.3f}",
        f"Confusion matrix at threshold 0.50: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
        "",
        "Risk-score bands (model-estimated):",
        f"  High (>60%): {high_count}",
        f"  Medium (30-60%): {med_count}",
        f"  Low (<=30%): {low_count}",
        "",
        "TOP 10 MACHINES TO REVIEW FIRST",
        "-" * 32
    ]
    for _, row in ranked.head(10).iterrows():
        lines.append(
            f"{row['machine_id']} | type={row['machine_type']} | risk={row['risk_score_pct']}% | "
            f"criticality={row['production_criticality']} | action={row['recommended_action']}"
        )
    lines += [
        "",
        "Suggested operational use:",
        "- Inspect high-risk machines first, especially those with high production criticality.",
        "- Validate sensor readings and maintenance history before scheduling downtime.",
        "- Review recurring faults, overheating, vibration, and overdue maintenance.",
        "- Re-train and validate the model with real plant data before operational use.",
        "",
        "IMPORTANT: This dataset is synthetic and results are for demonstration only."
    ]
    (OUT / "maintenance_risk_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Predictive maintenance analysis complete.")
    print(f"Machines analyzed: {len(df)}")
    print(f"Test ROC-AUC: {auc:.3f}")
    print(f"High risk: {high_count} | Medium risk: {med_count} | Low risk: {low_count}")
    print(f"Charts and report saved in: {OUT.resolve()}")
    print("Top 5 machines:")
    print(ranked[["machine_id", "machine_type", "risk_score_pct", "recommended_action"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
