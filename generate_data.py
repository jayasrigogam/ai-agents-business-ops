"""Step 1: synthetic supply-chain data -> supply_data.csv (40 suppliers x 24 months)
Each row = one supplier-month. 'DisruptionNextMonth' = 1 if that supplier's delivery is badly disrupted next month."""
import numpy as np, pandas as pd
rng = np.random.default_rng(21)
REGIONS = ["Asia", "Europe", "North America", "Middle East"]
CATS = ["Raw Materials", "Electronics", "Packaging", "Components"]
S = 40
sup = pd.DataFrame({"SupplierID": [f"S{i:02d}" for i in range(1, S + 1)],
                    "Region": rng.choice(REGIONS, S, p=[.4, .25, .2, .15]),
                    "Category": rng.choice(CATS, S),
                    "SingleSource": rng.random(S) < 0.3,
                    "FinancialRisk": np.clip(rng.beta(2, 5, S), 0.02, 0.95).round(2),
                    "BaseLeadTime": rng.integers(7, 35, S)})
rows = []
for _, s in sup.iterrows():
    stress = 0.0
    for m in range(1, 25):
        stress = 0.7 * stress + rng.normal(0, 0.6) + (0.5 if (s.Region == "Middle East" and 12 <= m <= 18) else 0)  # regional event
        lead = s.BaseLeadTime * (1 + 0.08 * stress + rng.normal(0, 0.04))
        transport = max(rng.normal(1 + 1.5 * stress + (s.Region == "Asia") * .8, 1), 0)
        ontime = np.clip(0.95 - 0.07 * stress - 0.1 * s.FinancialRisk + rng.normal(0, .03), .4, 1)
        defect = np.clip(0.012 + 0.008 * stress + 0.03 * s.FinancialRisk + rng.normal(0, .006), 0.001, .2)
        cover = np.clip(rng.normal(28 - 6 * stress - 6 * s.SingleSource, 4), 2, 60)       # days of inventory cover
        util = np.clip(rng.normal(0.75 + 0.05 * stress, .08), .3, 1.1)                     # production schedule load
        z = (-3.4 + 1.1 * stress + 2.4 * s.FinancialRisk - 0.07 * (cover - 20) + 0.25 * transport
             + 1.0 * s.SingleSource + 2.0 * (util - .75) - 3 * (ontime - .9))
        disrupt = int(rng.random() < 1 / (1 + np.exp(-z)))
        rows.append((s.SupplierID, m, round(lead, 1), round(transport, 1), round(ontime, 3), round(defect, 4),
                     round(cover, 1), round(util, 2), disrupt))
d = pd.DataFrame(rows, columns=["SupplierID", "Month", "LeadTimeDays", "TransportDelayDays", "OnTimeRate", "DefectRate",
                                "InventoryCoverDays", "ProductionLoad", "_Disrupted"])
d["DisruptionNextMonth"] = d.groupby("SupplierID")._Disrupted.shift(-1)       # signals today -> disruption next month
df = d.merge(sup, on="SupplierID").drop(columns="_Disrupted")
df.to_csv("supply_data.csv", index=False)
print(f"supply_data.csv created: {len(df)} supplier-months, {S} suppliers, 24 months")
print(f"Disruption rate: {df.DisruptionNextMonth.dropna().mean():.1%}")
