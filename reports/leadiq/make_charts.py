import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

PD = "/app/backend/project_data/653622fa8476494fadbc62df2bfa405b"
OUT = "/app/reports/leadiq/assets"
os.makedirs(OUT, exist_ok=True)

m = json.load(open(f"{PD}/metrics.json"))
imp = json.load(open(f"{PD}/importance.json"))
ss = json.load(open(f"{PD}/score_summary.json"))

INK = "#1e1b2e"; VIO = "#6d5bd0"; VIO2 = "#a594f2"; GREY = "#c9c6d6"
ACC = "#e0592a"; GRID = "#ece9f5"; TEAL = "#2bb3a3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#b9b4cc",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlecolor": INK, "figure.dpi": 150,
})

def style(ax):
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)

models = m["models"]
best = m["best"]

# 1. Model comparison (ROC-AUC & PR-AUC) bar
names = [x["name"] for x in models]
roc = [x["roc_auc"] for x in models]
pr = [x["pr_auc"] for x in models]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = np.arange(len(names)); h = 0.38
ax.barh(y + h/2, roc, h, label="ROC-AUC", color=VIO)
ax.barh(y - h/2, pr, h, label="PR-AUC (avg precision)", color=ACC)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.invert_yaxis(); ax.set_xlim(0, 1)
for i,(r,p) in enumerate(zip(roc,pr)):
    ax.text(r+0.01, i+h/2, f"{r:.3f}", va="center", fontsize=9)
    ax.text(p+0.01, i-h/2, f"{p:.3f}", va="center", fontsize=9)
style(ax); ax.set_xlabel("Score"); ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("Model comparison on the untouched test set", loc="left", fontweight="bold")
save(fig, "chart_model_compare.png")

# 2. ROC curve of winner
roc_pts = best["roc"]
xs = [p["x"] for p in roc_pts]; ys = [p["y"] for p in roc_pts]
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot(xs, ys, color=VIO, lw=2.4, label=f"{best['name']} (AUC = {best['roc_auc']:.3f})")
ax.plot([0,1],[0,1], color=GREY, lw=1.6, ls="--", label="No-skill")
style(ax); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("ROC curve — selected model", loc="left", fontweight="bold")
save(fig, "chart_roc.png")

# 3. PR curve of winner
pr_pts = best.get("pr") or best.get("precision_recall") or []
prev = m["prevalence"]
fig, ax = plt.subplots(figsize=(5.2, 4.4))
if pr_pts:
    xs = [p["x"] for p in pr_pts]; ys = [p["y"] for p in pr_pts]
    ax.plot(xs, ys, color=ACC, lw=2.4, label=f"{best['name']} (AP = {best['pr_auc']:.3f})")
ax.axhline(prev, color=GREY, lw=1.6, ls="--", label=f"Prevalence = {prev:.3f}")
style(ax); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title("Precision–Recall curve — selected model", loc="left", fontweight="bold")
save(fig, "chart_pr.png")

# 4. Confusion matrix
cm = np.array(best["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(cm, cmap="Purples")
labels = [["TN","FP"],["FN","TP"]]
mx = cm.max()
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white" if cm[i,j]>mx*0.5 else INK)
ax.set_xticks([0,1]); ax.set_xticklabels(["Pred: no","Pred: yes"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Actual: no","Actual: yes"])
ax.set_title(f"Confusion matrix (threshold = {best['threshold']:.3f})", loc="left", fontweight="bold")
save(fig, "chart_confusion.png")

# 5. Lift & cumulative gain
gain = m.get("gain", {})
gcurve = gain.get("curve", [])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
if gcurve:
    gx = [p["x"] for p in gcurve]; gy = [p["y"] for p in gcurve]
    a1.plot(gx, gy, color=VIO, lw=2.4, marker="o", ms=3, label="LeadIQ model")
    a1.plot([0,1],[0,1], color=GREY, ls="--", lw=1.6, label="Random")
    a1.set_title("Cumulative gain", loc="left", fontweight="bold")
    a1.set_xlabel("Fraction of leads contacted"); a1.set_ylabel("Fraction of conversions captured")
    a1.set_xlim(0,1); a1.set_ylim(0,1); a1.legend(frameon=False, fontsize=9); style(a1)
    lx = gx[1:]; ly = [ (y/x if x>0 else 0) for x,y in zip(gx[1:], gy[1:]) ]
    a2.plot(lx, ly, color=ACC, lw=2.4, marker="o", ms=3, label="LeadIQ model")
    a2.axhline(1.0, color=GREY, ls="--", lw=1.6, label="Baseline lift = 1")
    a2.set_title("Lift curve", loc="left", fontweight="bold")
    a2.set_xlabel("Fraction of leads contacted"); a2.set_ylabel("Lift over random")
    a2.set_xlim(0,1); a2.legend(frameon=False, fontsize=9); style(a2)
save(fig, "chart_lift_gain.png")

# 6. Calibration reliability (before/after)
cal = m.get("calibration", {})
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for key, col, lab in [("before", GREY, "Before calibration"), ("after", VIO, "After isotonic calibration")]:
    pts = cal.get(key)
    if pts:
        xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
        ax.plot(xs, ys, color=col, lw=2.2, marker="o", ms=3, label=lab)
ax.plot([0,1],[0,1], color="#9a95ad", ls="--", lw=1.4, label="Perfect calibration")
style(ax); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed conversion rate")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Calibration reliability", loc="left", fontweight="bold")
save(fig, "chart_calibration.png")

# 7. Learning curve
lc = m.get("learning_curve", [])
fig, ax = plt.subplots(figsize=(5.6, 4.0))
if lc:
    sizes = [p["size"] for p in lc]
    tr = [p["train"] for p in lc]; va = [p["validation"] for p in lc]
    std = [p.get("std",0) for p in lc]
    ax.plot(sizes, tr, color=VIO, lw=2.2, marker="o", ms=4, label="Training score")
    ax.plot(sizes, va, color=ACC, lw=2.2, marker="s", ms=4, label="Validation score")
    va = np.array(va); std = np.array(std)
    ax.fill_between(sizes, va-std, va+std, color=ACC, alpha=0.12)
style(ax); ax.set_xlabel("Training records used"); ax.set_ylabel("PR-AUC (avg precision)")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Learning curve — selected model", loc="left", fontweight="bold")
save(fig, "chart_learning.png")

# 8. SHAP importance (top 10)
sh = imp["shap"][:10][::-1]
def pretty(f):
    f = f.replace("numeric__","").replace("categorical__","").replace("_"," ")
    return f
labels = [pretty(x["feature"]) for x in sh]
vals = [x["importance"] for x in sh]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.barh(range(len(labels)), vals, color=VIO)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
style(ax); ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Global feature importance (SHAP)", loc="left", fontweight="bold")
save(fig, "chart_shap.png")

# 9. Score distribution histogram with tiers
hist = ss["histogram"]
tier_col = {"Very High": ACC, "High": VIO, "Medium": VIO2, "Low": GREY}
names = [h["name"] for h in hist]
counts = [h["count"] for h in hist]
cols = [tier_col.get(h["tier"], GREY) for h in hist]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.bar(range(len(names)), counts, color=cols)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
style(ax); ax.set_xlabel("Lead score band (0–100)"); ax.set_ylabel("Number of leads")
from matplotlib.patches import Patch
leg = [Patch(color=c, label=t) for t,c in tier_col.items()]
ax.legend(handles=leg, frameon=False, fontsize=8.5, ncol=4, loc="upper right")
ax.set_title("Distribution of lead scores by priority tier", loc="left", fontweight="bold")
save(fig, "chart_scores.png")

# 10. Tier pie/donut
tiers = ss["tiers"]
fig, ax = plt.subplots(figsize=(4.8, 4.0))
tnames = list(tiers.keys()); tvals = list(tiers.values())
tcols = [tier_col.get(t, GREY) for t in tnames]
wedges,_,_ = ax.pie(tvals, colors=tcols, autopct=lambda p: f"{p:.0f}%",
                    startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"),
                    pctdistance=0.78, textprops=dict(fontsize=9, color=INK))
ax.legend(wedges, [f"{t} ({v})" for t,v in zip(tnames,tvals)], frameon=False,
          fontsize=9, loc="center", bbox_to_anchor=(0.5,0.5))
ax.set_title(f"Priority tiers across {ss['total']} scored leads", loc="left", fontweight="bold")
save(fig, "chart_tiers.png")

# 11. Class imbalance
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.bar(["Not converted (no)","Converted (yes)"], [5422, 578], color=[GREY, VIO])
style(ax); ax.set_ylabel("Records")
for i,v in enumerate([5422,578]):
    ax.text(i, v+40, str(v), ha="center", fontsize=10, fontweight="bold")
ax.set_title("Target balance — 'Converted' (9.4 : 1 imbalance)", loc="left", fontweight="bold")
save(fig, "chart_imbalance.png")

print("ALL CHARTS DONE")
