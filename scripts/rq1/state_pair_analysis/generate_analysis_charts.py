import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl   # ➊ new import to tweak global fonts

# ➋ ── global font settings ─────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "Gill Sans",   # make sure Gill Sans is installed on your system
    "font.size": 16,              # base size for everything
    "axes.titlesize": 14,         # axes title
    "axes.labelsize": 13,         # axis labels
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 14,
})

# ── data ───────────────────────────────────────────────────────────────
data = [
    ("withinapp", "RTED", 3101, 3605, 1176, 7882),
    ("withinapp", "PDiff", 3146, 3573, 732, 7451),
    ("withinapp", "WebE.", 2734, 2311, 2027, 7072),
    ("withinapp", "FragG.", 2804, 3306, 1428, 7538),
    ("withinapp", "SNN (T1)", 3161, 4559, 2874, 10594),
    ("withinapp", "Total", 3177, 4631, 2886, 10694),
    ("acrossapp", "RTED", 3126, 7471, 1169, 11766),
    ("acrossapp", "PDiff", 3170, 5935, 740, 9845),
    ("acrossapp", "WebE.", 2761, 6014, 1953, 10728),
    ("acrossapp", "SNN (T8)", 3190, 6948, 1591, 11729),
    ("acrossapp", "Total", 3204, 8675, 2886, 14765),
]
df = pd.DataFrame(
    data, columns=["Setting", "Method", "pred_clones", "pred_nd2", "pred_nd3", "pred_nd_total"]
)

base_path = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"

settings = ["withinapp", "acrossapp"]
methods_map = {
    "withinapp": ["RTED", "PDiff", "WebE.", "FragG.", "SNN (T1)", "Total"],
    "acrossapp": ["RTED", "PDiff", "WebE.", "SNN (T8)", "Total"],
}
metrics = ["pred_clones", "pred_nd2", "pred_nd3"]
metric_labels = {"pred_clones": "Clones", "pred_nd2": "ND2", "pred_nd3": "ND3"}
colors = {"pred_clones": "#34A853", "pred_nd2": "#FBBC05", "pred_nd3": "#4285F4"}

bar_w = 0.14
gap = 0.02
group_x = np.array([0, 1])

fig, ax = plt.subplots(figsize=(8, 6))  # slightly wider

for gi, setting in enumerate(settings):
    methods = methods_map[setting]
    n = len(methods)
    total_width = n * bar_w + (n - 1) * gap
    offsets = np.linspace(-total_width / 2 + bar_w / 2, total_width / 2 - bar_w / 2, n)

    subdf = df[df["Setting"] == setting].set_index("Method").loc[methods]
    bottoms = np.zeros(n)

    for metric in metrics:
        vals = subdf[metric].values
        rects = ax.bar(
            group_x[gi] + offsets,
            vals,
            bar_w,
            bottom=bottoms,
            color=colors[metric],
            label=metric_labels[metric] if gi == 0 else None,
        )
        ax.bar_label(
            rects,
            labels=[f"{v:,}" for v in vals],
            label_type="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
        bottoms += vals

    # method label above each stacked bar
    for i, m in enumerate(methods):
        ax.text(
            group_x[gi] + offsets[i],
            bottoms[i] + 250,             # space above bar
            m,
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=45,
        )

ax.set_xticks(group_x)
ax.set_xticklabels(["Within App", "Across App"])
ax.set_ylabel("# state-pairs")
ax.legend(title="Near Duplicate Types", frameon=False)
ax.margins(y=0.1)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{base_path}/results/report/state_pair_analysis.png", dpi=300, bbox_inches="tight")
