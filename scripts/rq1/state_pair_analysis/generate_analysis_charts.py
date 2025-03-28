import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_bars_on_ax(ax, df, title):
    categories = [ "Clones", "ND2", "ND3", "Total ND", "Distinct"]
    col_map = {
        "ND2": "pred_nd2",
        "ND3": "pred_nd3",
        "Clones": "pred_clones",
        "Total ND": "pred_nd_total",
        "Distinct": "pred_distinct",
    }
    methods = df["Method"].unique().tolist()
    x = np.arange(len(categories))
    bar_width = 0.8 / len(methods)

    # Use the previous color mapping
    color_map = {
        "RTED":      "#7B4CF0",
        "PDiff":     "#34A853",
        "Webembed":  "#FBBC05",
        "FragGen":   "#26CDE8",
        "SNN (T1)":  "#4285F4",
        "SNN (T4)":  "#26CDE8",
        "SNN (T6)":  "#4285F4",
        "Total":     "#EA4335"
    }

    for i, method in enumerate(methods):
        df_method = df[df["Method"] == method]
        bar_values = [df_method[col_map[cat]].values[0] for cat in categories]
        method_color = color_map.get(method, "gray")
        rects = ax.bar(
            x + i * (bar_width + 0.01),
            bar_values,
            bar_width,
            label=method,
            color=method_color
        )
        ax.bar_label(rects, fmt='%.0f', rotation=90, padding=3)

    ax.set_xticks(x + (len(methods) - 1) * (bar_width + 0.005) / 2)
    ax.set_xticklabels(categories, rotation=0, ha="right")
    ax.set_ylabel("# state-pairs")
    ax.legend()
    ax.spines["top"].set_visible(False)

if __name__ == "__main__":
    base_path = "/Users/kasun/Documents/uni/semester-4/thesis/NDD"
    data = [
        ("withinapp", "RTED", 3101, 3605, 1176, 7882, 65429),
        ("withinapp", "PDiff", 3146, 3573, 732, 7451, 66119),
        ("withinapp", "Webembed", 2734, 2311, 2027, 7072, 65625),
        ("withinapp", "FragGen", 2804, 3306, 1428, 7538, 66453),
        ("withinapp", "SNN (T1)", 3175, 4625, 2881, 10681, 67960),
        ("withinapp", "Total", 3177, 4631, 2886, 10694, 67971),
        ("acrossapp", "RTED", 3126, 7471, 1169, 11766, 69034),
        ("acrossapp", "PDiff", 3170, 5935, 740, 9845, 73065),
        ("acrossapp", "Webembed", 2761, 6014, 1953, 10728, 73158),
        ("acrossapp", "SNN (T4)", 3174, 6187, 758, 10119, 69421),
        ("acrossapp", "SNN (T6)", 3190, 6948, 1591, 11729, 67754),
        ("acrossapp", "Total", 3204, 8675, 2886, 14765, 75225),
    ]
    df = pd.DataFrame(data, columns=[
        "Setting", "Method", "pred_clones", "pred_nd2", "pred_nd3", "pred_nd_total", "pred_distinct"
    ])

    df_within = df[df["Setting"].str.strip() == "withinapp"]
    df_across = df[df["Setting"].str.strip() == "acrossapp"]

    # Create the Within App plot as a separate figure
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    plot_grouped_bars_on_ax(ax1, df_within, "Within App")
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/report/state_pair_analysis_within.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create the Across App plot as a separate figure
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    plot_grouped_bars_on_ax(ax2, df_across, "Across App")
    plt.tight_layout()
    plt.savefig(f"{base_path}/results/report/state_pair_analysis_across.png", dpi=300, bbox_inches='tight')
    plt.show()
