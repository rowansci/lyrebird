#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use("seaborn-whitegrid")


# load data
df = pd.read_csv("energy_benchmark_molecule_summary.csv")

methods = ["etflow", "etkdgv3", "lyrebird", "torsional_diffusion"]

# melt for plotting
df_melt = df.melt(
    id_vars=["molecule"],
    value_vars=[f"{m}_gap_vs_gt" for m in methods],
    var_name="method",
    value_name="gap_vs_gt",
)
df_melt["method"] = df_melt["method"].str.replace("_gap_vs_gt", "", regex=False)

# ------------------------------------------------------------------
# 1. violin plot of energy gap distributions
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.violinplot(
    data=df_melt,
    x="method",
    y="gap_vs_gt",
    inner="box",
    palette="Set2",
    cut=0
)
plt.axhline(0, color="gray", ls="--", lw=1)
plt.ylabel("energy gap vs ground truth (hartree)")
plt.title("distribution of energy gaps per method")
plt.tight_layout()
plt.savefig("fig_violin_gap.png", dpi=300)

# ------------------------------------------------------------------
# 2. win fraction bar chart
# ------------------------------------------------------------------
win_counts = df["best_method"].value_counts(normalize=True).reindex(methods).fillna(0)
plt.figure(figsize=(6, 4))
sns.barplot(x=win_counts.index, y=win_counts.values, palette="Set2")
plt.ylabel("fraction of molecules where method wins")
plt.title("win fraction by method")
plt.tight_layout()
plt.savefig("fig_win_fraction.png", dpi=300)

# ------------------------------------------------------------------
# 3. scatterplots vs descriptors
# ------------------------------------------------------------------
desc_cols = ["frac_c_sp3", "n_rot_bonds"]
for desc in desc_cols:
    plt.figure(figsize=(8, 4))
    for m in methods:
        plt.scatter(
            df[desc],
            df[f"{m}_gap_vs_gt"],
            s=30,
            alpha=0.7,
            label=m,
        )
    plt.xlabel(desc.replace("_", " "))
    plt.ylabel("energy gap (hartree)")
    plt.title(f"{desc} vs energy gap")
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fig_corr_{desc}.png", dpi=300)

# ------------------------------------------------------------------
# 4. descriptor–performance correlation heatmap
# ------------------------------------------------------------------
desc_cols = [
    "n_heavy",
    "n_rot_bonds",
    "n_aromatic_rings",
    "mol_wt",
    "frac_c_sp3",
    "frac_rot_bonds",
]
corr = []
for desc in desc_cols:
    row = {}
    for m in methods:
        vals = df[[desc, f"{m}_gap_vs_gt"]].dropna()
        if len(vals) > 1:
            row[m] = vals[desc].corr(vals[f"{m}_gap_vs_gt"])
        else:
            row[m] = np.nan
    corr.append(row)
corr_df = pd.DataFrame(corr, index=desc_cols)

plt.figure(figsize=(7, 5))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("correlation between molecular descriptors and performance")
plt.tight_layout()
plt.savefig("fig_corr_heatmap.png", dpi=300)

    # -------------------------------------------------------------------------
    # ring-size / macrocycle analysis
    # -------------------------------------------------------------------------
sns.set(style="ticks", font_scale=1.1)

df_plot = df.dropna(subset=["lyrebird_gap_vs_gt", "max_ring_size", "mol_wt"]).copy()
df_plot["is_macrocycle"] = df_plot["is_macrocycle"].astype(bool)

    # scatter: max ring size vs energy gap (Lyrebird)
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(
        data=df_plot,
        x="max_ring_size",
        y="lyrebird_gap_vs_gt",
        hue="is_macrocycle",
        palette={True: "#e74c3c", False: "#3498db"},
        s=70, ax=ax
    )
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("Max ring size")
ax.set_ylabel("Energy gap vs ground truth (Hartree)")
ax.set_title("Lyrebird performance vs ring size")
ax.legend(title="Macrocycle")
sns.despine()
plt.tight_layout()
plt.savefig("fig_ring_vs_gap.png", dpi=300)

    # scatter: molecular weight vs gap, colored by ring size
fig, ax = plt.subplots(figsize=(6,4))
sc = ax.scatter(
        df_plot["mol_wt"],
        df_plot["lyrebird_gap_vs_gt"],
        c=df_plot["max_ring_size"],
        cmap="viridis", s=70
    )
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("Molecular weight (Da)")
ax.set_ylabel("Energy gap vs ground truth (Hartree)")
ax.set_title("Lyrebird energy gap vs molecular weight")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Max ring size")
sns.despine()
plt.tight_layout()
plt.savefig("fig_molwt_vs_gap.png", dpi=300)
plt.show()


print("plots saved: fig_violin_gap.png, fig_win_fraction.png, fig_corr_frac_c_sp3.png, fig_corr_n_rot_bonds.png, fig_corr_heatmap.png")
