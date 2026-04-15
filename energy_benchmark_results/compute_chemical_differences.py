#!/usr/bin/env python3
import os, json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

sns.set(style="whitegrid", context="talk")
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

ROOT = Path(".").resolve()
EXCLUDE = {"true", "__pycache__", ".git"}

# --------------------------------------------------------------------------------
# json helpers
# --------------------------------------------------------------------------------

def iter_json_files(d: Path):
    for root, _, files in os.walk(d):
        for fname in files:
            if fname.endswith(".json") and fname != "workflows.json":
                yield Path(root) / fname

def load_conformer_energies(mol_dir: Path):
    confs = []
    for jf in iter_json_files(mol_dir):
        try:
            data = json.loads(jf.read_text())
            ini = float(data.get("initial_energy", np.nan))
            fin = float(data.get("final_energy", np.nan))
            if not np.isnan(ini) and not np.isnan(fin):
                confs.append((ini, fin))
        except Exception:
            pass
    return confs

def per_molecule_stats(mol_dir: Path):
    confs = load_conformer_energies(mol_dir)
    if not confs:
        return None, None, 0, 0.0
    deltas = [fin - ini for ini, fin in confs]
    return np.mean(deltas), min(fin for _, fin in confs), len(confs), sum(deltas)

def get_molecule_dirs(method_dir: Path):
    return [p for p in method_dir.iterdir() if p.is_dir()]

def load_ground_truth(root: Path):
    true_dir = root / "true"
    gt = {}
    if not true_dir.exists():
        raise RuntimeError("'true' directory missing")
    for mol_dir in get_molecule_dirs(true_dir):
        _, min_final, n, _ = per_molecule_stats(mol_dir)
        if n > 0 and min_final is not None:
            gt[mol_dir.name] = min_final
    return gt

# --------------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------------

def compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: None for k in [
            "n_heavy","n_atoms","n_rot_bonds","n_rings","n_aromatic_rings",
            "max_ring_size","is_macrocycle","mol_wt","tpsa","hbd","hba",
            "formal_charge","frac_hetero","frac_c_sp3","frac_rot_bonds"
        ]}
    n_heavy = mol.GetNumHeavyAtoms()
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    rings = Chem.GetSymmSSSR(mol)
    max_ring = max((len(r) for r in rings), default=0)
    return {
        "n_heavy": n_heavy,
        "n_atoms": mol.GetNumAtoms(),
        "n_rot_bonds": n_rot,
        "n_rings": n_rings,
        "n_aromatic_rings": n_arom,
        "max_ring_size": max_ring,
        "is_macrocycle": bool(max_ring >= 12),
        "mol_wt": Descriptors.ExactMolWt(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "frac_hetero": sum(a.GetAtomicNum() not in (1,6) for a in mol.GetAtoms()) / n_heavy,
        "frac_c_sp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "frac_rot_bonds": n_rot / max(n_heavy-1,1),
    }

# --------------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------------

def main():
    gt_min = load_ground_truth(ROOT)
    method_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDE and p.name != "true"]
    method_dirs.sort(key=lambda p: p.name)

    energy_stats = defaultdict(dict)
    all_molecules = set()

    for method_dir in method_dirs:
        method_name = method_dir.name
        for mol_dir in get_molecule_dirs(method_dir):
            mol = mol_dir.name
            all_molecules.add(mol)
            avg_delta, gen_min, n_confs, sum_delta = per_molecule_stats(mol_dir)
            gap = gen_min - gt_min.get(mol, np.nan) if gen_min and mol in gt_min else np.nan
            energy_stats[method_name][mol] = {
                "n_confs": n_confs,
                "avg_delta": avg_delta,
                "gap_vs_gt": gap,
            }

    # dataframe
    methods = sorted(energy_stats.keys())
    rows = []
    for mol in sorted(all_molecules):
        row = {"molecule": mol}
        row.update(compute_descriptors(mol))
        for m in methods:
            s = energy_stats[m].get(mol, {"n_confs":0,"avg_delta":np.nan,"gap_vs_gt":np.nan})
            row[f"{m}_n_confs"] = s["n_confs"]
            row[f"{m}_avg_delta"] = s["avg_delta"]
            row[f"{m}_gap_vs_gt"] = s["gap_vs_gt"]
        rows.append(row)
    df = pd.DataFrame(rows)

    # best method
    gap_cols = [f"{m}_gap_vs_gt" for m in methods]
    df["best_method"] = df[gap_cols].idxmin(axis=1).str.replace("_gap_vs_gt","")
    df["best_gap"] = df.lookup(df.index, df["best_method"] + "_gap_vs_gt")

    out_csv = ROOT / "energy_benchmark_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved → {out_csv}")

    # --------------------------------------------------------------------------------
    # visualization (publication-ready)
    # --------------------------------------------------------------------------------

    # color + typography setup
    sns.set(style="whitegrid", context="talk", font_scale=1.1)
    plt.rcParams.update({
        "axes.edgecolor": "0.3",
        "axes.linewidth": 1.0,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "font.sans-serif": "DejaVu Sans",
    })

    # heatmap — energy gap per molecule
    plt.figure(figsize=(10,6))
    sns.heatmap(df[gap_cols], cmap="RdBu_r", center=0, linewidths=0.4,
                cbar_kws={"label": "Energy gap vs ground truth (Hartree)"})
    plt.title("Energy Gap vs Ground Truth per Molecule\n(red = higher energy, blue = lower / more stable)")
    plt.xlabel("Method")
    plt.ylabel("Molecules")
    plt.tight_layout()
    plt.savefig("fig1_gap_heatmap_pub.png", dpi=400)
    plt.close()

    # win fractions
    win_counts = df["best_method"].value_counts(normalize=True)
    plt.figure(figsize=(6,5))
    sns.barplot(x=win_counts.index, y=win_counts.values, palette="muted", edgecolor="k")
    plt.title("Win Fraction per Method", fontweight="bold")
    plt.ylabel("Fraction of Molecules")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig("fig2_win_fraction_pub.png", dpi=400)
    plt.close()

    # lyrebird vs etflow (rotatable bonds + heavy atoms)
    if all(x in df.columns for x in ["lyrebird_gap_vs_gt","etflow_gap_vs_gt"]):
        df["lyre_minus_et"] = df["lyrebird_gap_vs_gt"] - df["etflow_gap_vs_gt"]
        fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
        for ax, xcol, xlabel in zip(
            axes,
            ["n_rot_bonds","n_heavy"],
            ["# Rotatable Bonds","# Heavy Atoms"]
        ):
            sns.regplot(x=xcol, y="lyre_minus_et", data=df, lowess=True,
                        scatter_kws={"s":80,"alpha":0.7,"edgecolor":"k"},
                        line_kws={"color":"darkred","lw":2.0}, ax=ax)
            ax.axhline(0, color="gray", linestyle="--", lw=1)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{xlabel} vs (Lyrebird − ET-Flow Gap)\n(Negative → Lyrebird better)")
        axes[0].set_ylabel("ΔEnergy (Hartree)")
        plt.tight_layout()
        plt.savefig("fig3_lyre_vs_et_pub.png", dpi=400)
        plt.close()

        # bubble — size vs flexibility
        plt.figure(figsize=(8,6))
        plt.scatter(df["n_heavy"], df["lyre_minus_et"],
                    s=df["n_rot_bonds"]*6+30, alpha=0.7,
                    color="#4B9CD3", edgecolor="k", linewidth=0.7)
        plt.axhline(0, color="gray", linestyle="--")
        plt.xlabel("# Heavy Atoms")
        plt.ylabel("Lyrebird − ETFlow ΔEnergy (Hartree)")
        plt.title("Size vs Flexibility Influence on Lyrebird Advantage", fontweight="bold")
        plt.tight_layout()
        plt.savefig("fig4_bubble_pub.png", dpi=400)
        plt.close()

    # descriptor correlation
    if "lyre_minus_et" in df:
        desc_cols = ["n_heavy","n_rot_bonds","n_rings","tpsa","frac_c_sp3","frac_rot_bonds"]
        corr = df[desc_cols+["lyre_minus_et"]].corr()["lyre_minus_et"].drop("lyre_minus_et")
        plt.figure(figsize=(7,5))
        sns.barplot(x=corr.values, y=corr.index, palette="coolwarm_r", edgecolor="k")
        for i, (val, name) in enumerate(zip(corr.values, corr.index)):
            plt.text(val, i, f"ρ={val:+.2f}", va="center",
                    ha="left" if val>0 else "right",
                    fontsize=10, color="black",
                    fontweight="bold", clip_on=False)
        plt.axvline(0, color="gray", linestyle="--", lw=1.2)
        plt.title("Descriptor Correlation with Lyrebird − ETFlow Gap\n(negative → Lyrebird improves)", fontweight="bold")
        plt.xlabel("Pearson r")
        plt.tight_layout()
        plt.savefig("fig5_corr_pub.png", dpi=400)
        plt.close()

    print("✅ polished publication figures saved: *_pub.png")

if __name__ == "__main__":
    main()
