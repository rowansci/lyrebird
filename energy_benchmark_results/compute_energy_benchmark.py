#!/usr/bin/env python3
import os
import json
import statistics
from pathlib import Path

ROOT = Path(".").resolve()
EXCLUDE = {"true", "__pycache__", ".git"}

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
            ini = float(data.get("initial_energy", None))
            fin = float(data.get("final_energy", None))
            if ini is not None and fin is not None:
                confs.append((ini, fin))
        except Exception:
            continue
    return confs

def per_molecule_stats(mol_dir: Path):
    confs = load_conformer_energies(mol_dir)
    if not confs:
        return None, None, 0, 0.0, None
    deltas = [fin - ini for (ini, fin) in confs]
    avg_delta = sum(deltas) / len(deltas)
    min_final = min(fin for (_, fin) in confs)
    sum_delta = sum(deltas)
    # energy diversity proxy: std dev of final energies
    finals = [fin for (_, fin) in confs]
    diversity = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    return avg_delta, min_final, len(confs), sum_delta, diversity

def get_molecule_dirs(method_dir: Path):
    return [p for p in method_dir.iterdir() if p.is_dir()]

def load_ground_truth(root: Path):
    gt = {}
    true_dir = root / "true"
    for mol_dir in get_molecule_dirs(true_dir):
        _, min_final, n, _, _ = per_molecule_stats(mol_dir)
        if n > 0 and min_final is not None:
            gt[mol_dir.name] = min_final
    return gt

def main():
    gt_min = load_ground_truth(ROOT)
    method_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDE]
    method_dirs.sort(key=lambda p: p.name)

    print("=" * 80)
    print("energy benchmark results (with diversity)")
    print("=" * 80)

    for method_dir in method_dirs:
        method_name = method_dir.name
        mol_dirs = get_molecule_dirs(method_dir)

        mol_avgs, mol_gaps, mol_avg_wrt_gt, mol_divs = [], [], [], []
        total_delta, total_confs, total_div = 0.0, 0, 0.0

        for mol_dir in mol_dirs:
            mol_name = mol_dir.name
            avg_delta, gen_min, n_confs, sum_delta, diversity = per_molecule_stats(mol_dir)
            confs = load_conformer_energies(mol_dir)

            total_delta += sum_delta
            total_confs += n_confs
            total_div += diversity * n_confs  # weighted

            gap = avg_vs_gt = None
            if mol_name in gt_min and gen_min is not None:
                gap = gen_min - gt_min[mol_name]
                avg_vs_gt = sum((fin - gt_min[mol_name]) for (_, fin) in confs) / len(confs)

            if avg_delta is not None:
                mol_avgs.append(avg_delta)
            if gap is not None:
                mol_gaps.append(gap)
            if avg_vs_gt is not None:
                mol_avg_wrt_gt.append(avg_vs_gt)
            if diversity is not None:
                mol_divs.append(diversity)

        unweighted_avg_delta = sum(mol_avgs) / len(mol_avgs) if mol_avgs else None
        weighted_avg_delta = total_delta / total_confs if total_confs > 0 else None
        avg_gap = sum(mol_gaps) / len(mol_gaps) if mol_gaps else None
        avg_avg_vs_gt = sum(mol_avg_wrt_gt) / len(mol_avg_wrt_gt) if mol_avg_wrt_gt else None
        avg_diversity_unweighted = sum(mol_divs) / len(mol_divs) if mol_divs else None
        avg_diversity_weighted = total_div / total_confs if total_confs > 0 else None

        print(f"\nmethod: {method_name}")
        print(f"  avg (final - initial) over ensembles (unweighted): {unweighted_avg_delta:.6f}" if unweighted_avg_delta else "  avg (final - initial): N/A")
        print(f"  avg (final - initial) over all conformers (weighted): {weighted_avg_delta:.6f}" if weighted_avg_delta else "  avg (final - initial): N/A")
        print(f"  avg (min_gen_final - min_gt_final): {avg_gap:.6f}" if avg_gap else "  avg (min_gen_final - min_gt_final): N/A")
        print(f"  avg (all_gen_final - min_gt_final): {avg_avg_vs_gt:.6f}" if avg_avg_vs_gt else "  avg (all_gen_final - min_gt_final): N/A")
        print(f"  energy diversity (stddev final E) unweighted: {avg_diversity_unweighted:.6f}" if avg_diversity_unweighted else "  energy diversity: N/A")
        print(f"  energy diversity (stddev final E) weighted: {avg_diversity_weighted:.6f}" if avg_diversity_weighted else "  energy diversity: N/A")

if __name__ == "__main__":
    main()