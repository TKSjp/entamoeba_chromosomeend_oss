"""Step04: end-type clustering and MAFFT alignments for every scale.

* Hierarchical clustering (1 - similarity) per scale; the scale `cluster.similarity_scale_kb`
  defines the primary end types (T1, T2, ... by size; singletons = U).
* MAFFT per scale: all chromosomes + each primary type with >= per_type_min_members members.
  Sequences keep the chromosome orientation (3' end = right edge of the alignment).
Outputs (result/04_cluster/):
  clusters_{W}.tsv / linkage_{W}.json     per-scale clusters, leaf order, linkage matrix
  end_types.tsv                           primary end type per chromosome
  msa/msa_{W}_{group}.fasta               alignments (group = all / T1 / T2 ...)
  msa/msa_{W}_{group}.colstats.tsv        per-column consensus / conservation / gap fraction
  consensus_{W}.fna                       gap-stripped majority consensus of every group
  msa_index.tsv                           list of alignments with size / runtime
An existing alignment is reused (msa_*.cache.json) when the MAFFT options are unchanged and its
gap-stripped sequences equal the current input, so re-runs only realign what changed.
"""
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from common import Logger, load_config, path, read_fasta, step_dir, tool_version, window_label, write_fasta

cfg = load_config()
cc, mc = cfg["cluster"], cfg["mafft"]
log = Logger(cfg, "STEP04_cluster_msa")
log.start("clustering and MAFFT", ["cluster", "mafft", "windows_kb"], cfg)
log(f"tool: mafft {tool_version('mafft --version')}")
out = step_dir(cfg, "04_cluster")
msadir = os.path.join(out, "msa")
os.makedirs(msadir, exist_ok=True)
sim_dir = path(cfg, cfg["result_dir"], "03_similarity")
qc = path(cfg, cfg["result_dir"], "01_qc")


def cluster(lab):
    M = pd.read_csv(f"{sim_dir}/similarity_{lab}.tsv", sep="\t", index_col=0)
    names = list(M.index)
    D = np.clip(1 - M.to_numpy(), 0, 1)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    cd = squareform(D, checks=False)
    Z = optimal_leaf_ordering(linkage(cd, method=cc["linkage"]), cd)
    lab_ = fcluster(Z, t=cc["distance_cut"], criterion="distance")
    order = [names[i] for i in leaves_list(Z)]
    return names, Z, lab_, order


def name_types(names, labels, order):
    """T1.. by size (ties: first appearance in leaf order); singletons -> U."""
    cnt = Counter(labels)
    first = {labels[names.index(n)]: i for i, n in reversed(list(enumerate(order)))}
    multi = sorted([c for c in cnt if cnt[c] > 1], key=lambda c: (-cnt[c], first[c]))
    m = {c: f"T{i + 1}" for i, c in enumerate(multi)}
    return {n: m.get(labels[i], "U") for i, n in enumerate(names)}


primary = window_label(cc["similarity_scale_kb"])
per_scale = {}
for kb in cfg["windows_kb"]:
    lab = window_label(kb)
    names, Z, labels, order = cluster(lab)
    types = name_types(names, list(labels), order)
    per_scale[lab] = types
    pd.DataFrame({"chrom": names, "cluster_id": labels, "type": [types[n] for n in names],
                  "leaf_rank": [order.index(n) for n in names]}).to_csv(f"{out}/clusters_{lab}.tsv", sep="\t", index=False)
    with open(f"{out}/linkage_{lab}.json", "w") as fh:
        json.dump({"names": names, "order": order, "linkage": Z.tolist()}, fh)
    c = Counter(types.values())
    log(f"{lab}: " + ", ".join(f"{t}={c[t]}" for t in sorted(c, key=lambda x: (x == 'U', len(x), x))))

ptypes = per_scale[primary]
with open(f"{out}/end_types.tsv", "w") as fh:
    fh.write("chrom\tend_type\t" + "\t".join(f"type_{window_label(k)}" for k in cfg["windows_kb"]) + "\n")
    for n in ptypes:
        fh.write(f"{n}\t{ptypes[n]}\t" + "\t".join(per_scale[window_label(k)][n] for k in cfg["windows_kb"]) + "\n")
groups = {"all": list(ptypes)}
for t in sorted(set(ptypes.values()) - {"U"}, key=lambda x: int(x[1:])):
    mem = [n for n in ptypes if ptypes[n] == t]
    if len(mem) >= mc["per_type_min_members"]:
        groups[t] = mem
log(f"primary end types from {primary}: " + "; ".join(f"{g}: {','.join(m)}" for g, m in groups.items() if g != "all"))


def colstats(aln):
    names = list(aln)
    A = np.array([list(aln[n].upper()) for n in names])
    rows, cons = [], []
    for j in range(A.shape[1]):
        col = A[:, j]
        nong = col[col != "-"]
        if len(nong):
            b, c = Counter(nong).most_common(1)[0]
            cons_frac = c / len(col)
        else:
            b, cons_frac = "-", 0.0
        gapf = 1 - len(nong) / len(col)
        cons.append(b if gapf < 0.5 else "-")
        rows.append((j + 1, b, round(cons_frac, 4), round(gapf, 4)))
    return rows, "".join(cons).replace("-", "")


def cached_ok(outp, cache, mem, seqs):
    """reuse an alignment when MAFFT options match and its gap-stripped sequences equal the input."""
    if not (os.path.exists(outp) and os.path.getsize(outp) and os.path.exists(cache)):
        return False
    if json.load(open(cache)).get("options") != mc["options"]:
        return False
    aln = read_fasta(outp)
    return list(aln) == mem and all(aln[n].replace("-", "") == seqs[n] for n in mem)


idx_rows = []
for kb in cfg["windows_kb"]:
    lab = window_label(kb)
    seqs = read_fasta(f"{qc}/ends_3p_{lab}.fna")
    cons_recs = []
    for g, mem in groups.items():
        inp = f"{msadir}/msa_{lab}_{g}.input.fna"
        outp = f"{msadir}/msa_{lab}_{g}.fasta"
        cache = f"{msadir}/msa_{lab}_{g}.cache.json"
        if cached_ok(outp, cache, mem, seqs):
            dt = json.load(open(cache))["runtime_s"]
            log(f"MAFFT {lab} {g}: reuse existing alignment (same input sequences and options)")
        else:
            write_fasta(inp, [(n, seqs[n]) for n in mem])
            t0 = time.time()
            log.run(f"mafft {mc['options']} --thread {cfg['threads']} {inp} > {outp} 2> {msadir}/msa_{lab}_{g}.mafft.log")
            dt = time.time() - t0
            os.remove(inp)
            json.dump({"options": mc["options"], "runtime_s": round(dt, 1)}, open(cache, "w"))
        aln = read_fasta(outp)
        rows, cons = colstats(aln)
        with open(f"{msadir}/msa_{lab}_{g}.colstats.tsv", "w") as fh:
            fh.write("column\tmajority_base\tmajority_frac\tgap_frac\n")
            for r in rows:
                fh.write("\t".join(map(str, r)) + "\n")
        cons_recs.append((f"{lab}_{g}_consensus n={len(mem)}", cons))
        alen = len(next(iter(aln.values())))
        mean_cons = np.mean([r[2] for r in rows])
        idx_rows.append((lab, g, len(mem), alen, round(mean_cons, 4), round(dt, 1), os.path.relpath(outp, out)))
        log(f"MAFFT {lab} {g}: n={len(mem)}, aln length {alen:,}, mean majority frac {mean_cons:.3f}, {dt:.1f}s")
    write_fasta(f"{out}/consensus_{lab}.fna", cons_recs)

pd.DataFrame(idx_rows, columns=["window", "group", "n_seq", "aln_length", "mean_majority_frac", "runtime_s", "file"]) \
    .to_csv(f"{out}/msa_index.tsv", sep="\t", index=False)
log.end()
