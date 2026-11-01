"""Step06: define the shared terminal element (the conserved terminal core) and map it on every chromosome end and in the genome.

1. Seed: 3'-most `seed_bp` of the majority consensus of the `seed_window_kb` MAFFT alignment of the largest end type
   (Step04). Ends whose 3'-terminal window has a + strand hit containing the seed's 3' edge and covering
   >= element_min_qcov of it are the seed carriers.
2. Conserved terminal core: the `core_probe_kb` terminal windows of the seed carriers, trimmed at the seed's 3' edge
   (overhang removed), are aligned with MAFFT. Variable sites are defined with the Step07 criteria (support <
   micro.core_support, insertion columns present in < micro.ignore_insertion_below ignored). Along the consensus,
   from the 3' edge, the single change point that maximises the two-segment Bernoulli log-likelihood of variable-site
   occurrence (inner segment less variable, each segment >= core_min_seg_bp) is the core boundary.
   Terminal element = the 3'-most core-length bases of that alignment's consensus. Inverted repeats (hairpin
   candidates) inside the element are listed.
3. BLASTN element vs each 3'-terminal window (5 kb): position of the element, identity, and the
   "overhang" (bases between the element's 3' edge and the chromosome end).
4. BLASTN element vs whole genome: full-length copies and diverged homologs.
Outputs (result/06_terminal_element/):
  seed_element.fna / core_probe_msa.fasta / core_profile.tsv   seed, probe alignment, per-position variability
  terminal_element.fna               element (= conserved terminal core)
  element_inverted_repeats.tsv       exact inverted repeats (stem >= min_stem) inside the element
  element_on_ends.tsv                per chromosome: best hit in 3' window, overhang length / sequence
  element_hits_ends.tsv              all hits in the terminal windows (dist coordinates; for the viewer)
  element_genome_hits.tsv            all genome hits with class (3p_terminal / 5p_terminal / interior)
"""
import os
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

from common import Logger, genome_fasta, load_config, path, read_fasta, revcomp, step_dir, tool_version, window_label, write_fasta

cfg = load_config()
ec, mc = cfg["terminal_element"], cfg["micro"]
log = Logger(cfg, "STEP06_terminal_element")
log.start("terminal element (conserved terminal core) and mapping", ["terminal_element"], cfg)
log(f"tool: mafft {tool_version('mafft --version')}")
out = step_dir(cfg, "06_terminal_element")
R = path(cfg, cfg["result_dir"])
genome = genome_fasta(cfg)
seqs = read_fasta(genome)
types = pd.read_csv(f"{R}/04_cluster/end_types.tsv", sep="\t")

COLS = "qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen"
bopt = f"-task blastn -evalue {ec['evalue']} -dust no -outfmt '6 {COLS}' -max_target_seqs 5000 -max_hsps 1000"


def blast(query_fa, subject_fa):
    cmd = f"blastn -query {query_fa} -subject {subject_fa} {bopt}"
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    rows = [ln.split("\t") for ln in r.stdout.splitlines()]
    df = pd.DataFrame(rows, columns=COLS.split())
    for c in COLS.split()[2:]:
        df[c] = pd.to_numeric(df[c])
    return df[df.length >= ec["min_hit_len"]].copy()


win_kb = ec["search_window_kb"]
wfa = f"{R}/01_qc/ends_3p_{window_label(win_kb)}.fna"
ends = read_fasta(wfa)


def map_on_ends(query_fa):
    """hits of the query in every 3'-terminal window and the per-end call (has_element, overhang)."""
    h = blast(query_fa, wfa)
    h["strand"] = (h.send >= h.sstart).map({True: "+", False: "-"})
    lo, hi = h[["sstart", "send"]].min(axis=1), h[["sstart", "send"]].max(axis=1)
    h["dist_from"] = h.slen - hi + 1
    h["dist_to"] = h.slen - lo + 1
    h["qcov"] = h.length / h.qlen
    rows = []
    for c, s in ends.items():
        g = h[(h.sseqid == c) & (h.strand == "+")]
        # hit containing the element's 3' edge (qend near qlen) and covering >= element_min_qcov, closest to the chromosome end
        edge = g[(g.qend >= g.qlen - ec["edge_tolerance"]) & (g.qcov >= ec.get("element_min_qcov", 0))].sort_values("dist_from")
        best = edge.iloc[0] if len(edge) else (g.sort_values("bitscore", ascending=False).iloc[0] if len(g) else None)
        if best is None:
            rows.append((c, 0, "", "", "", "", "", "", "", len(h[h.sseqid == c])))
            continue
        # overhang = bases beyond the element's 3' edge (corrected for unaligned element tail)
        overhang = int(best.dist_from - 1 - (best.qlen - best.qend))
        oh_seq = s[len(s) - overhang:] if overhang > 0 else ""
        # has_element: an edge hit as above; otherwise the best partial hit is reported with 0
        rows.append((c, int(len(edge) > 0), int(best.dist_from), int(best.dist_to), round(best.pident, 1), round(best.qcov, 3),
                     int(best.qend == best.qlen or best.qend >= best.qlen - ec["edge_tolerance"]), overhang,
                     oh_seq[:ec["overhang_show_bp"]] + ("..." if len(oh_seq) > ec["overhang_show_bp"] else ""),
                     len(h[h.sseqid == c])))
    on = pd.DataFrame(rows, columns=["chrom", "has_element", "dist_from", "dist_to", "pident", "qcov",
                                     "has_3p_edge", "overhang_bp", "overhang_seq", "n_hits_in_window"])
    return h, on.merge(types[["chrom", "end_type"]], on="chrom")


# --- 1. seed and its carriers
major = types[types.end_type != "U"].end_type.value_counts().idxmax()
slab = window_label(ec["seed_window_kb"])
cons_all = read_fasta(f"{R}/04_cluster/consensus_{slab}.fna")
key = next(k for k in cons_all if k.startswith(f"{slab}_{major}_"))
seed = cons_all[key][-ec["seed_bp"]:].upper()
write_fasta(f"{out}/seed_element.fna", [(f"seed source={key} len={len(seed)}", seed)])
_, son = map_on_ends(f"{out}/seed_element.fna")
carriers = son[son.has_element == 1]
log(f"seed: 3'-most {len(seed)} bp of '{key}' (largest type {major}); carriers: {len(carriers)} ({', '.join(carriers.chrom)})")


# --- 2. conserved terminal core by change point
def changepoint(v, min_seg):
    """index b maximising the two-segment Bernoulli log-likelihood of v (v[0] = element 3' edge), with v[:b] less variable."""
    n = len(v)
    c = np.concatenate([[0], np.cumsum(v)])

    def ll(k, m):
        p = k / m
        return (k * np.log(p) if k else 0.0) + ((m - k) * np.log(1 - p) if m - k else 0.0)

    best, bi = -np.inf, None
    for b in range(min_seg, n - min_seg + 1):
        k1, k2 = c[b], c[n] - c[b]
        if k1 / b >= k2 / (n - b):
            continue
        s = ll(k1, b) + ll(k2, n - b)
        if s > best:
            best, bi = s, b
    return bi


pends = read_fasta(f"{R}/01_qc/ends_3p_{window_label(ec['core_probe_kb'])}.fna")
recs = []
for r in carriers.itertuples():
    s = pends[r.chrom]
    oh = max(0, int(r.overhang_bp))
    recs.append((r.chrom, s[:len(s) - oh] if oh else s))
element = seed
if len(recs) >= 2:
    write_fasta(f"{out}/core_probe_input.fna", recs)
    log.run(f"mafft {ec['core_mafft_options']} --thread {cfg['threads']} {out}/core_probe_input.fna > {out}/core_probe_msa.fasta "
            f"2> {out}/core_probe_mafft.log")
    os.remove(f"{out}/core_probe_input.fna")
    aln = read_fasta(f"{out}/core_probe_msa.fasta")
    A = np.array([list(aln[n].upper()) for n in aln])
    N = A.shape[0]
    pres = (A != "-").mean(0)
    maj = [Counter(col[col != "-"]).most_common(1)[0] if (col != "-").any() else ("-", 0) for col in A.T]
    sup = np.array([m[1] / N for m in maj])
    is_cons = pres >= 0.5
    var = (sup < mc["core_support"]) & (pres >= mc["ignore_insertion_below"])
    v = var[is_cons][::-1].astype(int)                     # consensus positions, index 0 = element 3' edge
    cons = "".join(m[0] for m, k in zip(maj, is_cons) if k)
    C = changepoint(v, ec["core_min_seg_bp"])
    if C is None:
        log("WARNING: no change point found; the seed is used as the element")
    else:
        element = cons[-C:]
        log(f"conserved terminal core: {C} bp from the element 3' edge (variable sites {v[:C].mean():.1%} inside, "
            f"{v[C:].mean():.1%} outside; probe {len(v):,} consensus bp, {N} ends)")
    pd.DataFrame({"dist": np.arange(1, len(v) + 1), "variable": v, "support": sup[is_cons][::-1].round(3),
                  "presence": pres[is_cons][::-1].round(3),
                  "region": ["core" if C and d <= C else "outer" for d in range(1, len(v) + 1)]}) \
        .to_csv(f"{out}/core_profile.tsv", sep="\t", index=False)
else:
    log("WARNING: fewer than 2 seed carriers; the seed is used as the element")
write_fasta(f"{out}/terminal_element.fna", [(f"terminal_element conserved_core len={len(element)} seed={key}", element)])
log(f"element 3' end: ...{element[-60:]}")

# inverted repeats (exact stems) in the element; coordinates as distance from the element 3' end
irs = []
L = len(element)
for k in range(ec["min_stem"], 16):
    for i in range(L - k + 1):
        stem = element[i:i + k]
        rc = revcomp(stem)
        for j in range(i + k, min(L - k + 1, i + k + ec["max_loop"] + 1)):
            if element[j:j + k] == rc:
                irs.append((k, j - i - k, L - (j + k) + 1, L - i, stem, element[i:j + k]))
irs_df = pd.DataFrame(irs, columns=["stem_len", "loop_len", "dist_from", "dist_to", "stem", "hairpin_seq"])
# keep maximal stems only (drop those contained in a longer one)
keep = []
for r in irs_df.sort_values("stem_len", ascending=False).itertuples():
    if not any(r.dist_from >= k.dist_from and r.dist_to <= k.dist_to and r.loop_len >= k.loop_len for k in keep):
        keep.append(r)
irs_df = pd.DataFrame(keep).drop(columns="Index") if keep else irs_df
irs_df.to_csv(f"{out}/element_inverted_repeats.tsv", sep="\t", index=False)
for r in irs_df.head(5).itertuples():
    log(f"inverted repeat in element: stem {r.stem_len} bp, loop {r.loop_len} bp, dist {r.dist_from}-{r.dist_to}: {r.hairpin_seq}")

# --- 3. element on the 3'-terminal windows
q = f"{out}/terminal_element.fna"
h, on = map_on_ends(q)
h.to_csv(f"{out}/element_hits_ends.tsv", sep="\t", index=False)
on.to_csv(f"{out}/element_on_ends.tsv", sep="\t", index=False)
cap = on[(on.has_element == 1) & (on.has_3p_edge == 1)]
log(f"element (+ strand, 3' edge) found in {len(cap)}/{len(on)} terminal {win_kb} kb windows")
log(f"overhang <= {ec['capped_max_overhang']} bp (element at the very end): "
    f"{(cap.overhang_bp <= ec['capped_max_overhang']).sum()} chromosomes; overhang values: "
    + ", ".join(f"{r.chrom}:{r.overhang_bp}" for r in cap.itertuples()))
log("no element: " + ", ".join(on[on.has_element == 0].chrom))
part = on[(on.has_element == 0) & (on.dist_from.astype(str) != "")]
if len(part):
    log(f"of these, partial hits only (no element 3' edge): {len(part)} ({', '.join(part.chrom)})")
if set(cap.chrom) != set(carriers.chrom):
    log(f"note: element carriers differ from seed carriers: + {sorted(set(cap.chrom) - set(carriers.chrom))}, "
        f"- {sorted(set(carriers.chrom) - set(cap.chrom))}")

# --- 4. element in the whole genome
g = blast(q, genome)
lo, hi = g[["sstart", "send"]].min(axis=1), g[["sstart", "send"]].max(axis=1)
ex = cfg["blast"]["internal_exclude_kb"] * 1000
g["subject_class"] = ["3p_terminal" if sl - b + 1 <= ex else ("5p_terminal" if b <= ex else "interior")
                      for sl, b in zip(g.slen, hi)]
g["strand"] = (g.send >= g.sstart).map({True: "+", False: "-"})
g["qcov"] = (g.length / g.qlen).round(3)
g.to_csv(f"{out}/element_genome_hits.tsv", sep="\t", index=False)
longh = g[g.qcov >= ec["full_copy_qcov"]]
full = longh[longh.pident >= ec["full_copy_min_pident"]]
div = longh[longh.pident < ec["full_copy_min_pident"]]
log(f"genome hits: {len(g)} (full-length copies, >= {ec['full_copy_qcov']:.0%} of element and >= {ec['full_copy_min_pident']}% identity: "
    f"{len(full)}); by class: " + ", ".join(f"{k}={v}" for k, v in full.subject_class.value_counts().items()))
if len(div):
    log(f"diverged full-length homologs (< {ec['full_copy_min_pident']}% identity): {len(div)} on {div.sseqid.nunique()} chromosomes, "
        f"identity {div.pident.min():.0f}-{div.pident.max():.0f}%; by class: " + ", ".join(f"{k}={v}" for k, v in div.subject_class.value_counts().items()))
log.end()
