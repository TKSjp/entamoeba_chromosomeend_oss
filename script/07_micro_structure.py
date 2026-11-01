"""Step07: micro-scale architecture of the chromosome 3' ends: the conserved terminal core and its flank.

1. Core length C = length of the terminal element (Step06, change point). For every chromosome carrying the
   element, take the C + `window_margin_bp` bases that end at the element's 3' edge (the "extension" beyond it is
   handled separately) and align them with MAFFT L-INS-i.
2. Column support = fraction of ALL sequences carrying the majority base.
   Inside the core (consensus distance from the element 3' edge <= C):
     variable sites: support < core_support; sites within link_bp are joined into variable regions
                     (insertion columns present in < ignore_insertion_below of the ends are not variable sites;
                      they remain inside the surrounding segment as insertions of those ends; variable regions
                      spanning < min_variable_columns columns are single SNPs and stay inside the module)
     module (T1, T2, ... numbered from the 3' end)     conserved stretch between variable regions, >= min_module_bp
     variable region (V1, V2, ... from the 3' end)     everything else
   Outside the core: outer region (O1, ...), not split into modules / variable regions.
   Both: shared block (S1, ...) inside long variable regions (>= shared_search_min_bp) and in the outer region:
         smoothed support >= shared_support, >= min_shared_bp (= sequence shared by a subset of the ends);
         homopolymer tract (PolyT, ...) = consensus run of one base >= homopolymer_min.
3. Per chromosome and segment: presence, length, identity to the consensus, substitution /
   insertion / deletion positions, and a sequence variant label (a, b, ... greedy clustering).
4. Family (A, B, ...) = average-linkage clusters of the core (MSA identity >= family_identity);
   micro type (A1, A2, ...) = combination of variant labels of the core segments (modules, variable regions,
   shared blocks); the whole-window haplotype (also shared blocks of the outer region) is reported as well.
5. Chromosomes without the element: module consensus sequences (>= scan_min_module_bp) are searched in their
   terminal window of the same length by local alignment.
Outputs (result/07_micro/):
  micro_input.fna / micro_msa.fasta         anchored input and alignment
  segments.tsv / segment_consensus.fna      segment definitions
  segment_by_chrom.tsv                      per chromosome x segment (coordinates, length, identity, variant, sequence)
  micro_types.tsv                           micro type per chromosome
  nonelement_module_hits.tsv                module hits in ends without the element
  micro_viewer.json                         data for the viewer (Step08)
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from Bio import Align
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from common import Logger, genome_fasta, load_config, path, read_fasta, revcomp, step_dir, tool_version, write_fasta

cfg = load_config()
mc = cfg["micro"]
log = Logger(cfg, "STEP07_micro")
log.start("micro-scale terminal architecture", ["micro"], cfg)
log(f"tool: mafft {tool_version('mafft --version')}")
out = step_dir(cfg, "07_micro")
R = path(cfg, cfg["result_dir"])
seqs = read_fasta(genome_fasta(cfg))
types = pd.read_csv(f"{R}/04_cluster/end_types.tsv", sep="\t").set_index("chrom")
on = pd.read_csv(f"{R}/06_terminal_element/element_on_ends.tsv", sep="\t", keep_default_na=False)
C = len(list(read_fasta(f"{R}/06_terminal_element/terminal_element.fna").values())[0])  # conserved terminal core
W = C + mc["window_margin_bp"]
log(f"conserved terminal core (= element) {C} bp; analysis window {W} bp")

# ---------- 1. anchored input + MAFFT
for c in ("has_element", "has_3p_edge", "overhang_bp"):
    on[c] = pd.to_numeric(on[c], errors="coerce")
elem = on[(on.has_element == 1) & (on.has_3p_edge == 1)].copy()
elem["overhang"] = elem.overhang_bp.astype(int).clip(lower=0)
recs, ext = [], {}
for r in elem.itertuples():
    s = seqs[r.chrom]
    anchored = s[:len(s) - r.overhang] if r.overhang else s
    recs.append((r.chrom, anchored[-W:]))
    ext[r.chrom] = s[len(s) - r.overhang:] if r.overhang else ""
write_fasta(f"{out}/micro_input.fna", recs)
log.run(f"mafft {mc['mafft_options']} --thread {cfg['threads']} {out}/micro_input.fna > {out}/micro_msa.fasta 2> {out}/micro_mafft.log")
aln = read_fasta(f"{out}/micro_msa.fasta")
names = list(aln)
A = np.array([list(aln[n].upper()) for n in names])
N, L = A.shape
log(f"aligned {N} element-bearing ends ({W} bp each, anchored at the element 3' edge): {L} columns")

# ---------- 2. column statistics and segmentation
pres, sup, maj_ng, cons = np.zeros(L), np.zeros(L), [], []
for j in range(L):
    col = A[:, j]
    ng = col[col != "-"]
    pres[j] = len(ng) / N
    b, c = Counter(ng).most_common(1)[0] if len(ng) else ("-", 0)
    sup[j] = c / N
    maj_ng.append(b)
    cons.append(b if pres[j] >= 0.5 else "-")
is_cons = pres >= 0.5  # columns that belong to the consensus
core = np.cumsum(is_cons[::-1])[::-1] <= C  # columns within C consensus bases of the element 3' edge
# (a) variable sites (core only) = columns with support < core_support; sites closer than link_bp form one variable cluster.
#     Insertion columns carried by only a few ends (presence < ignore_insertion_below) are not variable sites:
#     they stay inside the surrounding segment and appear as insertions of those ends.
minor_ins = pres < mc["ignore_insertion_below"]
var_site = (sup < mc["core_support"]) & ~minor_ins & core
log(f"minor insertion columns (present in < {mc['ignore_insertion_below']:.0%} of ends) ignored for segment boundaries: {int(minor_ins.sum())}")
lab = np.where(core, "M", "O").astype(object)
idx = np.flatnonzero(var_site)
if len(idx):
    st = prev = idx[0]
    for j in list(idx[1:]) + [None]:
        if j is None or j - prev > mc["link_bp"]:
            lab[st:prev + 1] = "V"
            if j is not None:
                st = j
        if j is not None:
            prev = j


def runs(x):
    out_, s = [], 0
    for j in range(1, len(x) + 1):
        if j == len(x) or x[j] != x[s]:
            out_.append([x[s], s, j])
            s = j
    return out_


# isolated variable sites spanning fewer than min_variable_columns columns (single SNPs) do not split modules:
# they stay inside the module and appear as substitutions of the ends that carry them
for t, s, e in runs(list(lab)):
    if t == "V" and e - s < mc.get("min_variable_columns", 1):
        lab[s:e] = "M"


def clean(lab):
    """modules / shared blocks whose CONSENSUS length (not column count: minor insertion columns do not
    count) is below the minimum become variable (core) or outer region (outside the core)."""
    changed = True
    while changed:
        changed = False
        for t, s, e in runs(list(lab)):
            cl = int(is_cons[s:e].sum())
            if (t == "M" and cl < mc["min_module_bp"]) or (t == "S" and cl < mc["min_shared_bp"]):
                lab[s:e] = np.where(core[s:e], "V", "O"); changed = True; break


clean(lab)
# (b) inside long variable clusters and in the outer region: blocks shared by a subset of ends (smoothed support >= shared_support)
k = mc["smooth_bp"]
sm = np.convolve(sup, np.ones(k) / k, mode="same")
for t, s, e in runs(list(lab)):
    if (t == "V" and e - s >= mc["shared_search_min_bp"]) or t == "O":
        for j in range(s, e):
            if sm[j] >= mc["shared_support"]:
                lab[j] = "S"
clean(lab)
# (c) homopolymer tracts on the majority-of-non-gap consensus (columns present in >= 15% of sequences)
hp_str = "".join(b if pres[j] >= 0.15 else "#" for j, b in enumerate(maj_ng))
for m in re.finditer(r"(A{%d,}|C{%d,}|G{%d,}|T{%d,})" % ((mc["homopolymer_min"],) * 4), hp_str):
    lab[m.start():m.end()] = "H:" + m.group(0)[0]
clean(lab)

segs = []
nm = nv = ns = no = 0
hp_count = Counter()
for t, s, e in reversed(runs(list(lab))):          # name from the 3' end
    if t == "M":
        nm += 1; name, kind = f"T{nm}", "module"
    elif t == "S":
        ns += 1; name, kind = f"S{ns}", "shared_block"
    elif t == "V":
        nv += 1; name, kind = f"V{nv}", "variable"
    elif t == "O":
        no += 1; name, kind = f"O{no}", "outer"
    else:
        base = t.split(":")[1]; hp_count[base] += 1
        name, kind = f"Poly{base}" + (str(hp_count[base]) if hp_count[base] > 1 else ""), "homopolymer"
    segs.append({"name": name, "kind": kind, "c0": s, "c1": e})
segs.reverse()  # 5' -> 3' order


def cons_len(c0, c1):
    return sum(1 for j in range(c0, c1) if cons[j] != "-")


# consensus coordinates as distance from the element 3' edge
tot = cons_len(0, L)
acc = 0
for sg in segs:
    sg["cons_seq"] = "".join(cons[j] for j in range(sg["c0"], sg["c1"]) if cons[j] != "-")
    sg["cons_len"] = len(sg["cons_seq"])
    sg["cons_dist_to"] = tot - acc
    acc += sg["cons_len"]
    sg["cons_dist_from"] = tot - acc + 1
    sg["support"] = round(float(sup[sg["c0"]:sg["c1"]].mean()), 3)
    sg["gap_frac"] = round(float(1 - pres[sg["c0"]:sg["c1"]].mean()), 3)

# ---------- 3. per chromosome x segment
def seg_detail(n, sg):
    """walk the segment columns of chromosome n: length, identity, variant events (offsets in the chromosome)."""
    row = aln[n]
    pos = 0  # bases of this chromosome consumed before the segment
    for j in range(sg["c0"]):
        pos += row[j] != "-"
    seq, events, off, match, comp = [], [], 0, 0, 0
    for j in range(sg["c0"], sg["c1"]):
        b, c = row[j], cons[j]
        if b != "-":
            seq.append(b)
            if c == "-":
                events.append(["ins", off])
            else:
                comp += 1
                if b.upper() == c:
                    match += 1
                else:
                    events.append(["sub", off, b])
            off += 1
        elif c != "-":
            comp += 1
            events.append(["del", off])
    if sg["cons_len"] == 0:  # insertion-only segment: present when the chromosome has bases here
        return pos, "".join(seq), events, 0.0, float(len(seq) > 0)
    present_cov = sum(1 for j in range(sg["c0"], sg["c1"]) if row[j] != "-" and cons[j] != "-") / sg["cons_len"]
    return pos, "".join(seq), events, (match / comp if comp else 0.0), present_cov


rows = []
for n in names:
    total = len(aln[n].replace("-", ""))
    oh = int(elem.set_index("chrom").loc[n, "overhang"])
    L_chr = len(seqs[n])
    for sg in segs:
        pos, sq, ev, ident, cov = seg_detail(n, sg)
        present = len(sq) > 0 and cov >= mc["present_min_cov"]
        # distance from the chromosome 3' end (element-anchored distance + extension)
        d_to = total - pos + oh
        d_from = d_to - len(sq) + 1
        rows.append({"chrom": n, "segment": sg["name"], "kind": sg["kind"], "present": int(present),
                     "length": len(sq), "dist_from": d_from if sq else None, "dist_to": d_to if sq else None,
                     "genome_start": L_chr - d_to + 1 if sq else None, "genome_end": L_chr - d_from + 1 if sq else None,
                     "identity": round(ident, 3), "coverage": round(cov, 3), "seq": sq, "events": ev})
sbc = pd.DataFrame(rows)


def msa_identity(x, y):
    """identity of two gapped MSA slices = matching bases / columns where at least one is not a gap."""
    cols = [(a, b) for a, b in zip(x, y) if a != "-" or b != "-"]
    return sum(a == b for a, b in cols) / len(cols) if cols else 1.0


def variants(sg):
    """greedy clustering of segment sequences (most frequent first), joined when MSA identity >= variant_identity."""
    d = sbc[(sbc.segment == sg["name"]) & (sbc.present == 1)]
    sl = {n: aln[n][sg["c0"]:sg["c1"]].upper() for n in d.chrom}
    reps, lab_ = [], {}
    for x, _ in Counter(sl.values()).most_common():
        best = next((i for i, r in enumerate(reps) if msa_identity(x, r) >= mc["variant_identity"]), None)
        if best is None:
            reps.append(x)
            best = len(reps) - 1
        lab_[x] = "abcdefghijklmnopqrstuvwxyz"[best] if best < 26 else str(best)
    return {n: lab_[x] for n, x in sl.items()}


var = {}
for sg in segs:
    v = variants(sg) if sg["kind"] not in ("homopolymer", "outer") else {}
    var[sg["name"]] = v
    d = sbc[(sbc.segment == sg["name"]) & (sbc.present == 1)]
    sg["n_present"] = int(len(d))
    sg["len_min"] = int(d.length.min()) if len(d) else 0
    sg["len_max"] = int(d.length.max()) if len(d) else 0
    sg["mean_identity"] = round(float(d.identity.mean()), 3) if len(d) else 0
    sg["n_variants"] = len(set(v.values())) if v else (1 if len(d) else 0)
sbc["variant"] = [var[r.segment].get(r.chrom, "" if r.kind in ("homopolymer", "outer") else "-") if r.present
                  else "-" for r in sbc.itertuples()]

# ---------- 4. micro types
er = C  # micro types are defined over the conserved terminal core
log(f"micro type region: the conserved core, 1-{er} bp from the element 3' edge")
el_segs = [sg["name"] for sg in segs if sg["cons_dist_from"] <= er and sg["kind"] not in ("homopolymer", "outer")][::-1]
all_segs = [sg["name"] for sg in segs if sg["kind"] not in ("homopolymer", "outer")][::-1]


def code(n, seg_list):
    d = sbc[sbc.chrom == n].set_index("segment")
    return " ".join(f"{s}{d.loc[s, 'variant']}" for s in seg_list)


def name_groups(codes, prefix):
    cnt = Counter(codes.values())
    order = sorted(cnt, key=lambda c: (-cnt[c], c))
    m = {c: f"{prefix}{i + 1}" for i, c in enumerate(order)}
    return {n: m[c] for n, c in codes.items()}, m


el_codes = {n: code(n, el_segs) for n in names}
kb_codes = {n: code(n, all_segs) for n in names}
kb_type, kb_map = name_groups(kb_codes, "K")

# families: average-linkage clustering on MSA identity of the core, cut at family_identity
c_from = int(np.argmax(core))
D = np.array([[1 - msa_identity(aln[a][c_from:].upper(), aln[b][c_from:].upper()) for b in names] for a in names])
Z = linkage(squareform(D, checks=False), method="average")
fam_id = fcluster(Z, t=1 - mc["family_identity"], criterion="distance")
fam_order = [c for c, _ in Counter(fam_id).most_common()]
family = {n: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[fam_order.index(f)] for n, f in zip(names, fam_id)}
# micro types nested in families: A1, A2, ... (by size), defined by the exact variant combination
el_type, el_map = {}, {}
for F in sorted(set(family.values())):
    mem = [n for n in names if family[n] == F]
    t_, m_ = name_groups({n: el_codes[n] for n in mem}, F)
    el_type.update(t_)
    el_map.update(m_)
fam_ident = {F: round(float(np.mean([1 - D[i, j] for i, a in enumerate(names) for j, b in enumerate(names)
                                     if i < j and family[a] == family[b] == F])) if sum(v == F for v in family.values()) > 1 else 1.0, 3)
             for F in set(family.values())}
for F in sorted(fam_ident):
    log(f"family {F}: n={sum(v == F for v in family.values())}, mean pairwise identity {fam_ident[F]} "
        f"({', '.join(n for n in names if family[n] == F)})")
mt = pd.DataFrame({"chrom": names, "family": [family[n] for n in names], "micro_type": [el_type[n] for n in names], "element_code": [el_codes[n] for n in names],
                   "kb_haplotype": [kb_type[n] for n in names], "kb_code": [kb_codes[n] for n in names],
                   "end_type": [types.loc[n, "end_type"] for n in names],
                   "extension_bp": [len(ext[n]) for n in names],
                   "polyT_len": [int(sbc[(sbc.chrom == n) & (sbc.kind == "homopolymer")].length.max(skipna=True))
                                 if ((sbc.chrom == n) & (sbc.kind == "homopolymer")).any() else 0 for n in names]})
log(f"segments (5'->3'): " + " | ".join(f"{sg['name']}({sg['kind'][0]},{sg['cons_len']}bp)" for sg in segs))
for t, c in sorted(el_map.items(), key=lambda x: (x[1][0], int(x[1][1:]))):
    mem = [n for n in names if el_type[n] == c]
    log(f"micro type {c} (n={len(mem)}): {t}  <- {', '.join(mem)}")
log(f"{W} bp haplotypes: {len(kb_map)} distinct among {N} ends")

# ---------- 5. ends without the element: local search of module consensus
loc = Align.PairwiseAligner(mode="local", match_score=2, mismatch_score=-3, open_gap_score=-5, extend_gap_score=-2)
hits = []
mods = [sg for sg in segs if sg["kind"] == "module" and sg["cons_len"] >= mc.get("scan_min_module_bp", 0)]
scan_w = W
for n in [c for c in seqs if c not in names]:
    s = seqs[n].upper()
    win = s[-scan_w:]
    for sg in mods:
        q = sg["cons_seq"]
        for strand, tgt in (("+", win), ("-", revcomp(win))):
            a = loc.align(tgt, q)[0]
            ts, te = a.aligned[0][0][0], a.aligned[0][-1][1]
            qs, qe = a.aligned[1][0][0], a.aligned[1][-1][1]
            c = a.counts()
            alen = c.identities + c.mismatches + c.gaps
            ident = c.identities / alen if alen else 0
            qcov = (qe - qs) / len(q)
            if ident >= mc["scan_identity"] and qcov >= mc["scan_coverage"]:
                if strand == "+":
                    d_to, d_from = len(win) - ts, len(win) - te + 1
                else:
                    d_from, d_to = ts + 1, te
                hits.append({"chrom": n, "segment": sg["name"], "strand": strand, "dist_from": d_from, "dist_to": d_to,
                             "identity": round(ident, 3), "qcov": round(qcov, 3), "seq": seqs[n][len(s) - d_to:len(s) - d_from + 1]})
nh = pd.DataFrame(hits, columns=["chrom", "segment", "strand", "dist_from", "dist_to", "identity", "qcov", "seq"])
log(f"ends without the element: {len(seqs) - N}; module hits ({len(mods)} modules >= {mc.get('scan_min_module_bp', 0)} bp) "
    f"in their terminal {scan_w} bp: {len(nh)}"
    + (" (" + ", ".join(f"{r.chrom}:{r.segment}{r.strand}@{r.dist_from}-{r.dist_to}" for r in nh.head(20).itertuples())
       + (", ..." if len(nh) > 20 else "") + ")" if len(nh) else ""))

# ---------- outputs
seg_df = pd.DataFrame([{k: v for k, v in sg.items() if k not in ("c0", "c1")} | {"aln_col_from": sg["c0"] + 1, "aln_col_to": sg["c1"]}
                       for sg in segs])
seg_df.to_csv(f"{out}/segments.tsv", sep="\t", index=False)
write_fasta(f"{out}/segment_consensus.fna", [(f"{sg['name']} {sg['kind']} cons_dist={sg['cons_dist_from']}-{sg['cons_dist_to']}", sg["cons_seq"])
                                             for sg in segs if sg["cons_seq"]])
sbc.drop(columns="events").to_csv(f"{out}/segment_by_chrom.tsv", sep="\t", index=False)
mt.to_csv(f"{out}/micro_types.tsv", sep="\t", index=False)
nh.to_csv(f"{out}/nonelement_module_hits.tsv", sep="\t", index=False)
def nn(v):
    return None if pd.isna(v) else int(v)


viewer = {
    "windowBp": W, "coreBp": C, "elementRegionBp": er, "nAligned": N,
    "segments": [{k: sg[k] for k in ("name", "kind", "cons_len", "cons_dist_from", "cons_dist_to", "support", "gap_frac",
                                     "n_present", "len_min", "len_max", "mean_identity", "n_variants", "cons_seq")} for sg in segs],
    "colVar": [[sg["name"], [round(float(1 - sup[j]), 2) for j in range(sg["c0"], sg["c1"]) if cons[j] != "-"],
                [round(float(1 - pres[j]), 2) for j in range(sg["c0"], sg["c1"]) if cons[j] != "-"]] for sg in segs],
    "chroms": {n: {"family": family[n], "microType": el_type[n], "kbHap": kb_type[n], "code": el_codes[n], "ext": ext[n],
                   "segs": [{"name": r.segment, "present": r.present, "len": r.length, "from": nn(r.dist_from), "to": nn(r.dist_to),
                             "id": r.identity, "var": r.variant, "seq": r.seq, "ev": r.events}
                            for r in sbc[sbc.chrom == n].itertuples()]} for n in names},
    "typeRegionBp": er,
    "families": [{"family": F, "members": [n for n in names if family[n] == F], "identity": fam_ident[F]} for F in sorted(fam_ident)],
    "types": [{"type": c, "code": t, "family": c[0], "members": [n for n in names if el_type[n] == c]}
              for t, c in sorted(el_map.items(), key=lambda x: (x[1][0], int(x[1][1:])))],
    "nonElement": {"scanBp": scan_w, "chroms": [c for c in seqs if c not in names], "hits": nh.to_dict(orient="records")},
}
with open(f"{out}/micro_viewer.json", "w") as fh:
    json.dump(viewer, fh, separators=(",", ":"), allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log.end()
