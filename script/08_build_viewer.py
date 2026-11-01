"""Step08: build the self-contained interactive HTML viewer.

All results of Step01-07 are packed into one JSON, gzip-compressed and base64-embedded
in script/viewer_template.html (no external libraries / network needed).
Output: result/08_viewer/chrom_end_viewer.html
"""
import base64
import datetime
import gzip
import json
import os

import numpy as np
import pandas as pd

from common import Logger, load_config, path, read_fasta, step_dir, window_label

cfg = load_config()
log = Logger(cfg, "STEP08_viewer")
log.start("interactive HTML viewer", ["windows_kb"], cfg)
R = path(cfg, cfg["result_dir"])
out = step_dir(cfg, "08_viewer")
W = max(cfg["windows_kb"])
labs = [window_label(k) for k in cfg["windows_kb"]]
primary = window_label(cfg["cluster"]["similarity_scale_kb"])
INTRA_CAP = 50  # intrachromosomal copies shown as ">= 50" above this

ends = read_fasta(f"{R}/01_qc/ends_3p_{window_label(W)}.fna")
names = list(ends)
idx = {n: i for i, n in enumerate(names)}
stats = pd.read_csv(f"{R}/01_qc/chr_stats.tsv", sep="\t").set_index("chrom")
bnd = pd.read_csv(f"{R}/02_kmer/terminal_boundary.tsv", sep="\t").set_index("chrom")
types = pd.read_csv(f"{R}/04_cluster/end_types.tsv", sep="\t").set_index("chrom")
feats = pd.read_csv(f"{R}/05_annotation/features.tsv", sep="\t")
isum = pd.read_csv(f"{R}/03_similarity/internal_summary.tsv", sep="\t").set_index("chrom")
FT = ["softmask", "tandem_repeat", "tRNA", "direct_repeat", "inverted_repeat", "terminal_element"]
el_on = pd.read_csv(f"{R}/06_terminal_element/element_on_ends.tsv", sep="\t", keep_default_na=False).set_index("chrom")
el_hits = pd.read_csv(f"{R}/06_terminal_element/element_hits_ends.tsv", sep="\t")
el_ir = pd.read_csv(f"{R}/06_terminal_element/element_inverted_repeats.tsv", sep="\t")
element = list(read_fasta(f"{R}/06_terminal_element/terminal_element.fna").values())[0]
el_feats = pd.DataFrame({"chrom": el_hits.sseqid, "dist_from": el_hits.dist_from, "dist_to": el_hits.dist_to,
                         "type": "terminal_element",
                         "name": [f"element {r.qstart}-{r.qend}/{r.qlen} ({r.pident:.1f}%)" for r in el_hits.itertuples()],
                         "strand": el_hits.strand, "pident": el_hits.pident})
feats = pd.concat([feats, el_feats], ignore_index=True)


def rle(a):
    a = np.asarray(a)
    if not len(a):
        return []
    ch = np.flatnonzero(np.diff(a)) + 1
    st = np.r_[0, ch]
    ln = np.diff(np.r_[st, len(a)])
    return np.column_stack([a[st], ln]).ravel().tolist()


prof = np.load(f"{R}/02_kmer/kmer_profile.npz")
ori_fn = f"{R}/00_orient/orientation.tsv"
ori = pd.read_csv(ori_fn, sep="\t", keep_default_na=False).set_index("chrom") if os.path.exists(ori_fn) else None
CALL_JA = {"as_is": "そのまま", "flip": "反転", "both": "両端", "none": "なし"}
chroms = []
for n in names:
    f = feats[feats.chrom == n]
    chroms.append({
        "name": n, "length": int(stats.loc[n, "length"]), "gc": float(stats.loc[n, "GC"]),
        "softmask": float(stats.loc[n, "softmask_frac"]), "last30": stats.loc[n, "last30"],
        "type": types.loc[n, "end_type"], "types": {l: types.loc[n, f"type_{l}"] for l in labs},
        "offset": int(bnd.loc[n, "terminal_offset"]), "consEnd": int(bnd.loc[n, "conserved_end"]),
        "consLen": int(bnd.loc[n, "conserved_len"]), "maxShare": int(bnd.loc[n, "max_share"]),
        "nTRNA": int((f.type == "tRNA").sum()), "nTR": int((f.type == "tandem_repeat").sum()),
        "element": int(el_on.loc[n, "has_element"]),
        # Step06 "found": a hit containing the element's 3' edge (same criterion as the Step06 log / tiles / micro tab)
        "elemEdge": int(str(el_on.loc[n, "has_element"]) == "1" and str(el_on.loc[n, "has_3p_edge"]) == "1"),
        "elemOverhang": int(el_on.loc[n, "overhang_bp"]) if el_on.loc[n, "has_element"] == 1 else None,
        "elemPident": float(el_on.loc[n, "pident"]) if el_on.loc[n, "has_element"] == 1 else None,
        "orient": (CALL_JA[ori.loc[n, "call"]] + ("（逆相補済）" if int(ori.loc[n, "flipped"]) else "")) if ori is not None and n in ori.index else "",
        "intHits": int(isum.loc[n].filter(like="n_hits_interior").sum()) if n in isum.index else 0,
    })

data = {
    "meta": {
        "genome": cfg["genome_label"], "windows": labs, "maxW": W * 1000, "primaryScale": primary,
        "generated": datetime.date.today().isoformat(), "intraCap": INTRA_CAP,
        "maxInter": int(max(prof[f"{n}__inter_chroms"].max() for n in names)),
        "maxShare": int(max(c["maxShare"] for c in chroms)), "distanceCut": cfg["cluster"]["distance_cut"],
        "typeCounts": types.end_type.value_counts().to_dict(), "internalQueryKb": cfg["blast"]["internal_query_kb"],
        "internalExcludeKb": cfg["blast"]["internal_exclude_kb"],
        "msaRef": cfg["viewer"]["msa_ref_chrom"],
    },
    "chroms": chroms,
    "seq": ends,
    "share": {n: rle(prof[f"{n}__end_share"]) for n in names},          # other 3' ends (tooltip)
    "inter": {n: rle(prof[f"{n}__inter_chroms"]) for n in names},       # interchromosomal similarity
    "intra": {n: rle(np.minimum(prof[f"{n}__intra_copies"], INTRA_CAP)) for n in names},  # intrachromosomal
    "featTypes": FT,
    "feats": {n: [[FT.index(r.type), int(r.dist_from), int(r.dist_to), str(r.name), r.strand]
                  + ([round(float(r.pident), 1)] if r.type == "terminal_element" else [])  # identity to the element
                  for r in feats[feats.chrom == n].itertuples()] for n in names},
    "sim": {}, "hsps": {}, "msa": [],
    "element": {"seq": element, "irs": el_ir[["stem_len", "loop_len", "dist_from", "dist_to", "hairpin_seq"]].values.tolist(),
                "maxOverhang": cfg["terminal_element"]["capped_max_overhang"]},
}

for l in labs:
    M = pd.read_csv(f"{R}/03_similarity/similarity_{l}.tsv", sep="\t", index_col=0).loc[names, names]
    lk = json.load(open(f"{R}/04_cluster/linkage_{l}.json"))
    off = M.to_numpy()[~np.eye(len(names), dtype=bool)]
    data["sim"][l] = {"matrix": M.round(3).to_numpy().tolist(), "order": lk["order"], "names": lk["names"],
                      "linkage": [[int(z[0]), int(z[1]), round(float(z[2]), 4), int(z[3])] for z in lk["linkage"]],
                      "nPairs50": int((off >= 0.5).sum() // 2)}
    h = pd.read_csv(f"{R}/03_similarity/blast_allvsall_{l}.tsv", sep="\t")
    data["hsps"][l] = [[idx[r.qseqid], idx[r.sseqid], int(r.q_dist_from), int(r.q_dist_to), int(r.s_dist_from),
                        int(r.s_dist_to), round(float(r.pident), 1), r.strand] for r in h.itertuples()]

mi = pd.read_csv(f"{R}/04_cluster/msa_index.tsv", sep="\t")
for r in mi.itertuples():
    aln = read_fasta(f"{R}/04_cluster/{r.file}")
    data["msa"].append({"scale": r.window, "group": r.group, "names": list(aln), "seqs": list(aln.values())})

ih = pd.read_csv(f"{R}/03_similarity/internal_hits.tsv", sep="\t")
classes = ["3p_terminal", "5p_terminal", "interior"]
data["internalClasses"] = classes
data["internal"] = [[idx[r.qseqid], int(r.q_dist_from), int(r.q_dist_to), idx[r.sseqid], int(r.sstart), int(r.send),
                     classes.index(r.subject_class), round(float(r.pident), 1), r.strand, round(float(r.bitscore))]
                    for r in ih.itertuples()]
data["meta"]["nInterior"] = int((ih.subject_class == "interior").sum())

data["micro"] = json.load(open(f"{R}/07_micro/micro_viewer.json"))

raw = json.dumps(data, separators=(",", ":")).encode()
gz = gzip.compress(raw, 9)
b64 = base64.b64encode(gz).decode()
tpl = open(path(cfg, "script", "viewer_template.html"), encoding="utf-8").read()
title = f"{cfg['genome_label']} 3′末端構造ビューア"
html = tpl.replace("__TITLE__", title).replace("__DATA_B64__", b64)
fn = f"{out}/chrom_end_viewer.html"
with open(fn, "w", encoding="utf-8") as fh:
    fh.write(html)
log(f"payload JSON {len(raw) / 1e6:.1f} MB -> gzip {len(gz) / 1e6:.1f} MB; HTML {os.path.getsize(fn) / 1e6:.1f} MB")
log(f"MSAs embedded: {len(data['msa'])}; HSPs: {sum(map(len, data['hsps'].values())):,}; internal hits: {len(data['internal']):,}")
log(f"output: {os.path.relpath(fn, cfg['_root'])}")
log.end()
