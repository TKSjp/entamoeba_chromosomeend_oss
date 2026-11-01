"""Step03: all-vs-all BLASTN of 3'-terminal windows (per scale) + search of terminal sequences in the whole genome.

Similarity for scale W (pair i, j):
  S_ij = (identical bases of i covered by HSPs to j, merged) / len_i ; matrix = (S_ij + S_ji) / 2
Outputs (result/03_similarity/):
  blast_allvsall_{W}.tsv        raw HSPs (outfmt 6 + qlen slen) with dist-from-3'-end coordinates
  similarity_{W}.tsv            symmetric similarity matrix (0-1)
  internal_hits.tsv             terminal {internal_query_kb} kb vs genome: hits classified as
                                3p_terminal / 5p_terminal / interior
  internal_summary.tsv          per query chromosome: number / bp of hits per class
"""
import os

import numpy as np
import pandas as pd

from common import Logger, genome_fasta, load_config, path, read_fasta, step_dir, tool_version, window_label

cfg = load_config()
bc = cfg["blast"]
T = cfg["threads"]
log = Logger(cfg, "STEP03_similarity")
log.start("pairwise similarity (BLASTN)", ["blast", "windows_kb"], cfg)
log(f"tool: {tool_version('blastn -version')}")
out = step_dir(cfg, "03_similarity")
qc = path(cfg, cfg["result_dir"], "01_qc")
genome = genome_fasta(cfg)
seqs = read_fasta(genome)
names = list(seqs)
COLS = "qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen nident"
BOPT = f"-task {bc['task']} -evalue {bc['evalue']} -num_threads {T} -outfmt '6 {COLS}' -max_target_seqs 1000"


def add_dist(df):
    """dist-from-3'-end coordinates (last base = 1). q/s 'from' is the end-proximal side."""
    df["q_dist_from"] = df.qlen - df.qend + 1
    df["q_dist_to"] = df.qlen - df.qstart + 1
    lo, hi = df[["sstart", "send"]].min(axis=1), df[["sstart", "send"]].max(axis=1)
    df["s_dist_from"] = df.slen - hi + 1
    df["s_dist_to"] = df.slen - lo + 1
    df["strand"] = np.where(df.send >= df.sstart, "+", "-")
    return df


def merged_identical(intervals):
    """intervals: list of (start, end, nident, length) on the query; approx identical bases after merging."""
    if not intervals:
        return 0
    intervals.sort()
    tot, cur_s, cur_e, cur_id = 0, None, None, 0.0
    for s, e, ni, ln in intervals:
        rate = ni / ln
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_id
            cur_s, cur_e, cur_id = s, e, (e - s + 1) * rate
        elif e > cur_e:
            cur_id += (e - cur_e) * rate
            cur_e = e
    return tot + cur_id


for kb in cfg["windows_kb"]:
    lab = window_label(kb)
    fa = f"{qc}/ends_3p_{lab}.fna"
    db = f"{out}/db_{lab}"
    log.run(f"makeblastdb -in {fa} -dbtype nucl -out {db} > /dev/null")
    raw = f"{out}/blast_allvsall_{lab}.raw"
    log.run(f"blastn -query {fa} -db {db} {BOPT} -out {raw}")
    df = pd.read_csv(raw, sep="\t", names=COLS.split())
    df = df[(df.length >= bc["min_hsp_len"]) & (df.qseqid != df.sseqid)].copy()
    df = add_dist(df)
    df.to_csv(f"{out}/blast_allvsall_{lab}.tsv", sep="\t", index=False)
    os.remove(raw)
    for ext in os.listdir(out):
        if ext.startswith(f"db_{lab}."):
            os.remove(f"{out}/{ext}")
    lens = {n: min(len(seqs[n]), kb * 1000) for n in names}
    S = pd.DataFrame(0.0, index=names, columns=names)
    for (q, s), g in df.groupby(["qseqid", "sseqid"]):
        iv = [(r.qstart, r.qend, r.nident, r.length) for r in g.itertuples()]
        S.loc[q, s] = merged_identical(iv) / lens[q]
    arr = (S.to_numpy() + S.to_numpy().T) / 2
    np.fill_diagonal(arr, 1.0)
    M = pd.DataFrame(arr, index=names, columns=names)
    M.round(4).to_csv(f"{out}/similarity_{lab}.tsv", sep="\t")
    off = M.values[~np.eye(len(names), dtype=bool)]
    log(f"{lab}: {len(df):,} HSPs; similarity off-diagonal mean {off.mean():.3f}, "
        f"pairs >=0.5: {(off >= 0.5).sum() // 2}")

# --- terminal sequences vs whole genome
qkb, exkb = bc["internal_query_kb"], bc["internal_exclude_kb"]
db = f"{out}/db_genome"
log.run(f"makeblastdb -in {genome} -dbtype nucl -out {db} > /dev/null")
raw = f"{out}/internal_hits.raw"
log.run(f"blastn -query {qc}/ends_3p_{window_label(qkb)}.fna -db {db} {BOPT} -out {raw}")
df = pd.read_csv(raw, sep="\t", names=COLS.split())
df = df[df.length >= bc["min_hsp_len"]].copy()
os.remove(raw)
for f in os.listdir(out):
    if f.startswith("db_genome."):
        os.remove(f"{out}/{f}")
lo, hi = df[["sstart", "send"]].min(axis=1), df[["sstart", "send"]].max(axis=1)
ex = exkb * 1000
cls = np.where(df.slen - lo + 1 <= ex, "3p_terminal", np.where(hi <= ex, "5p_terminal", "interior"))
df["subject_class"] = cls
df["strand"] = np.where(df.send >= df.sstart, "+", "-")
df["q_dist_from"] = df.qlen - df.qend + 1
df["q_dist_to"] = df.qlen - df.qstart + 1
df["s_dist_from_3p"] = df.slen - hi + 1
self_hit = (df.qseqid == df.sseqid) & (df.subject_class == "3p_terminal")
df = df[~self_hit]
df.to_csv(f"{out}/internal_hits.tsv", sep="\t", index=False)
summ = (df.groupby(["qseqid", "subject_class"]).agg(n_hits=("length", "size"), aligned_bp=("length", "sum"))
        .unstack(fill_value=0))
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reindex(names, fill_value=0)
summ.to_csv(f"{out}/internal_summary.tsv", sep="\t", index_label="chrom")
for c, n in df.subject_class.value_counts().items():
    log(f"terminal {qkb}kb vs genome: {c} hits = {n:,}")
log.end()
