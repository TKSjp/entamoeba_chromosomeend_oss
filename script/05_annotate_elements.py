"""Step05: annotate elements inside the largest 3'-terminal window.

Features (all reported in genome coordinates and in dist-from-3'-end coordinates):
  tandem_repeat     TRF
  tRNA              tRNAscan-SE (eukaryotic model)
  softmask          lower-case runs of the input FASTA (repeat-masked)
  direct_repeat / inverted_repeat   self-BLASTN of each window (off-diagonal HSPs)
Also: GC per bin, end-anchored terminal motif (last N bp).
Outputs (result/05_annotation/):
  features.tsv / features.bed   all features
  self_blast.tsv                self-BLASTN HSPs (for dot plots)
  gc_bins.tsv                   GC / soft-mask fraction per bin (dist coordinates)
  terminal_motif.fna / terminal_motif_freq.tsv   last N bp (end-anchored) and base frequency per position
  trf.ngs.txt / trnascan.txt    raw tool output
"""
import os
import re
import subprocess

import pandas as pd

from common import Logger, load_config, path, read_fasta, step_dir, tool_version, window_label, write_fasta

cfg = load_config()
ac = cfg["annot"]
T = cfg["threads"]
W = max(cfg["windows_kb"])
lab = window_label(W)
log = Logger(cfg, "STEP05_annotation")
log.start("element annotation", ["trf", "trnascan", "annot"], cfg)
log(f"tools: {tool_version('tRNAscan-SE -h 2>&1 | head -2 | tail -1')} ; blastn {tool_version('blastn -version')}")
out = step_dir(cfg, "05_annotation")
qc = path(cfg, cfg["result_dir"], "01_qc")
fa = f"{qc}/ends_3p_{lab}.fna"
ends = read_fasta(fa)
win = pd.read_csv(f"{qc}/ends_3p_windows.tsv", sep="\t")
win = win[win.window == lab].set_index("chrom")
feats = []  # chrom, wstart, wend (1-based in window), type, name, score, strand


def add(chrom, s, e, ftype, name, score=".", strand="."):
    feats.append((chrom, min(s, e), max(s, e), ftype, name, score, strand))


# --- TRF
trf_out = f"{out}/trf.ngs.txt"
log.run(f"trf {fa} {cfg['trf']['params']} -d -h -ngs > {trf_out}", cwd=out)
cur = None
with open(trf_out) as fh:
    for line in fh:
        if line.startswith("@"):
            cur = line[1:].split()[0]
            continue
        f = line.split()
        if len(f) < 14:
            continue
        s, e, period, copies = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        add(cur, s, e, "tandem_repeat", f"period={period};copies={copies};unit={f[13]}", f[7])
log(f"TRF: {sum(x[3] == 'tandem_repeat' for x in feats)} tandem repeats")

# --- tRNAscan-SE
trna_out = f"{out}/trnascan.txt"
if os.path.exists(trna_out):
    os.remove(trna_out)
log.run(f"tRNAscan-SE {cfg['trnascan']['options']} --brief --thread {T} -o {trna_out} {fa}")
n0 = len(feats)
with open(trna_out) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or not f[2].strip().isdigit():
            continue
        b, e = int(f[2]), int(f[3])
        add(f[0].strip(), b, e, "tRNA", f"tRNA-{f[4].strip()}({f[5].strip()})", f[8].strip(), "+" if e >= b else "-")
log(f"tRNAscan-SE: {len(feats) - n0} tRNAs")

# --- soft-mask runs
n0 = len(feats)
for c, s in ends.items():
    for m in re.finditer(r"[a-z]+", s):
        add(c, m.start() + 1, m.end(), "softmask", "lowercase")
log(f"soft-mask runs: {len(feats) - n0}")

# --- self BLAST (direct / inverted repeats inside each window)
rows = []
tmp = f"{out}/_self.fna"
self_cmd = (f"blastn -task blastn -query {tmp} -subject {tmp} -evalue 1e-5 -dust yes "
            f"-outfmt '6 qstart qend sstart send pident length bitscore'")
log(f"self-BLAST per chromosome window: {self_cmd}")
for c, s in ends.items():
    write_fasta(tmp, [(c, s.upper())])
    r = subprocess.run(self_cmd, shell=True, capture_output=True, text=True, check=True)
    for ln in r.stdout.splitlines():
        qs, qe, ss, se, pid, ln_, bs = ln.split("\t")
        qs, qe, ss, se, ln_ = int(qs), int(qe), int(ss), int(se), int(ln_)
        if ln_ < ac["self_blast_min_len"] or (qs == ss and qe == se):
            continue
        rows.append((c, qs, qe, ss, se, float(pid), ln_, float(bs)))
os.remove(tmp)
sb = pd.DataFrame(rows, columns=["chrom", "qstart", "qend", "sstart", "send", "pident", "length", "bitscore"])
sb["strand"] = (sb.send >= sb.sstart).map({True: "+", False: "-"})
sb.to_csv(f"{out}/self_blast.tsv", sep="\t", index=False)
for r in sb[sb.qstart < sb[["sstart", "send"]].min(axis=1)].itertuples():  # one copy of each symmetric pair
    ft = "direct_repeat" if r.strand == "+" else "inverted_repeat"
    partner = f"{min(r.sstart, r.send)}-{max(r.sstart, r.send)}"
    add(r.chrom, r.qstart, r.qend, ft, f"partner={partner};pident={r.pident}", r.bitscore, r.strand)
log(f"self-BLAST: {len(sb)} HSPs ({(sb.strand == '-').sum()} inverted)")

# --- write features with both coordinate systems
recs = []
for c, s, e, ft, nm, sc, st in feats:
    wl = int(win.loc[c, "win_length"])
    off = int(win.loc[c, "start"]) - 1
    recs.append((c, off + s, off + e, wl - e + 1, wl - s + 1, ft, nm, sc, st))
df = pd.DataFrame(recs, columns=["chrom", "start", "end", "dist_from", "dist_to", "type", "name", "score", "strand"])
df = df.sort_values(["chrom", "dist_from"])
df.to_csv(f"{out}/features.tsv", sep="\t", index=False)
with open(f"{out}/features.bed", "w") as fh:
    for r in df.itertuples():
        fh.write(f"{r.chrom}\t{r.start - 1}\t{r.end}\t{r.type}:{r.name}\t{r.score if r.score != '.' else 0}\t{r.strand}\n")
log("feature counts: " + ", ".join(f"{k}={v}" for k, v in df.type.value_counts().items()))

# --- GC / softmask bins (dist coords)
B = ac["gc_bin_bp"]
with open(f"{out}/gc_bins.tsv", "w") as fh:
    fh.write("chrom\tbin_from_dist\tbin_to_dist\tGC\tsoftmask_frac\n")
    for c, s in ends.items():
        r = s[::-1]
        for b0 in range(0, len(r), B):
            t = r[b0:b0 + B]
            u = t.upper()
            gcv = (u.count("G") + u.count("C")) / len(t)
            lm = sum(ch.islower() for ch in t) / len(t)
            fh.write(f"{c}\t{b0 + 1}\t{b0 + len(t)}\t{gcv:.3f}\t{lm:.3f}\n")

# --- terminal motif (end-anchored, no alignment)
N = ac["terminal_motif_bp"]
write_fasta(f"{out}/terminal_motif.fna", [(c, s[-N:]) for c, s in ends.items()])
with open(f"{out}/terminal_motif_freq.tsv", "w") as fh:
    fh.write("dist\tA\tC\tG\tT\n")
    for d in range(1, N + 1):
        col = [s[-d].upper() for s in ends.values()]
        fh.write(f"{d}\t" + "\t".join(f"{col.count(b) / len(col):.3f}" for b in "ACGT") + "\n")
log.end()
