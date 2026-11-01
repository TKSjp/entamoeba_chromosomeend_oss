"""Step00: contig orientation check against a reference chromosome's terminal motif.

The 3'-terminal `ref_bp` of the reference chromosome (`ref_chrom`, orientation taken as correct) is
the terminal motif. For every contig both ends are tested in "3'-end orientation":
  3p window = last `search_window_kb` kb as given
  5p window = reverse complement of the first `search_window_kb` kb (= 3' end if the contig is flipped)
BLASTN motif vs windows; a hit counts when it is on the + strand of the window, covers >= min_qcov of
the motif and its 3' edge lies within `max_dist_bp` of the window end.
  as_is : motif only at the 3' end      flip : motif only at the 5' end (reverse-complemented when apply)
  both  : motif at both ends (kept)     none : no motif (kept)
Outputs (result/<genome>/00_orient/):
  reference_motif.fna      the motif
  orientation.tsv          per contig: best hit at each end and the call
  <genome>_analysis.fna    genome used by Step01-07: `exclude_contigs` (e.g. plasmids) removed and, when
                           orient.apply is true, "flip" contigs reverse-complemented
                           (written when either applies; otherwise Step01-07 use the input FASTA)
"""
import subprocess

import pandas as pd

from common import Logger, genome_fasta, load_config, path, read_fasta, revcomp, step_dir, write_fasta

cfg = load_config()
oc = cfg["orient"]
log = Logger(cfg, "STEP00_orient")
log.start("contig exclusion and orientation check", ["input_fasta", "exclude_contigs", "orient"], cfg)
out = step_dir(cfg, "00_orient")
seqs = read_fasta(path(cfg, cfg["input_fasta"]))
excl = list(cfg.get("exclude_contigs") or [])
missing = [c for c in excl if c not in seqs]
if missing:
    raise SystemExit(f"exclude_contigs not in {cfg['input_fasta']}: {', '.join(missing)}")
if excl:
    log("excluded from the analysis (exclude_contigs): " + ", ".join(f"{c} ({len(seqs[c]):,} bp)" for c in excl))
    seqs = {c: s for c, s in seqs.items() if c not in excl}
ref = oc["ref_chrom"]
if ref not in seqs:
    raise SystemExit(f"reference chromosome {ref} not in {cfg['input_fasta']}")
W, R = oc["search_window_kb"] * 1000, oc["ref_bp"]
motif = seqs[ref][-R:]
write_fasta(f"{out}/reference_motif.fna", [(f"{ref}_3prime_{R}bp", motif)])
log(f"reference motif: 3'-terminal {R} bp of {ref} ...{motif[-50:]}")

wins = []
for c, s in seqs.items():
    wins.append((f"{c}|3p", s[-W:].upper()))
    wins.append((f"{c}|5p", revcomp(s[:W]).upper()))
write_fasta(f"{out}/_windows.fna", wins)
cols = "qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen"
cmd = (f"blastn -task blastn -query {out}/reference_motif.fna -subject {out}/_windows.fna -evalue {oc['evalue']} "
       f"-dust no -max_target_seqs 10000 -outfmt '6 {cols}'")
log(f"$ {cmd}")
res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
subprocess.run(f"rm -f {out}/_windows.fna", shell=True)
h = pd.DataFrame([ln.split("\t") for ln in res.stdout.splitlines()], columns=cols.split())
for c in cols.split()[2:]:
    h[c] = pd.to_numeric(h[c])
h["chrom"] = h.sseqid.str.split("|").str[0]
h["end"] = h.sseqid.str.split("|").str[1]
h["qcov"] = (h.qend - h.qstart + 1) / h.qlen
h["dist_to_end"] = h.slen - h[["sstart", "send"]].max(axis=1)        # bases beyond the hit's 3' edge
ok = h[(h.send > h.sstart) & (h.qcov >= oc["min_qcov"]) & (h.dist_to_end <= oc["max_dist_bp"])]
best = ok.sort_values("bitscore", ascending=False).drop_duplicates(["chrom", "end"]).set_index(["chrom", "end"])

rows, oriented = [], []
for c, s in seqs.items():
    r = {"chrom": c, "length": len(s)}
    for e in ("3p", "5p"):
        b = best.loc[(c, e)] if (c, e) in best.index else None
        r[f"{e}_found"] = int(b is not None)
        r[f"{e}_pident"] = round(float(b.pident), 1) if b is not None else ""
        r[f"{e}_qcov"] = round(float(b.qcov), 3) if b is not None else ""
        r[f"{e}_dist_to_end"] = int(b.dist_to_end) if b is not None else ""
        r[f"{e}_bitscore"] = float(b.bitscore) if b is not None else ""
    call = {(1, 0): "as_is", (0, 1): "flip", (1, 1): "both", (0, 0): "none"}[(r["3p_found"], r["5p_found"])]
    r["call"] = call
    r["flipped"] = int(bool(oc["apply"]) and call == "flip")
    rows.append(r)
    oriented.append((c, revcomp(s) if r["flipped"] else s))
df = pd.DataFrame(rows)
df.to_csv(f"{out}/orientation.tsv", sep="\t", index=False)
cnt = df.call.value_counts()
log("calls: " + ", ".join(f"{k}={cnt.get(k, 0)}" for k in ("as_is", "flip", "both", "none")))
for k in ("flip", "both", "none"):
    if cnt.get(k, 0):
        log(f"{k}: " + ", ".join(df[df.call == k].chrom))
if oc["apply"] or excl:
    fn = genome_fasta(cfg)
    write_fasta(fn, oriented)
    log(f"apply={oc['apply']}: reverse-complemented {int(df.flipped.sum())} contigs, excluded {len(excl)} -> "
        f"{fn} ({len(oriented)} contigs, used by Step01-07)")
else:
    log("apply=false: report only; Step01-07 use the input FASTA as given")
log.end()
