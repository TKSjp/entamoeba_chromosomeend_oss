"""Step01: per-chromosome QC and extraction of 3'-terminal windows.

Coordinates: `dist` = distance from the 3' end, last base = 1.
Outputs (result/01_qc/):
  chr_stats.tsv               length / GC / soft-mask fraction / terminal bases
  ends_3p_{W}kb.fna           3'-terminal W kb of every chromosome (case preserved)
  ends_3p_windows.tsv         window coordinates (1-based, genome) per chromosome
"""
from common import Logger, genome_fasta, load_config, path, read_fasta, step_dir, window_label, write_fasta

cfg = load_config()
log = Logger(cfg, "STEP01_qc_extract")
log.start("QC and 3'-end extraction", ["input_fasta", "windows_kb"], cfg)
out = step_dir(cfg, "01_qc")

seqs = read_fasta(genome_fasta(cfg))
log(f"loaded {len(seqs)} sequences, total {sum(map(len, seqs.values())):,} bp")


def gc(s):
    s = s.upper()
    acgt = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / acgt if acgt else 0.0


def lower_frac(s):
    return sum(c.islower() for c in s) / len(s) if s else 0.0


with open(f"{out}/chr_stats.tsv", "w") as fh:
    fh.write("chrom\tlength\tGC\tN\tsoftmask_frac\tsoftmask_frac_3p1kb\tGC_3p1kb\tlast30\n")
    for n, s in seqs.items():
        t = s[-1000:]
        fh.write(f"{n}\t{len(s)}\t{gc(s):.4f}\t{s.upper().count('N')}\t{lower_frac(s):.4f}"
                 f"\t{lower_frac(t):.4f}\t{gc(t):.4f}\t{s[-30:]}\n")

with open(f"{out}/ends_3p_windows.tsv", "w") as fh:
    fh.write("window\tchrom\tchrom_length\tstart\tend\twin_length\ttruncated\n")
    for kb in cfg["windows_kb"]:
        W = kb * 1000
        recs, ntrunc = [], 0
        for n, s in seqs.items():
            st = max(0, len(s) - W)
            sub = s[st:]
            trunc = len(sub) < W
            ntrunc += trunc
            recs.append((n, sub))
            fh.write(f"{window_label(kb)}\t{n}\t{len(s)}\t{st + 1}\t{len(s)}\t{len(sub)}\t{int(trunc)}\n")
        write_fasta(f"{out}/ends_3p_{window_label(kb)}.fna", recs)
        log(f"window {window_label(kb)}: wrote {len(recs)} seqs ({ntrunc} shorter than window -> whole chromosome)")

log("outputs: chr_stats.tsv, ends_3p_windows.tsv, ends_3p_{W}kb.fna")
log.end()
