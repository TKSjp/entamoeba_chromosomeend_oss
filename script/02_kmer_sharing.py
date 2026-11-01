"""Step02: k-mer sharing profile along the 3' ends and terminal conserved-region boundary.

For every base in the largest 3'-terminal window of each chromosome:
  inter_chroms    INTERchromosomal similarity: number of OTHER chromosomes that contain the k-mer anywhere
  intra_copies    INTRAchromosomal similarity: copies of the k-mer elsewhere on the SAME chromosome
                  (whole chromosome, including repeats inside the terminal window)
  end_share       number of OTHER chromosomes whose 3'-terminal window contains the k-mer
                  (subset of inter_chroms; used for the conserved terminal region below)
  genome_copies   genome-wide occurrence count of that k-mer (canonical, both strands)
  interior_copies occurrences outside every chromosome's terminal window
(per-base value = max over the k-mers overlapping the base). Low-complexity k-mers (fewer than
`low_complexity_min_trimers` distinct 3-mers, e.g. poly-T, dinucleotide repeats) are not counted for
inter_chroms / intra_copies (value 0; flagged in the `lowcomp` array).

Outputs (result/02_kmer/):
  kmer_profile.npz              per-chromosome uint arrays, index 0 = last base (dist 1)
  kmer_profile.tsv.gz           same in long format (chrom, dist, base, ...)
  kmer_share_binned.tsv         mean/max end_share per bin
  terminal_boundary.tsv         conserved-region length from the 3' end per chromosome
  share_by_distance.tsv         number of chromosomes conserved at each distance bin
"""
import gzip

import numpy as np

from common import Logger, genome_fasta, load_config, path, read_fasta, step_dir

cfg = load_config()
kc = cfg["kmer"]
K, MIN_O, MAX_GAP, BIN = kc["k"], kc["min_other_chrs"], kc["max_gap"], kc["bin_bp"]
MAX_OFF, MIN_RUN = kc["max_offset"], kc["min_run"]
W = max(cfg["windows_kb"]) * 1000
log = Logger(cfg, "STEP02_kmer")
log.start("k-mer sharing profile", ["kmer", "windows_kb"], cfg)
log(f"profile window = {W:,} bp (largest window), k = {K}")
out = step_dir(cfg, "02_kmer")

seqs = read_fasta(genome_fasta(cfg))
names = list(seqs)
LUT = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    LUT[ord(b)] = i
    LUT[ord(b.lower())] = i


def kmers(seq):
    """canonical 2-bit k-mer codes (uint64) for each start; invalid (non-ACGT) -> max uint64."""
    c = LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    n = len(c) - K + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    c64 = c.astype(np.uint64)
    fw = np.zeros(n, dtype=np.uint64)
    rc = np.zeros(n, dtype=np.uint64)
    bad = np.zeros(n, dtype=bool)
    for j in range(K):
        sl = c64[j:j + n]
        fw |= (sl & np.uint64(3)) << np.uint64(2 * (K - 1 - j))
        rc |= ((np.uint64(3) - (sl & np.uint64(3))) << np.uint64(2 * j))
        bad |= c[j:j + n] == 255
    can = np.minimum(fw, rc)
    can[bad] = np.iinfo(np.uint64).max
    return can


def per_base_max(vals, L):
    """value at k-mer start -> per-base max over overlapping k-mers."""
    res = np.zeros(L, dtype=vals.dtype)
    for s in range(K):
        seg = res[s:s + len(vals)]
        np.maximum(seg, vals[:len(seg)], out=seg)
    return res


# --- terminal-window k-mers
INVALID = np.iinfo(np.uint64).max
end_km, end_off = {}, {}
for n in names:
    s = seqs[n]
    st = max(0, len(s) - W)
    end_off[n] = st
    end_km[n] = kmers(s[st:])
pairs = np.concatenate([np.unique(v[v != INVALID]) for v in end_km.values()])
uk, nchr = np.unique(pairs, return_counts=True)
log(f"unique terminal k-mers: {len(uk):,}; shared by >=2 chromosomes: {(nchr >= 2).sum():,}; max chromosomes: {nchr.max()}")


def distinct_trimers(codes):
    """number of distinct 3-mers in each k-mer (2-bit codes); low values = low complexity."""
    tri = np.stack([((codes >> np.uint64(2 * (K - 3 - j))) & np.uint64(63)).astype(np.uint8) for j in range(K - 2)], axis=1)
    tri.sort(axis=1)
    return (np.diff(tri, axis=1) != 0).sum(axis=1) + 1


lowc = distinct_trimers(uk) < kc["low_complexity_min_trimers"]
log(f"low-complexity terminal k-mers (< {kc['low_complexity_min_trimers']} distinct 3-mers): {lowc.sum():,} ({lowc.mean():.1%}); "
    "excluded from inter/intra-chromosomal similarity")

# --- genome-wide counts restricted to terminal k-mers
gcount = np.zeros(len(uk), dtype=np.int64)
icount = np.zeros(len(uk), dtype=np.int64)
NC = len(names)
pair_keys = []  # k-mer index * NC + chromosome index, one entry per occurrence
for ci, n in enumerate(names):
    s = seqs[n]
    km = kmers(s)
    keep = np.isin(km, uk)
    idx = np.searchsorted(uk, km[keep])
    np.add.at(gcount, idx, 1)
    pos = np.nonzero(keep)[0]
    interior = pos + K <= end_off[n]  # k-mer entirely outside the terminal window
    np.add.at(icount, idx[interior], 1)
    pair_keys.append(idx.astype(np.int64) * NC + ci)
log(f"k-mers with interior (non-terminal) copies: {(icount > 0).sum():,}")
# occurrences per (k-mer, chromosome): number of chromosomes carrying each k-mer, and copies per chromosome
pk, pcnt = np.unique(np.concatenate(pair_keys), return_counts=True)
n_chrom_any = np.bincount(pk // NC, minlength=len(uk))


def own_copies(idx, ci):
    """occurrences of k-mers `idx` on chromosome ci (whole chromosome)."""
    key = idx.astype(np.int64) * NC + ci
    j = np.searchsorted(pk, key)
    j[j >= len(pk)] = 0
    return np.where(pk[j] == key, pcnt[j], 0)

# --- per-base profiles (index 0 = last base)
prof = {}
for ci, n in enumerate(names):
    km = end_km[n]
    L = len(seqs[n]) - end_off[n]
    ok = km != INVALID
    idx = np.searchsorted(uk, km)
    idx[~ok] = 0
    share = np.where(ok, nchr[idx] - 1, 0).astype(np.int32)
    gc_ = np.where(ok, gcount[idx], 0).astype(np.int32)
    ic_ = np.where(ok, icount[idx], 0).astype(np.int32)
    cnt = ok & ~lowc[idx]
    inter = np.where(cnt, n_chrom_any[idx] - 1, 0).astype(np.int32)          # other chromosomes, anywhere
    intra = np.where(cnt, own_copies(idx, ci) - 1, 0).astype(np.int32)       # other copies on the same chromosome
    lc = np.where(ok, lowc[idx], False).astype(np.int32)
    prof[n] = {
        "inter_chroms": per_base_max(inter, L)[::-1].astype(np.uint8),
        "intra_copies": np.minimum(per_base_max(intra, L), 65535)[::-1].astype(np.uint16),
        "lowcomp": per_base_max(lc, L)[::-1].astype(np.uint8),
        "end_share": per_base_max(share, L)[::-1].astype(np.uint8),
        "genome_copies": np.minimum(per_base_max(gc_, L), 65535)[::-1].astype(np.uint16),
        "interior_copies": np.minimum(per_base_max(ic_, L), 65535)[::-1].astype(np.uint16),
    }

np.savez_compressed(f"{out}/kmer_profile.npz",
                    **{f"{n}__{k}": v for n in names for k, v in prof[n].items()})
with gzip.open(f"{out}/kmer_profile.tsv.gz", "wt") as fh:
    fh.write("chrom\tdist\tbase\tinter_chroms\tintra_copies\tlowcomp\tend_share\tgenome_copies\tinterior_copies\n")
    for n in names:
        tail = seqs[n][end_off[n]:][::-1]
        p = prof[n]
        for i in range(len(tail)):
            fh.write(f"{n}\t{i + 1}\t{tail[i]}\t{p['inter_chroms'][i]}\t{p['intra_copies'][i]}\t{p['lowcomp'][i]}\t"
                     f"{p['end_share'][i]}\t{p['genome_copies'][i]}\t{p['interior_copies'][i]}\n")


def boundary(share):
    """(offset, end): conserved region = dist offset+1 .. end.
    offset = distance to the first conserved run (>= MIN_RUN bp) starting within MAX_OFF of the 3' end."""
    cons = share >= MIN_O
    start, run = None, 0
    for i in range(min(len(cons), MAX_OFF + MIN_RUN)):
        run = run + 1 if cons[i] else 0
        if run >= MIN_RUN:
            start = i - run + 1
            break
    if start is None:
        return 0, 0
    last, gap = start, 0
    for i in range(start, len(cons)):
        if cons[i]:
            last, gap = i + 1, 0
        else:
            gap += 1
            if gap > MAX_GAP:
                break
    return start, last


with open(f"{out}/terminal_boundary.tsv", "w") as fh:
    fh.write("chrom\twindow_len\tterminal_offset\tconserved_end\tconserved_len\tconserved_frac_in_region\tmean_share_in_region\t"
             "max_share\tshare_at_last_base\tinterior_copy_frac_in_region\n")
    for n in names:
        sh = prof[n]["end_share"]
        o, e = boundary(sh)
        b = e - o
        reg = sh[o:e]
        ic = prof[n]["interior_copies"][o:e]
        fh.write(f"{n}\t{len(sh)}\t{o}\t{e}\t{b}\t{(reg >= MIN_O).mean() if b else 0:.3f}\t{reg.mean() if b else 0:.2f}\t"
                 f"{sh.max()}\t{sh[0]}\t{(ic > 0).mean() if b else 0:.3f}\n")
        log(f"{n}: conserved terminal region dist {o + 1 if b else 0:,}-{e:,} ({b:,} bp, offset {o} bp; max share {sh.max()})")

with open(f"{out}/kmer_share_binned.tsv", "w") as fh:
    fh.write("chrom\tbin_from_dist\tbin_to_dist\tmean_inter_chroms\tfrac_inter\tmean_intra_copies\tfrac_intra\t"
             "mean_share\tmax_share\tmean_interior_copies\n")
    for n in names:
        p = prof[n]
        sh, ic, it, ia = p["end_share"], p["interior_copies"], p["inter_chroms"], p["intra_copies"]
        for b0 in range(0, len(sh), BIN):
            s_, i_, t_, a_ = sh[b0:b0 + BIN], ic[b0:b0 + BIN], it[b0:b0 + BIN], ia[b0:b0 + BIN]
            fh.write(f"{n}\t{b0 + 1}\t{b0 + len(s_)}\t{t_.mean():.2f}\t{(t_ > 0).mean():.3f}\t{a_.mean():.2f}\t{(a_ > 0).mean():.3f}\t"
                     f"{s_.mean():.2f}\t{s_.max()}\t{i_.mean():.2f}\n")

with open(f"{out}/share_by_distance.tsv", "w") as fh:
    fh.write("bin_from_dist\tbin_to_dist\tn_chrom_with_seq\tn_chrom_conserved\n")
    for b0 in range(0, W, BIN):
        have = [n for n in names if len(prof[n]["end_share"]) > b0]
        cons = sum((prof[n]["end_share"][b0:b0 + BIN] >= MIN_O).mean() >= 0.5 for n in have)
        fh.write(f"{b0 + 1}\t{b0 + BIN}\t{len(have)}\t{cons}\n")

# per-chromosome summary of inter/intra-chromosomal similarity in the conserved terminal region and the whole window
with open(f"{out}/inter_intra_summary.tsv", "w") as fh:
    fh.write("chrom\twindow_len\tfrac_inter_3p1kb\tfrac_intra_3p1kb\tfrac_inter_window\tfrac_intra_window\tmax_inter_chroms\n")
    for n in names:
        p = prof[n]
        it, ia = p["inter_chroms"], p["intra_copies"]
        fh.write(f"{n}\t{len(it)}\t{(it[:1000] > 0).mean():.3f}\t{(ia[:1000] > 0).mean():.3f}\t{(it > 0).mean():.3f}\t{(ia > 0).mean():.3f}\t{it.max()}\n")
log("inter/intra-chromosomal similarity (fraction of bases with >=1 copy, terminal 1 kb): "
    + ", ".join(f"{n} {(prof[n]['inter_chroms'][:1000] > 0).mean():.2f}/{(prof[n]['intra_copies'][:1000] > 0).mean():.2f}" for n in names))
log.end()
