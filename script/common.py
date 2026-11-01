"""Shared helpers for the chromosome 3'-end pipeline (config, logging, FASTA I/O)."""
import datetime
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT, "script")


CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")


def available_genomes():
    return sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml"))


def load_config():
    """Load script/config/<genome>.yaml; the genome is given by the CHR_END_GENOME environment variable
    (set by run_all.sh)."""
    genome = os.environ.get("CHR_END_GENOME")
    if not genome:
        raise SystemExit(f"set CHR_END_GENOME (one of: {', '.join(available_genomes())}) or use run_all.sh <genome>")
    fn = os.path.join(CONFIG_DIR, f"{genome}.yaml")
    if not os.path.exists(fn):
        raise SystemExit(f"config not found: {fn} (available: {', '.join(available_genomes())})")
    with open(fn) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = ROOT
    cfg["_config_path"] = os.path.relpath(fn, ROOT)
    return cfg


def genome_fasta(cfg):
    """FASTA used by Step01-07: the genome written by Step00 (exclude_contigs removed, orientation corrected
    when orient.apply is true); the input FASTA itself when neither applies."""
    oc = cfg.get("orient", {})
    if cfg.get("exclude_contigs") or (oc.get("enabled") and oc.get("apply")):
        return path(cfg, cfg["result_dir"], "00_orient", f"{cfg['genome_label']}_analysis.fna")
    return path(cfg, cfg["input_fasta"])


def path(cfg, *parts):
    return os.path.join(cfg["_root"], *parts)


def step_dir(cfg, name):
    d = path(cfg, cfg["result_dir"], name)
    os.makedirs(d, exist_ok=True)
    return d


class Logger:
    """Append-only logger writing to log.txt (shared by all genomes; every line carries the genome label)."""

    def __init__(self, cfg, step):
        self.file = path(cfg, cfg["log_file"])
        self.step = step
        self.genome = cfg["genome_label"]

    def __call__(self, msg=""):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = str(msg).splitlines() or [""]
        with open(self.file, "a") as fh:
            for ln in lines:
                out = f"[{ts}] [{self.genome}] [{self.step}] {ln}"
                fh.write(out + "\n")
                print(out, flush=True)

    def start(self, title, cfg_keys=None, cfg=None):
        self("=" * 70)
        self(f"START {title}")
        self(f"cmd: {' '.join(sys.argv)}")
        if cfg_keys and cfg:
            for k in cfg_keys:
                self(f"param {k}: {cfg.get(k)}")

    def end(self):
        self("END")

    def run(self, cmd, **kw):
        """Run a shell command, log it, raise on failure."""
        self(f"$ {cmd}")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        if res.returncode != 0:
            self(f"ERROR (exit {res.returncode})\n{res.stderr[-3000:]}")
            raise SystemExit(res.returncode)
        return res


def read_fasta(fn):
    """Return an ordered dict name -> sequence (case preserved)."""
    seqs, name, buf = {}, None, []
    with open(fn) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def write_fasta(fn, records, width=80):
    with open(fn, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


_RC = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(s):
    return s.translate(_RC)[::-1]


def window_label(kb):
    return f"{kb}kb"


def tool_version(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().splitlines()
        return out[0] if out else "?"
    except Exception:
        return "?"
