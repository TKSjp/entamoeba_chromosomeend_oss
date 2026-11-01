# entamoeba_chromosomeend_oss

Identify entamoeba chromosome end structure

A pipeline for microbial genomes that are thought to lack telomeres (telomerase-type repeats), such as *Entamoeba*. It examines how the sequences at chromosome 3′ ends are shared or similar between chromosomes, and lets you visualize the end structures and retrieve their sequences.

- Extracts 3′-end windows (1 / 2 / 5 / 20 / 50 kb) and measures similarity between and within chromosomes with k-mers and BLASTN
- Classifies the ends into types and aligns them with MAFFT
- Defines a **terminal element** shared by multiple chromosome ends and classifies its micro-structure (conserved modules and variable regions)
- Summarizes the results in a **single-file interactive HTML viewer** and HTML readme reports

## Sample data and results

The repository includes the input and the full results for two genomes.

| Genome | Input | Analyzed | Ends with the terminal element | Report | Viewer |
|---|---|---|---|---|---|
| Eh_30rc (*E. histolytica*) | `input/Eh_30rc.fna` | 36 sequences (Chr37, a plasmid, excluded) | 23 | `readme_Eh_30rc.html` | `result/Eh_30rc/08_viewer/chrom_end_viewer.html` |
| NA442 (*E. dispar*) | `input/NA442_chr35p1_masked_rc_capital.smask.fna` | 35 sequences (Chr36, a plasmid, excluded) | 12 | `readme_NA442.html` | `result/NA442/08_viewer/chrom_end_viewer.html` |

GitHub shows HTML files as source code, so clone the repository (or download it as a ZIP) and open the files directly in a browser. The viewer is a single self-contained file that needs no external libraries or network connection.

- `readme.html`: index of all genomes
- `readme_<genome>.html`: summary of results; methods, parameters and output files for each step; how to use the viewer; caveats

The HTML reports and the viewer interface are in Japanese.

## Setup

Create the environment with [conda](https://docs.conda.io/) (miniconda, miniforge, etc.). The environment name must match `conda_env` in the config file (default: `claude_chr_end`).

```bash
conda create -n claude_chr_end -c conda-forge -c bioconda \
  python=3.11 biopython numpy pandas scipy pyyaml blast mafft trf trnascan-se
```

Tested with Python 3.11.16, BLAST+ 2.17.0, MAFFT 7.526, TRF 4.10.0rc2, tRNAscan-SE 2.0.13, and numpy 2.4.6 / pandas 3.0.6 / scipy 1.17.1 on macOS (Apple M4).

## Usage

Run everything from the repository root. `run_all.sh` activates the conda environment named in the config file.

```bash
bash script/run_all.sh                     # list the configured genomes
bash script/run_all.sh NA442               # run all steps (Step00-09)
bash script/run_all.sh NA442 07 08 09      # run only the selected steps
CHR_END_GENOME=NA442 python script/02_kmer_sharing.py   # run a single step (with the environment already activated)
```

A run updates:

- `result/<genome>/`: output of each step (TSV / FASTA / JSON)
- `result/<genome>/08_viewer/chrom_end_viewer.html`: the viewer
- `readme_<genome>.html`, `readme.html`: the reports
- `log.txt`: start and end of each step, parameters, commands and summary statistics (shared by all genomes and appended to; not tracked by git)

Approximate run time (Apple M4, 8 threads): the MAFFT alignment in Step04 (50 kb windows, all chromosomes) takes the longest; every other step finishes within about a minute. MAFFT alignments are reused when their input has not changed.

### Analyzing a new genome

1. Put the genome FASTA in `input/`.
2. Copy an existing config and edit it:

   ```bash
   cp script/config/NA442.yaml script/config/NEW.yaml
   ```

   Keys you need to change:

   | Key | Description |
   |---|---|
   | `input_fasta` | Path to the input FASTA (relative to the repository root) |
   | `genome_label` | Genome name shown in the reports and log |
   | `result_dir` | Output directory (e.g. `result/NEW`) |
   | `exclude_contigs` | Sequences to leave out of the analysis, such as plasmids (`[]` if none) |
   | `orient` | Reference chromosome for orientation (`ref_chrom`) and whether to apply the flips (`apply`) |

3. Run it:

   ```bash
   bash script/run_all.sh NEW
   ```

The comments in the config file describe the other parameters (`windows_kb`, `kmer`, `blast`, `cluster`, `mafft`, `terminal_element`, `micro`, `viewer`, etc.).

### Input FASTA assumptions

- The pipeline analyzes the **3′ end** of each sequence; 5′ ends are not analyzed. Sequences are expected to be oriented so that their 3′ ends correspond.
- Step00 checks orientation by searching both ends of every contig for the 3′-terminal 300 bp of the reference chromosome. With `orient.apply: true`, contigs that carry this motif only at the 5′ end are reverse-complemented before the later steps; with `false`, orientation is only reported.

## Pipeline

| Step | Script | Description |
|---|---|---|
| 00 | `00_orient_contigs.py` | Remove excluded sequences and check contig orientation |
| 01 | `01_qc_extract_ends.py` | QC and extraction of 3′-end windows |
| 02 | `02_kmer_sharing.py` | Similarity between and within chromosomes (k-mers) and conserved end regions |
| 03 | `03_pairwise_similarity.py` | All-vs-all similarity (BLASTN) and copies elsewhere in the genome |
| 04 | `04_cluster_msa.py` | End-type classification and MAFFT multiple alignment |
| 05 | `05_annotate_elements.py` | Annotation of end regions (repeats, tRNAs, etc.) |
| 06 | `06_terminal_element.py` | Definition and mapping of the shared terminal element |
| 07 | `07_micro_structure.py` | Segmentation and classification of the micro-structure (conserved modules and variable regions) |
| 08 | `08_build_viewer.py` | Build the interactive HTML viewer (from `viewer_template.html`) |
| 09 | `09_build_readme.py` | Build the HTML reports |

Shared code is in `script/common.py`; per-genome parameters are in `script/config/<genome>.yaml`.

```
.
├── input/                  input genome FASTA files
├── result/<genome>/        output of each step (00_orient ... 08_viewer)
├── script/
│   ├── run_all.sh          runs the pipeline
│   ├── 00_...09_*.py       pipeline steps
│   ├── common.py
│   ├── viewer_template.html
│   └── config/<genome>.yaml
├── readme.html             index
└── readme_<genome>.html    results and methods for each genome
```

## Caveats

- End structures are inferred from the assembly alone. Missing ends, overhangs and chimeras can be assembly artifacts; confirm them with read coverage or long reads.
- Similarity is limited to homology that BLASTN can detect, so diverged sequences are underestimated. In strongly AT-rich genomes, low-identity hits (around 70%) may include chance similarity.
- k-mer similarity counts only near-identical copies (roughly 95% identity or higher).
- See the caveats section of each `readme_<genome>.html` for details.

## License

MIT License (see [LICENSE](LICENSE))
