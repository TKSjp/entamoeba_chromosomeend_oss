#!/usr/bin/env bash
# Run the chromosome 3'-end pipeline for one genome.
#   bash script/run_all.sh <genome>              # all steps (genome = script/config/<genome>.yaml)
#   bash script/run_all.sh <genome> 07 08 09     # selected steps
#   bash script/run_all.sh                       # list available genomes
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "usage: bash script/run_all.sh <genome> [steps...]"
  echo "genomes: $(cd "$SCRIPT_DIR/config" && ls *.yaml | sed 's/\.yaml$//' | tr '\n' ' ')"
  exit 1
fi
GENOME="$1"; shift
CONFIG="$SCRIPT_DIR/config/${GENOME}.yaml"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG"; exit 1; }
export CHR_END_GENOME="$GENOME"
ENV_NAME="$(awk '/^conda_env:/{print $2}' "$CONFIG")"
RESULT_DIR="$(awk '/^result_dir:/{print $2}' "$CONFIG")"

# make the conda environment's tools (blastn, mafft, trf, tRNAscan-SE) and python available
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  CONDA_BIN="$(command -v conda || echo "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")"
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$ENV_NAME"
fi

STEPS=("00_orient_contigs" "01_qc_extract_ends" "02_kmer_sharing" "03_pairwise_similarity" "04_cluster_msa"
       "05_annotate_elements" "06_terminal_element" "07_micro_structure" "08_build_viewer" "09_build_readme")
SELECT=("$@")

cd "$SCRIPT_DIR"
for s in "${STEPS[@]}"; do
  num="${s%%_*}"
  if [[ ${#SELECT[@]} -gt 0 && ! " ${SELECT[*]} " =~ " ${num} " ]]; then continue; fi
  echo ">>> [${GENOME}] Step ${num}: ${s}"
  python "${s}.py"
done
echo ">>> done. viewer: ${RESULT_DIR}/08_viewer/chrom_end_viewer.html  readme: readme_${GENOME}.html (index: readme.html)  log: log.txt"
