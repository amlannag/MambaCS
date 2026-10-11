#!/bin/bash
# copy_t2.sh
# Split the T2 files from QRISdata into N chunks and copy each chunk to scratch
# using a Slurm job array (one array task per chunk).
#
# File lists live in $WORK_DIR on scratch, NOT in QRISdata (which is read-only for you).
#
# Usage:
#   bash copy_t2.sh submit      # build the chunk lists and submit one array per split
#   bash copy_t2.sh verify      # compare file counts (source T2 files vs. copied files)
#
# Optional overrides (env vars), e.g.:
#   N_NODES=3 ACCOUNT=a_xxx PARTITION=general bash copy_t2.sh submit

# ----------------------------- CONFIG ---------------------------------------
SRC_ROOT="${SRC_ROOT:-/QRISdata/Q9618/brain}"
DEST_ROOT="${DEST_ROOT:-/scratch/user/uqanag/fastmri_brain}"
WORK_DIR="${WORK_DIR:-/scratch/user/uqanag/copy_t2_work}"

# split name -> source folder name (under SRC_ROOT) and destination folder (under DEST_ROOT)
# format: "dest_name:src_folder"
SPLITS=("val:multicoil_val" "train:multicoil_train")

PATTERN="${PATTERN:-*T2*.h5}"     # change to '*AXT2*.h5' if you want to be stricter
N_NODES="${N_NODES:-5}"           # number of chunks == number of array tasks

ACCOUNT="${ACCOUNT:-}"            # leave empty to use your default account
PARTITION="${PARTITION:-}"        # leave empty to use the default partition
TIME="${TIME:-04:00:00}"
CPUS="${CPUS:-2}"
MEM="${MEM:-4G}"
# -----------------------------------------------------------------------------

SCRIPT="$(readlink -f "$0")"
MODE="${1:-submit}"

list_file()  { echo "$WORK_DIR/$1/t2_files.txt"; }
chunk_file() { printf "%s/%s/part_%02d" "$WORK_DIR" "$1" "$2"; }

# Look up the source folder for a split name
src_dir_for() {
  local entry
  for entry in "${SPLITS[@]}"; do
    if [[ "${entry%%:*}" == "$1" ]]; then
      echo "$SRC_ROOT/${entry##*:}"
      return
    fi
  done
}

# ------------------------------ SUBMIT ---------------------------------------
if [[ "$MODE" == "submit" ]]; then
  mkdir -p "$WORK_DIR/logs" "$DEST_ROOT"

  SBATCH_EXTRA=()
  [[ -n "$ACCOUNT"   ]] && SBATCH_EXTRA+=(--account="$ACCOUNT")
  [[ -n "$PARTITION" ]] && SBATCH_EXTRA+=(--partition="$PARTITION")

  for entry in "${SPLITS[@]}"; do
    split="${entry%%:*}"
    src="$SRC_ROOT/${entry##*:}"

    if [[ ! -d "$src" ]]; then
      echo "ERROR: source folder not found: $src" >&2
      exit 1
    fi

    mkdir -p "$WORK_DIR/$split" "$DEST_ROOT/$split"
    rm -f "$WORK_DIR/$split"/part_*

    # Build the list of T2 files (names only) and split it into N chunks by line
    find "$src" -maxdepth 1 -type f -name "$PATTERN" -printf '%f\n' | sort > "$(list_file "$split")"
    total=$(wc -l < "$(list_file "$split")")
    echo "[$split] found $total files matching $PATTERN in $src"

    if [[ "$total" -eq 0 ]]; then
      echo "[$split] nothing to copy, skipping"
      continue
    fi

    split -n "l/$N_NODES" -d -a 2 "$(list_file "$split")" "$WORK_DIR/$split/part_"

    # One job array per split, one task per chunk
    sbatch \
      --job-name="copy_${split}" \
      --array="0-$((N_NODES - 1))" \
      --nodes=1 --ntasks=1 \
      --cpus-per-task="$CPUS" \
      --mem="$MEM" \
      --time="$TIME" \
      --output="$WORK_DIR/logs/%x_%A_%a.out" \
      --export=ALL,SPLIT="$split" \
      "${SBATCH_EXTRA[@]}" \
      --wrap="bash $SCRIPT worker"
  done

  echo
  echo "Submitted. Check with:  squeue -u $USER"
  echo "Logs:                   $WORK_DIR/logs/"
  echo "When finished, run:     bash $SCRIPT verify"
  exit 0
fi

# ------------------------------ WORKER ---------------------------------------
if [[ "$MODE" == "worker" ]]; then
  : "${SPLIT:?SPLIT not set}"
  : "${SLURM_ARRAY_TASK_ID:?must be run as a Slurm array task}"

  src="$(src_dir_for "$SPLIT")"
  chunk="$(chunk_file "$SPLIT" "$SLURM_ARRAY_TASK_ID")"
  dest="$DEST_ROOT/$SPLIT"

  echo "Host:   $(hostname)"
  echo "Split:  $SPLIT   Task: $SLURM_ARRAY_TASK_ID"
  echo "Chunk:  $chunk ($(wc -l < "$chunk") files)"
  echo "From:   $src/"
  echo "To:     $dest/"
  echo "Start:  $(date)"

  rsync -ah --info=progress2 --files-from="$chunk" "$src/" "$dest/"
  rc=$?

  echo "End:    $(date)   rsync exit code: $rc"
  exit $rc
fi

# ------------------------------ VERIFY ---------------------------------------
if [[ "$MODE" == "verify" ]]; then
  for entry in "${SPLITS[@]}"; do
    split="${entry%%:*}"
    src="$SRC_ROOT/${entry##*:}"
    dest="$DEST_ROOT/$split"

    n_src=$(find "$src" -maxdepth 1 -type f -name "$PATTERN" | wc -l)
    n_dst=$(find "$dest" -maxdepth 1 -type f -name "$PATTERN" 2>/dev/null | wc -l)

    status="OK"
    [[ "$n_src" -ne "$n_dst" ]] && status="MISMATCH"
    echo "[$split] source=$n_src  copied=$n_dst  -> $status"

    if [[ "$status" == "MISMATCH" ]]; then
      echo "  missing files:"
      comm -23 \
        <(find "$src"  -maxdepth 1 -type f -name "$PATTERN" -printf '%f\n' | sort) \
        <(find "$dest" -maxdepth 1 -type f -name "$PATTERN" -printf '%f\n' | sort) | head -20
    fi
  done
  exit 0
fi

echo "Usage: bash $0 [submit|verify]" >&2
exit 1