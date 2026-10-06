#!/bin/bash
# Download dataset directories from Bunya HPC into the local MambaCS data directory.
# Uses SSH ControlMaster so you only authenticate (Okta MFA) once per run.
#
# Usage:
#   ./"Download Bunya Data.sh"                                  # pull every directory in DATASETS below
#   ./"Download Bunya Data.sh" /scratch/user/uqanag/fastmri/singlecoil_train
#   ./"Download Bunya Data.sh" --max-files 20 /scratch/user/uqanag/fastmri/singlecoil_train
#   ./"Download Bunya Data.sh" --dest ~/other/dir /scratch/user/uqanag/oasis
#   ./"Download Bunya Data.sh" --list /scratch/user/uqanag        # browse the remote, download nothing
#   ./"Download Bunya Data.sh" --dry-run                          # show what rsync would transfer
#
# Each remote directory lands in LOCAL_BASE/<basename of remote dir>. Relative remote paths are
# resolved against REMOTE_BASE. Completed files are skipped and partial transfers resume.
set -euo pipefail

REMOTE_HOST="${BUNYA_HOST:-uqanag@bunya.rcc.uq.edu.au}"
REMOTE_BASE="/scratch/user/uqanag"
LOCAL_BASE="$HOME/Desktop/MambaCS/data"
CONTROL_PATH="$HOME/.ssh/controlmasters/%r@%h:%p"

# Edit this list to change which datasets get pulled when no directories are given on the command line
DATASETS=(
  fastmri/singlecoil_val
)

MAX_FILES=""
DRY_RUN=""
LIST_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-files) MAX_FILES="$2"; shift 2 ;;
        --dest)      LOCAL_BASE="$2"; shift 2 ;;
        --dry-run)   DRY_RUN="--dry-run"; shift ;;
        --list)      LIST_DIR="${2:-$REMOTE_BASE}"; [[ $# -gt 1 ]] && shift 2 || shift ;;
        -h|--help)   sed -n '2,15p' "$0"; exit 0 ;;
        *)           ARGS+=("$1"); shift ;;
    esac
done
[[ ${#ARGS[@]} -gt 0 ]] && DATASETS=("${ARGS[@]}")

command -v rsync >/dev/null 2>&1 || { echo "rsync is required but was not found." >&2; exit 1; }
mkdir -p "$LOCAL_BASE" "$HOME/.ssh/controlmasters"

SSH_OPTS=(-o ControlPath="$CONTROL_PATH")
# Open one master connection (this is the only Okta/DUO prompt you'll get)
ssh -MNf -o ControlMaster=yes "${SSH_OPTS[@]}" -o ControlPersist=10m "$REMOTE_HOST"
trap 'ssh -O exit "${SSH_OPTS[@]}" "$REMOTE_HOST" 2>/dev/null || true' EXIT

resolve_remote() {
    case "$1" in /*|~*) echo "$1" ;; *) echo "$REMOTE_BASE/$1" ;; esac
}

if [[ -n "$LIST_DIR" ]]; then
    LIST_DIR=$(resolve_remote "$LIST_DIR")
    echo "Contents of $REMOTE_HOST:$LIST_DIR"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$LIST_DIR' && for d in */ .; do [ -d \"\$d\" ] || continue; printf '%8s  %6s files  %s\n' \"\$(du -sh \"\$d\" 2>/dev/null | cut -f1)\" \"\$(find \"\$d\" -maxdepth 1 -type f | wc -l | tr -d ' ')\" \"\$d\"; done"
    exit 0
fi

for ds in "${DATASETS[@]}"; do
    REMOTE_DIR=$(resolve_remote "$ds")
    LOCAL_DIR="$LOCAL_BASE/$(basename "$REMOTE_DIR")"
    mkdir -p "$LOCAL_DIR"
    echo "==> $REMOTE_HOST:$REMOTE_DIR/  ->  $LOCAL_DIR/"

    RSYNC_OPTS=(--archive --human-readable --partial --progress $DRY_RUN -e "ssh ${SSH_OPTS[*]}")
    if [[ -n "$MAX_FILES" ]]; then
        # Only the first N files (sorted by name) at the top level of the remote directory
        FILES=$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && find . -maxdepth 1 -type f | sort | head -n $MAX_FILES | sed 's|^\./||'")
        [[ -z "$FILES" ]] && { echo "    no files found in $REMOTE_DIR" >&2; continue; }
        echo "    limiting to the first $MAX_FILES files"
        rsync "${RSYNC_OPTS[@]}" --files-from=<(printf '%s\n' "$FILES") "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    else
        rsync "${RSYNC_OPTS[@]}" "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    fi
    [[ -z "$DRY_RUN" ]] && du -sh "$LOCAL_DIR"
done

echo "Done. Datasets are in $LOCAL_BASE"
