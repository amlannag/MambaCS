#!/bin/bash
# Pull files OR folders from Bunya (QRISdata) into the local MambaCS Data directory.
# Uses SSH ControlMaster so you only authenticate (Okta MFA) once per run.

REMOTE_HOST="uqanag@bunya.rcc.uq.edu.au"
REMOTE_BASE="/scratch/user/uqanag"          # was .../prostate; your knee files live under knee
LOCAL_BASE="$HOME/Desktop/MambaCS/Data"
CONTROL_PATH="$HOME/.ssh/controlmasters/%r@%h:%p"

mkdir -p "$LOCAL_BASE" "$HOME/.ssh/controlmasters"

# Open one master connection (the only Okta/DUO prompt you'll get)
ssh -MNf -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" -o ControlPersist=10m "$REMOTE_HOST"

# Always close the master connection, even if something fails or you hit Ctrl+C
cleanup() { ssh -O exit -o ControlPath="$CONTROL_PATH" "$REMOTE_HOST" 2>/dev/null; }
trap cleanup EXIT

# Paths are relative to REMOTE_BASE. Each entry can be a file or a folder.
ITEMS=(
  fastmri_knee      # a file
  fastmri_prostate                   # a folder (copied INTO LOCAL_BASE as LOCAL_BASE/some_folder)
)

failed=()
for item in "${ITEMS[@]}"; do
    item="${item%/}"              # strip any trailing slash so folders land inside LOCAL_BASE
    echo "Syncing $item..."
    # -a recurses into folders; --partial keeps interrupted transfers so re-running resumes them.
    # No -z: .xz / .gz / .nii.gz files are already compressed.
    if ! rsync -avP --partial -e "ssh -o ControlPath=$CONTROL_PATH" \
            "$REMOTE_HOST:$REMOTE_BASE/$item" "$LOCAL_BASE/"; then
        failed+=("$item")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "Done. Synced to $LOCAL_BASE"