#!/bin/bash
# Pull MRI_NYU k-space file(s) from Bunya HPC to the local MambaCS Data directory.
# Uses SSH ControlMaster so you only authenticate (Okta MFA) once per run.

REMOTE_HOST="uqanag@bunya.rcc.uq.edu.au"
# Source folder on Bunya (pulled directly from QRISdata)
REMOTE_BASE="/QRISdata/Q9618/knee"
LOCAL_BASE="$HOME/Desktop/MambaCS/Data/knee_multicoil"
CONTROL_PATH="$HOME/.ssh/controlmasters/%r@%h:%p"

mkdir -p "$LOCAL_BASE" "$HOME/.ssh/controlmasters"

# Open one master connection (this is the only Okta/DUO prompt you'll get)
ssh -MNf -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" -o ControlPersist=10m "$REMOTE_HOST"

# Edit this list to change which files get pulled
FILES=(
  knee_multicoil_test.tar.xz
)

for f in "${FILES[@]}"; do
    echo "Syncing $f..."
    # --partial keeps interrupted transfers so re-running resumes them.
    # No -z: the .nii.gz files are already compressed.
    rsync -avP --partial -e "ssh -o ControlPath=$CONTROL_PATH" "$REMOTE_HOST:$REMOTE_BASE/$f" "$LOCAL_BASE/"
done

# Close the master connection
ssh -O exit -o ControlPath="$CONTROL_PATH" "$REMOTE_HOST"

echo "Done. Synced to $LOCAL_BASE"