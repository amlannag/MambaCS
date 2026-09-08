# Project verification

- Run the APT k-space demonstration tests with `python -m pytest tests/test_apt_kspace.py tests/test_fastmri_inference.py -q` from the repository root.
- Execute the visualization notebook with `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks/fastmri_apt.ipynb`. It uses local `data/singlecoil_val/*.h5` files and writes figures, counts, and settings to `notebooks/apt_outputs/`.
- APT entropy/selection parity tests use the local reference at `apt-main/src/models/entropy_utils.py`; they skip that comparison if the reference checkout is absent.
- Existing `.gitignore` rules exclude `notebooks/`, `tests/`, and `apt-main/`. New notebook/test files will not appear in ordinary `git status`; do not change ignore rules or stage ignored files without being asked.
