# Environment of Record - RiskLabAI.py

This file pins the reproducibility baseline for `RiskLabAI.py` per GOVERNANCE
&sect;2 and the Engineering & Evidence Standards in `CLAUDE.md` ("Locked
environment: pinned Python/Julia versions, committed lockfile, recorded BLAS
backend + thread count. A drifting environment invalidates the results -
re-lock, don't paper over.").

All parity/replication results are valid only against the environment recorded
here. If any item below drifts, re-lock and regenerate the affected evidence.

## Operating system

| Field    | Value                          |
| -------- | ------------------------------ |
| OS       | Windows 10 Education (10.0.19045-SP0) |
| Platform | `Windows-10-10.0.19045-SP0`    |
| Arch     | AMD64 (x86_64, little-endian)  |
| Logical CPUs | 32                         |

## Python

| Field          | Value                                            |
| -------------- | ------------------------------------------------ |
| Version        | 3.12.3                                            |
| Build          | conda-forge, `main, Apr 15 2024, 18:20:11`       |
| Compiler       | MSC v.1938 64 bit (AMD64)                         |
| `requires-python` (pyproject) | `>=3.9`                           |

## Key package versions

Declared runtime dependencies (from `pyproject.toml` `[project].dependencies`),
pinned to the installed versions. Full transitive closure is in
`requirements-lock.txt`.

| Package        | Version  |
| -------------- | -------- |
| numpy          | 1.26.4   |
| scipy          | 1.11.4   |
| pandas         | 2.2.3    |
| scikit-learn   | 1.2.2    |
| statsmodels    | 0.14.5   |
| numba          | 0.59.1   |
| llvmlite       | 0.42.0   |
| joblib         | 1.3.2    |
| ta             | 0.11.0   |
| tqdm           | 4.66.4   |
| sympy          | 1.14.0   |

Optional extras (`pde` -> torch, `synth` -> quantecon, `plot` ->
matplotlib/seaborn/plotly, `dev` -> linters/test/etc.) are **not** part of this
runtime lock. The CI-pinned linters (`black==26.5.1`, `ruff==0.15.17`) live in
the `dev` extra of `pyproject.toml`.

## Numerical backend (BLAS / LAPACK)

NumPy is built against **Intel MKL**. Captured via
`python -c "import numpy; numpy.show_config()"`:

| Field                | Value      |
| -------------------- | ---------- |
| BLAS name            | `mkl-sdl`  |
| BLAS version         | 2023.1     |
| LAPACK               | bundled with MKL (numpy 1.26.4) |
| Detection method     | pkgconfig  |

SIMD baseline: SSE/SSE2/SSE3; found up to AVX2/FMA3/F16C (AVX-512 not used on
this host). Because the BLAS backend and its threading affect floating-point
reduction order, deterministic linear-algebra parity tolerances (covariance,
shrinkage, eigen/SVD) are stated against **this MKL build**; a different BLAS
(e.g. OpenBLAS) can shift results within tolerance.

## Thread counts

At the time of locking, all numerical-threading environment variables were
**unset**, so MKL/OpenBLAS use their default (all available cores). For
reproducible, deterministic numerics, pin them to `1` before running:

| Variable                  | Value at lock time |
| ------------------------- | ------------------ |
| `OMP_NUM_THREADS`         | unset (default)    |
| `OPENBLAS_NUM_THREADS`    | unset (default)    |
| `MKL_NUM_THREADS`         | unset (default)    |
| `NUMEXPR_NUM_THREADS`     | unset (default)    |
| `VECLIB_MAXIMUM_THREADS`  | unset (default)    |

Recommended for deterministic runs (PowerShell):

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
```

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Recreate this environment

1. Create and activate a clean Python 3.12 environment (conda or venv):

   ```bash
   conda create -n risklabai python=3.12.3
   conda activate risklabai
   ```

2. Install the pinned runtime dependencies from the lockfile:

   ```bash
   pip install -r requirements-lock.txt
   ```

3. Install the package itself (editable, no deps - they are already pinned):

   ```bash
   pip install -e . --no-deps
   ```

   For development/tests add the dev extra: `pip install -e ".[dev]"`.

4. Verify the numerical backend matches this record:

   ```bash
   python -c "import numpy; numpy.show_config()"
   ```

   Confirm the BLAS name is `mkl-sdl` and version `2023.1`. If it differs, the
   environment has drifted - re-lock and note the change before trusting any
   parity/replication numbers.

5. (Recommended) pin thread counts to `1` as shown above for deterministic
   numerics.

## Provenance

- Lockfile: `requirements-lock.txt` (filtered freeze; `pip-compile` was not
  available, so the transitive closure of the declared runtime deps was
  resolved from installed metadata and pinned).
- Generated: 2026-06-26.
