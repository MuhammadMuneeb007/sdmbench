#!/usr/bin/env bash
# Create a complete sdmbench environment, using the fastest solver available.
#
#   bash scripts/create_env.sh              # CPU
#   bash scripts/create_env.sh --gpu        # CUDA
#   bash scripts/create_env.sh --name myenv
#   bash scripts/create_env.sh --dry-run    # print the plan, change nothing
#
# What it does, in order:
#   1. picks micromamba > mamba > conda (micromamba is typically 5-10x faster
#      than classic conda on a solve this size)
#   2. creates the environment from environment.yml / environment-gpu.yml
#   3. installs `disdat`, the one R package not on conda-forge
#   4. suggests a cache location if $HOME looks quota-limited
#   5. verifies with `sdmbench env check`
#
# Nothing is created or modified until the plan has been printed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=0
DRY_RUN=0
ENV_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)      GPU=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --name)     ENV_NAME="$2"; shift 2 ;;
    --name=*)   ENV_NAME="${1#*=}"; shift ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ $GPU -eq 1 ]]; then
  ENV_FILE="$REPO_ROOT/environment-gpu.yml"
  DEFAULT_NAME="sdmbench-gpu"
else
  ENV_FILE="$REPO_ROOT/environment.yml"
  DEFAULT_NAME="sdmbench"
fi
ENV_NAME="${ENV_NAME:-$DEFAULT_NAME}"

# --- 1. pick the fastest solver ---------------------------------------------
# micromamba is a standalone C++ binary with no conda dependency and the
# fastest solve. mamba is the same solver inside a conda install. Classic conda
# is the slow fallback, so we switch it to the libmamba solver where possible.
TOOL=""
for candidate in micromamba mamba conda; do
  if command -v "$candidate" >/dev/null 2>&1; then TOOL="$candidate"; break; fi
done

if [[ -z "$TOOL" ]]; then
  cat <<'EOF'
error: no conda-family tool found (looked for micromamba, mamba, conda).

Install micromamba -- one binary, no bootstrap, and the fastest option:

    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

or use the pip-only route instead (slower, and you must supply GDAL and R
yourself):

    pip install -e .
    Rscript scripts/setup_r_packages.R
EOF
  exit 1
fi

echo "sdmbench environment setup"
echo "=========================="
echo "  solver      $TOOL ($(command -v "$TOOL"))"
echo "  env file    $ENV_FILE"
echo "  env name    $ENV_NAME"
echo "  mode        $([[ $GPU -eq 1 ]] && echo 'GPU (CUDA)' || echo 'CPU')"
if [[ $GPU -eq 1 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  driver      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  else
    echo "  driver      nvidia-smi not found -- CUDA packages will install but not run"
  fi
fi
echo

if [[ "$TOOL" == "conda" ]]; then
  echo "note: classic conda is slow on a solve this size. Either install mamba"
  echo "      (conda install -n base -c conda-forge mamba) or let this script"
  echo "      switch conda to the libmamba solver, which it will try below."
  echo
fi

# --- 2. build the command ----------------------------------------------------
CREATE_CMD=("$TOOL" env create -f "$ENV_FILE" -n "$ENV_NAME" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
if [[ "$TOOL" == "micromamba" ]]; then
  CREATE_CMD+=(-y)
fi

echo "PLAN"
echo "  1. $(printf '%q ' "${CREATE_CMD[@]}")"
echo "  2. activate $ENV_NAME"
echo "  3. Rscript scripts/setup_r_packages.R      # installs disdat"
echo "  4. sdmbench env check"
echo

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry run -- nothing was created or modified."
  exit 0
fi

# --- 3. speed up classic conda ----------------------------------------------
if [[ "$TOOL" == "conda" ]]; then
  # libmamba became the default in conda 23.10, but plenty of cluster installs
  # predate that. Setting it is idempotent and harmless if already active.
  conda config --set solver libmamba >/dev/null 2>&1 \
    && echo "conda: solver set to libmamba" \
    || echo "conda: could not set the libmamba solver; the solve will be slow"
fi
# Strict channel priority is both faster and safer with conda-forge: it stops
# the solver mixing ABI-incompatible builds from defaults.
"$TOOL" config --set channel_priority strict >/dev/null 2>&1 || true

# --- 4. create ---------------------------------------------------------------
echo
echo ">>> creating the environment (this is the slow part -- several minutes)"
if ! "${CREATE_CMD[@]}"; then
  echo
  echo "error: environment creation failed."
  echo "  If the solve conflicted, try removing any partial environment first:"
  echo "      $TOOL env remove -n $ENV_NAME"
  echo "  If it was a network timeout, simply re-run -- packages are cached."
  exit 1
fi

# --- 5. locate the new environment's binaries --------------------------------
ENV_PREFIX="$("$TOOL" env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF}' | head -1)"
if [[ -z "$ENV_PREFIX" || ! -d "$ENV_PREFIX" ]]; then
  echo
  echo "Environment created, but its path could not be determined automatically."
  echo "Finish manually:"
  echo "    $TOOL activate $ENV_NAME"
  echo "    Rscript scripts/setup_r_packages.R"
  echo "    sdmbench env check"
  exit 0
fi

if [[ -x "$ENV_PREFIX/bin/Rscript" ]]; then
  RSCRIPT="$ENV_PREFIX/bin/Rscript"
  SDMBENCH="$ENV_PREFIX/bin/sdmbench"
else                                        # Windows layout (git-bash)
  RSCRIPT="$ENV_PREFIX/Scripts/Rscript.exe"
  SDMBENCH="$ENV_PREFIX/Scripts/sdmbench.exe"
fi

# --- 6. the one R package conda-forge does not have --------------------------
echo
echo ">>> installing disdat (the benchmark data; not on conda-forge)"
if [[ -x "$RSCRIPT" ]]; then
  "$RSCRIPT" "$REPO_ROOT/scripts/setup_r_packages.R" || {
    echo "warning: some R packages did not install. The affected models will"
    echo "         report SKIPPED_DEPENDENCY; the rest of the benchmark runs."
  }
else
  echo "warning: Rscript not found in the new environment at $RSCRIPT"
fi

# --- 7. cache location -------------------------------------------------------
# The cache holds datasets, checkpoints and run outputs. On HPC the default
# ($HOME/.cache) is usually a small quota-limited volume.
echo
if [[ -z "${SDMBENCH_CACHE:-}" ]]; then
  HOME_FREE_KB="$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -n "$HOME_FREE_KB" && "$HOME_FREE_KB" -lt 20000000 ]]; then
    echo ">>> WARNING: \$HOME has $((HOME_FREE_KB / 1024 / 1024)) GB free."
    echo "    The cache needs real space. Point it at scratch BEFORE fetching data:"
    echo "        export SDMBENCH_CACHE=/path/to/scratch/sdmbench"
    echo "        echo 'export SDMBENCH_CACHE=...' >> ~/.bashrc"
  fi
fi

# --- 8. verify ---------------------------------------------------------------
echo
echo ">>> verifying"
if [[ -x "$SDMBENCH" ]]; then
  "$SDMBENCH" env check
else
  echo "sdmbench not on the environment's PATH; activate and check manually."
fi

cat <<EOF

Done. Next:

    $TOOL activate $ENV_NAME
    export SDMBENCH_CACHE=/path/with/space/sdmbench   # if \$HOME is limited
    sdmbench data fetch disdat
    sdmbench reproduce tabpfn-sdm-2026 --max-species 3
EOF
