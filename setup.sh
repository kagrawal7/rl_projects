#!/usr/bin/env bash
set -euo pipefail

on_error() {
  echo "Error: command failed on line $1: $2" >&2
}

trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

usage() {
  echo "Usage: $0 [-q|--quiet] [-v|--verbose] [--force]" >&2
  echo >&2
  echo "Options:" >&2
  echo "  -q, --quiet     Suppress pip installation output" >&2
  echo "  -v, --verbose   Show pip installation output, default" >&2
  echo "  --force         Remove existing .venv before creating a new one" >&2
  echo "  -h, --help      Show this help message" >&2
}

verbose=true
force=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--quiet)
      verbose=false
      shift
      ;;

    -v|--verbose)
      verbose=true
      shift
      ;;

    --force)
      force=true
      shift
      ;;

    -h|--help)
      usage
      exit 0
      ;;

    -*)
      echo "Error: unknown option: $1" >&2
      usage
      exit 1
      ;;

    *)
      echo "Error: unexpected argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Move to the directory containing this script.
# This assumes make_env.sh lives at the project root.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_name="$(basename -- "$project_dir")"

cd "$project_dir"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: no pyproject.toml found in $project_dir" >&2
  echo "This script should live in the project root." >&2
  exit 1
fi

log() {
  if [[ "$verbose" == true ]]; then
    echo "$@"
  fi
}

run_python() {
  if [[ "$verbose" == true ]]; then
    .venv/bin/python "$@"
  else
    .venv/bin/python "$@" >/dev/null
  fi
}

run_pip() {
  run_python -m pip "$@"
}

sanitize_kernel_name() {
  local name="$1"
  name="${name// /_}"
  name="${name//./_}"
  name="${name//-/_}"
  echo "${name}_jupyter"
}

kernel_name="$(sanitize_kernel_name "$project_name")"

log "Project: $project_dir"
log "Kernel name: $kernel_name"

if [[ -d ".venv" ]]; then
  if [[ "$force" == true ]]; then
    log "Removing existing virtual environment: $project_dir/.venv"
    rm -rf .venv
  else
    echo "Error: $project_dir already contains a .venv environment." >&2
    echo "Use --force to delete and recreate it." >&2
    exit 1
  fi
fi

log "Creating virtual environment: $project_dir/.venv"
python3 -m venv .venv

echo "Installing dependencies..."

run_pip install --upgrade pip setuptools wheel
run_pip install -e ".[notebooks,dev]"
run_pip install ipykernel jupyter

log "Registering Jupyter kernel: $kernel_name"

.venv/bin/python -m ipykernel install \
  --user \
  --name="$kernel_name" \
  --display-name "$kernel_name"

echo
echo "Setup complete."
echo "Project: $project_dir"
echo "Virtual environment: $project_dir/.venv"
echo "Jupyter kernel: $kernel_name"
echo
echo "To activate this environment manually, run:"
echo "  source \"$project_dir/.venv/bin/activate\""
echo
echo "You can also avoid activation and run Python directly:"
echo "  \"$project_dir/.venv/bin/python\""
echo
echo "To open Jupyter, run:"
echo "  \"$project_dir/.venv/bin/jupyter\" lab"
echo
echo "Then select this kernel inside the notebook:"
echo "  $kernel_name"
