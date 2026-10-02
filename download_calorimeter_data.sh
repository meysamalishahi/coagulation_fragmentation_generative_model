#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR="$PROJECT_DIR/data/calochallenge/dataset_1_photons"
mkdir -p "$DATA_DIR"

download_and_check() {
  NAME=$1
  EXPECTED_MD5=$2
  URL="https://zenodo.org/records/8099322/files/$NAME?download=1"
  TARGET="$DATA_DIR/$NAME"
  curl --fail --location --retry 5 --continue-at - --output "$TARGET" "$URL"
  printf '%s  %s\n' "$EXPECTED_MD5" "$TARGET" | md5sum --check --status
  printf '%s\n' "Verified $TARGET"
}

download_and_check dataset_1_photons_1.hdf5 005d2adeda7db034b388112661265656
download_and_check dataset_1_photons_2.hdf5 4767715ed56e99565fd9c67340661e70
