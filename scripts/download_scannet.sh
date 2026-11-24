#!/usr/bin/env bash
# Download the ScanNet files this project needs into room-data/.
#   scripts/download_scannet.sh [scene_id ...]
# Wraps the official downloader, which ScanNet emails you once you accept its Terms of Use
# (http://www.scan-net.org/). Its own terms prompt is left for you to answer.
# SCANNET_DOWNLOADER and ROOM_DATA override the default locations.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${ROOM_DATA:-$root/room-data}"
downloader="${SCANNET_DOWNLOADER:-$data/download-scannet.py}"
python="${PYTHON:-python3}"
suffixes=(.sens .txt _vh_clean_2.ply _vh_clean_2.labels.ply)
scenes=("$@")
((${#scenes[@]})) || scenes=(scene0000_00)

for scene in "${scenes[@]}"; do
  [[ $scene =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || { echo "bad scene id: $scene" >&2; exit 2; }
done
mkdir -p "$data"
if [[ ! -f $downloader ]]; then
  echo "Official downloader not found: $downloader" >&2
  echo "Request access at http://www.scan-net.org/, save the script there, and re-run." >&2
  exit 1
fi

# The official script may or may not add a scans/ level, so look in either.
locate_file() { find "$data" -type f -path "*/$1/$1$2" -size +0c -print -quit; }

failed=0
for scene in "${scenes[@]}"; do
  for suffix in "${suffixes[@]}"; do
    if [[ -n $(locate_file "$scene" "$suffix") ]]; then
      echo "have $scene$suffix"
      continue
    fi
    echo "downloading $scene$suffix"
    "$python" "$downloader" -o "$data" --id "$scene" --type "$suffix" || true
    if [[ -z $(locate_file "$scene" "$suffix") ]]; then
      echo "FAILED: $scene$suffix" >&2
      failed=1
    fi
  done
done
exit "$failed"
