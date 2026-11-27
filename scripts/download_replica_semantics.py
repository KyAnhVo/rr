#!/usr/bin/env python3
"""Download Replica's semantic ground truth (for scoring stage 5) from the original release.

    scripts/download_replica_semantics.py [room0 office0 ...]      default: room0 office0

Saves room-data/replica/<scene>_semantic/{mesh_semantic.ply, info_semantic.json}.

The release is one .tar.gz split into parts, which cannot be read at random: the whole archive
(tens of GB) is streamed, but only the wanted files are written, and it stops once it has them.
Data terms: non-commercial research/education,
https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE
"""
import argparse
import io
import string
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

BASE = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part"
SCENES = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
FILES = ("mesh_semantic.ply", "info_semantic.json")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "room-data" / "replica"


def part_suffixes():
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            yield a + b


class Parts(io.RawIOBase):
    """The parts base+'aa', base+'ab', ... read back to back, until one does not exist."""

    def __init__(self, base: str) -> None:
        self._base, self._suffixes = base, part_suffixes()
        self._response = None
        self._done = False
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def _next(self) -> bool:
        suffix = next(self._suffixes)
        for attempt in range(5):
            try:
                self._response = urllib.request.urlopen(self._base + suffix, timeout=120)
                return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                if attempt == 4:
                    raise
            except OSError:
                if attempt == 4:
                    raise
            time.sleep(2**attempt)
        return False

    def readinto(self, buffer) -> int:
        while not self._done:
            if self._response is None and not self._next():
                self._done = True
                break
            assert self._response is not None
            n = self._response.readinto(buffer)
            if n:
                self.bytes_read += n
                return n
            self._response.close()
            self._response = None
        return 0


def wanted_path(member: str, scenes: dict[str, str], out: Path) -> Path | None:
    """Where to save an archive member, if it is one we want; never outside `out`."""
    parts = PurePosixPath(member).parts
    if len(parts) < 3 or parts[-1] not in FILES or parts[-2] != "habitat" or ".." in parts:
        return None
    scene = scenes.get(parts[-3])
    return out / f"{scene}_semantic" / parts[-1] if scene else None


def download(base: str, scenes: list[str], out: Path) -> None:
    # Replica names scenes room_0, office_0, ...; NICE-SLAM drops the underscore.
    names = {s.rstrip("0123456789") + "_" + s[len(s.rstrip("0123456789")):]: s for s in scenes}
    todo = {out / f"{s}_semantic" / f for s in scenes for f in FILES}
    todo = {p for p in todo if not p.exists()}
    if not todo:
        print("all files present")
        return
    stream = Parts(base)
    shown = 0
    with tarfile.open(fileobj=io.BufferedReader(stream, 16 << 20), mode="r|gz") as archive:
        for member in archive:
            if stream.bytes_read - shown > 1 << 30:
                shown = stream.bytes_read
                print(f"{shown / 1e9:.0f} GB read, {len(todo)} files to go", flush=True)
            path = wanted_path(member.name, names, out)
            if path is None or path not in todo or not member.isfile():
                continue
            source = archive.extractfile(member)
            assert source is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            with part.open("wb") as f:
                while chunk := source.read(16 << 20):
                    f.write(chunk)
            part.replace(path)
            todo.discard(path)
            print(f"saved {path}", flush=True)
            if not todo:
                return
    if todo:
        sys.exit("not in the archive: " + ", ".join(str(p) for p in sorted(todo)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", default=["room0", "office0"], metavar="scene", help=", ".join(SCENES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    unknown = set(args.scenes) - set(SCENES)
    if unknown:
        parser.error(f"unknown scenes: {', '.join(sorted(unknown))}")
    download(BASE, args.scenes, args.out)


if __name__ == "__main__":
    main()
