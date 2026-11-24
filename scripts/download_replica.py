#!/usr/bin/env python3
"""Download Replica RGB-D sequences (rendered by the iMAP authors, hosted by NICE-SLAM).

    scripts/download_replica.py [room0 office0 ...]      default: room0 office0

Only the requested scenes are fetched from the 12.4 GB archive, with HTTP range requests.
Safe to re-run: complete files are skipped. Data terms: non-commercial research/education,
https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE
"""
import argparse
import http.client
import io
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
SCENES = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "room-data" / "replica"


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges, on one reused connection."""

    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        self._host, self._path, self._pos = parts.netloc, parts.path, 0
        self._scheme = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        self._connection: http.client.HTTPConnection | None = None
        response, _ = self._request("HEAD", {})
        self._size = int(response.getheader("Content-Length", "0"))

    def _request(self, method: str, headers: dict[str, str], attempts: int = 5):
        for attempt in range(attempts):
            try:
                if self._connection is None:
                    self._connection = self._scheme(self._host, timeout=60)
                self._connection.request(method, self._path, headers=headers)
                response = self._connection.getresponse()
                body = response.read()
                if response.status not in (200, 206):
                    raise OSError(f"HTTP {response.status} from {self._host}")
                if headers and response.status != 206:
                    raise OSError("server ignored the range request")
                return response, body
            except (OSError, http.client.HTTPException):
                if self._connection is not None:
                    self._connection.close()
                self._connection = None
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._pos = offset + {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        return self._pos

    def readinto(self, buffer) -> int:
        if self._pos >= self._size:
            return 0
        end = min(self._pos + len(buffer), self._size)
        _, body = self._request("GET", {"Range": f"bytes={self._pos}-{end - 1}"})
        buffer[:len(body)] = body
        self._pos += len(body)
        return len(body)


def download(url: str, scenes: list[str], out: Path) -> None:
    with zipfile.ZipFile(io.BufferedReader(RemoteFile(url), 16 << 20)) as archive:
        wanted = []
        for info in archive.infolist():
            path = Path(info.filename)
            if info.is_dir() or path.parts[:1] != ("Replica",) or ".." in path.parts:
                continue
            rel = Path(*path.parts[1:])
            if rel.name == "cam_params.json" or rel.parts[0] in scenes or rel.stem in {f"{s}_mesh" for s in scenes}:
                wanted.append((info, out / rel))
        # A file of the right size was renamed into place after its CRC check.
        todo = [(i, p) for i, p in wanted if not (p.exists() and p.stat().st_size == i.file_size)]
        total, done, shown = sum(i.compress_size for i, _ in todo), 0, -1
        print(f"{len(todo)} of {len(wanted)} files to fetch, {total / 1e9:.2f} GB", flush=True)
        for info, path in sorted(todo, key=lambda t: t[0].header_offset):
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            with archive.open(info) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            part.replace(path)
            done += info.compress_size
            if (percent := int(100 * done / max(total, 1))) // 5 != shown:
                shown = percent // 5
                print(f"{percent}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", default=["room0", "office0"], metavar="scene", help=", ".join(SCENES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    unknown = set(args.scenes) - set(SCENES)
    if unknown:
        parser.error(f"unknown scenes: {', '.join(sorted(unknown))}")
    download(URL, args.scenes, args.out)


if __name__ == "__main__":
    main()
