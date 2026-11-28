"""Imports pycolmap and reads COLMAP's dense arrays."""
import ctypes
import os
import sysconfig
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_pycolmap() -> Any:
    """pycolmap-cuda12 needs the pip-installed CUDA 12 libs, which are not on the loader path."""
    nvidia = Path(sysconfig.get_path("platlib")) / "nvidia"
    for name in ("libcudart.so.12", "libcurand.so.10"):
        for lib in nvidia.glob(f"*/lib/{name}"):
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    # The wheel's bundled curl looks for CA certificates where Arch, Fedora and others do not
    # keep them, so COLMAP's downloads (the vocabulary tree for loop detection) fail with SSL
    # error 77 and abort the process. Point it at the system bundle unless the user has.
    if "SSL_CERT_FILE" not in os.environ:
        for bundle in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem"):
            if os.path.exists(bundle):
                os.environ["SSL_CERT_FILE"] = bundle
                break
    try:
        return import_module("pycolmap")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "pycolmap is not installed. It has no Python 3.15 wheels; use Python 3.12-3.14 "
            "(uv python install 3.12 && uv python pin 3.12 && uv sync)."
        ) from e


def read_depth_map(path: Path) -> NDArray[np.float32]:
    """Read a COLMAP .bin depth map as (height, width); 0 means no estimate."""
    with path.open("rb") as f:
        header = b""
        while header.count(b"&") < 3:
            byte = f.read(1)
            if not byte:
                raise ValueError(f"{path} is not a COLMAP array")
            header += byte
        w, h, c = (int(x) for x in header.split(b"&")[:3])
        data = np.fromfile(f, dtype=np.float32)
    if c != 1 or data.size != w * h:
        raise ValueError(f"{path} is not a COLMAP depth map")
    return data.reshape(h, w)
