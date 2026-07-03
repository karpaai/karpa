"""DoS intake caps: a decompression bomb must be rejected before unpack."""
from __future__ import annotations

import gzip
import io

import pytest

from validator.hf_poller import _assert_blob_safe


def test_normal_blob_passes():
    _assert_blob_safe(gzip.compress(b"ok" * 2000))


def test_decompression_bomb_rejected():
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        z = b"\0" * (1 << 20)
        for _ in range(4096):  # 4 GiB decompressed from a ~4 MB blob
            g.write(z)
    with pytest.raises(ValueError, match="bomb|cap"):
        _assert_blob_safe(buf.getvalue())
