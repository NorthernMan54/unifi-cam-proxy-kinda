import os
import time

from unifi.cams.frigate_lib import snapshots


def test_write_snapshot_tempfile_uses_snapshot_dir():
    path = snapshots.write_snapshot_tempfile(b"jpeg-bytes")
    try:
        assert path.parent == snapshots.snapshot_dir()
        assert path.suffix == ".jpg"
        assert path.read_bytes() == b"jpeg-bytes"
    finally:
        path.unlink(missing_ok=True)


def test_prune_snapshot_dir_removes_only_stale_files():
    fresh = snapshots.write_snapshot_tempfile(b"fresh")
    stale = snapshots.write_snapshot_tempfile(b"stale")
    old = time.time() - snapshots.SNAPSHOT_MAX_AGE - 10
    os.utime(stale, (old, old))
    try:
        assert snapshots.prune_snapshot_dir() == 1
        assert fresh.exists()
        assert not stale.exists()
    finally:
        fresh.unlink(missing_ok=True)
