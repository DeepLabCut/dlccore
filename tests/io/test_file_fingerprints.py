import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from dlc_octopus.io import fingerprints
from dlc_octopus.io.errors import UnreadableFileError
from dlc_octopus.io.file_stat import FileStat
from dlc_octopus.io.fingerprints import FileFingerprintResult, fingerprint_file


def test_a_file_fingerprint_is_the_sha256_and_size_of_the_full_bytes(tmp_path: Path) -> None:
    content = b"labels" * 500_000  # spans several read chunks
    path = tmp_path / "CollectedData_x.h5"
    path.write_bytes(content)

    result = fingerprint_file(path)

    assert result == FileFingerprintResult(
        path=path, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )


def test_different_bytes_give_different_file_fingerprints(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"labels v1")
    second.write_bytes(b"labels v2")

    assert fingerprint_file(first).sha256 != fingerprint_file(second).sha256


def test_the_path_is_returned_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("Task: t\n")
    monkeypatch.chdir(tmp_path)

    assert fingerprint_file(Path("config.yaml")).path == Path("config.yaml")


def test_a_missing_file_raises_from_the_original(tmp_path: Path) -> None:
    with pytest.raises(UnreadableFileError, match=r"gone.h5") as raised:
        fingerprint_file(tmp_path / "gone.h5")

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_a_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    with pytest.raises(UnreadableFileError, match="not a regular file"):
        fingerprint_file(tmp_path)


def test_a_read_error_raises_from_the_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "locked.h5"
    path.write_bytes(b"x")

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", refuse)

    with pytest.raises(UnreadableFileError, match="denied") as raised:
        fingerprint_file(path)

    assert isinstance(raised.value.__cause__, PermissionError)


def test_a_file_that_changes_while_read_has_no_file_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "growing.h5"
    path.write_bytes(b"x")
    signatures = iter([(1, 100, 7, 1), (1, 200, 7, 1)])
    monkeypatch.setattr(fingerprints, "_signature", lambda _path: next(signatures))

    with pytest.raises(UnreadableFileError, match="changed while it was read"):
        fingerprint_file(path)


def test_a_file_whose_size_disagrees_with_the_bytes_read_has_no_file_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "truncated.h5"
    path.write_bytes(b"xyz")
    monkeypatch.setattr(fingerprints, "_signature", lambda _path: FileStat(99, 100, 7, 1))

    with pytest.raises(UnreadableFileError, match="changed while it was read"):
        fingerprint_file(path)


def test_a_file_removed_while_read_raises_from_the_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "removed.h5"
    path.write_bytes(b"x")
    signature = fingerprints._signature
    calls = 0

    def removed_after_the_read(target: Path) -> tuple[int, int, int, int] | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            target.unlink()
        return signature(target)

    monkeypatch.setattr(fingerprints, "_signature", removed_after_the_read)

    with pytest.raises(UnreadableFileError, match="changed while it was read") as raised:
        fingerprint_file(path)

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_a_file_replaced_while_read_has_no_file_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "replaced.h5"
    path.write_bytes(b"x")
    signatures = iter([FileStat(1, 100, 7, 1), FileStat(1, 100, 8, 1)])  # same size and mtime, another file
    monkeypatch.setattr(fingerprints, "_signature", lambda _path: next(signatures))

    with pytest.raises(UnreadableFileError, match="changed while it was read"):
        fingerprint_file(path)


def test_the_signature_tells_a_replacement_with_the_same_size_and_mtime(tmp_path: Path) -> None:
    # Control for the test above: the platform's inode and device really do change on an atomic replacement.
    path, replacement = tmp_path / "labels.h5", tmp_path / "labels.h5.tmp"
    path.write_bytes(b"one")
    replacement.write_bytes(b"two")
    mtime_ns = path.stat().st_mtime_ns
    os.utime(replacement, ns=(mtime_ns, mtime_ns))
    before = fingerprints._signature(path)

    os.replace(replacement, path)

    after = fingerprints._signature(path)
    assert before is not None and after is not None
    assert (before.size_bytes, before.mtime_ns) == (after.size_bytes, after.mtime_ns)
    assert before != after


def test_a_symlink_is_hashed_through(tmp_path: Path, create_symlink_or_skip: Callable[..., None]) -> None:
    target = tmp_path / "target.h5"
    target.write_bytes(b"labels")
    link = tmp_path / "link.h5"
    create_symlink_or_skip(link, target)

    result = fingerprint_file(link)

    assert (result.path, result.sha256) == (link, fingerprint_file(target).sha256)
