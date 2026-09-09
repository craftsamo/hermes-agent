"""Real-profile cookie refresh merges rows, newest wins, instead of overwriting.

Google rotates its session cookies (``__Secure-*PSIDTS``) a few times an hour and treats
an older value as a stolen session. Once the copy-browser has browsed, the user's own
profile holds the STALE value; overwriting the copy with it on every launch signed the
copy out, and the user's re-login only restarted the clock (2026-09-09).
"""

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import browser_connect as bc

# Chrome cookie store v24 (Brave 1.94), trimmed to the columns that matter plus the real
# identity index.
_SCHEMA = """
CREATE TABLE meta(key LONGVARCHAR NOT NULL UNIQUE PRIMARY KEY, value LONGVARCHAR);
INSERT INTO meta VALUES('version','24'),('last_compatible_version','24');
CREATE TABLE cookies(
  creation_utc INTEGER NOT NULL, host_key TEXT NOT NULL, top_frame_site_key TEXT NOT NULL,
  name TEXT NOT NULL, value TEXT NOT NULL, encrypted_value BLOB NOT NULL, path TEXT NOT NULL,
  expires_utc INTEGER NOT NULL, is_secure INTEGER NOT NULL, is_httponly INTEGER NOT NULL,
  last_access_utc INTEGER NOT NULL, has_expires INTEGER NOT NULL, is_persistent INTEGER NOT NULL,
  priority INTEGER NOT NULL, samesite INTEGER NOT NULL, source_scheme INTEGER NOT NULL,
  source_port INTEGER NOT NULL, last_update_utc INTEGER NOT NULL, source_type INTEGER NOT NULL,
  has_cross_site_ancestor INTEGER NOT NULL);
CREATE UNIQUE INDEX cookies_unique_index ON cookies(
  host_key, top_frame_site_key, has_cross_site_ancestor, name, path, source_scheme, source_port);
"""


def _store(path: Path, rows: list[tuple[str, str, bytes, int]], version: str = "24") -> Path:
    """Write a cookie store with ``(host, name, encrypted_value, last_update_utc)`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA.replace("'24'", f"'{version}'"))
        for i, (host, name, enc, updated) in enumerate(rows):
            conn.execute(
                "INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (1000 + i, host, "", name, "", enc, "/", 2_000_000, 1, 1, updated, 1, 1, 1, 0, 2,
                 443, updated, 0, 0),
            )
    return path


def _rows(path: Path) -> dict[tuple[str, str], tuple[bytes, int]]:
    with sqlite3.connect(path) as conn:
        return {
            (h, n): (bytes(e), u)
            for h, n, e, u in conn.execute(
                "SELECT host_key, name, encrypted_value, last_update_utc FROM cookies"
            )
        }


@pytest.fixture
def stores(tmp_path):
    src = tmp_path / "user" / "Profile 12" / "Cookies"
    dst = tmp_path / "copy" / "Default" / "Cookies"
    return src, dst


def test_merge_keeps_newer_copy_rows_and_takes_newer_source_rows(stores):
    src, dst = stores
    _store(src, [
        (".google.com", "__Secure-1PSIDTS", b"user-stale", 100),   # copy rotated since
        (".google.com", "SID", b"user-sid", 100),                   # identical both sides
        (".x.com", "auth_token", b"x-new-login", 500),              # new sign-in in user's browser
    ])
    _store(dst, [
        (".google.com", "__Secure-1PSIDTS", b"copy-fresh", 300),
        (".google.com", "SID", b"user-sid", 100),
        (".instagram.com", "sessionid", b"copy-only", 200),         # never in the user's store
    ])
    assert bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile 12") == 0
    got = _rows(dst)
    assert got[(".google.com", "__Secure-1PSIDTS")] == (b"copy-fresh", 300)  # copy wins
    assert got[(".google.com", "SID")] == (b"user-sid", 100)
    assert got[(".x.com", "auth_token")] == (b"x-new-login", 500)             # source added
    assert got[(".instagram.com", "sessionid")] == (b"copy-only", 200)         # untouched
    # The user's own store is never written.
    assert _rows(src)[(".google.com", "__Secure-1PSIDTS")] == (b"user-stale", 100)


def test_source_row_updated_later_replaces_copy_row(stores):
    src, dst = stores
    _store(src, [(".google.com", "__Secure-1PSIDTS", b"user-relogin", 900)])
    _store(dst, [(".google.com", "__Secure-1PSIDTS", b"copy-old", 300)])
    bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile 12")
    assert _rows(dst)[(".google.com", "__Secure-1PSIDTS")] == (b"user-relogin", 900)


def test_first_launch_still_copies_the_whole_store(stores):
    src, dst = stores
    _store(src, [(".google.com", "SID", b"user-sid", 100)])
    assert not dst.exists()
    bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile 12")
    assert _rows(dst) == {(".google.com", "SID"): (b"user-sid", 100)}


def test_schema_version_drift_falls_back_to_overwrite(stores, caplog):
    """A Chrome upgrade between launches changes the store version; merging across versions
    could write rows the new schema does not expect, so the copy is overwritten instead."""
    src, dst = stores
    _store(src, [(".google.com", "__Secure-1PSIDTS", b"user-stale", 100)], version="25")
    _store(dst, [(".google.com", "__Secure-1PSIDTS", b"copy-fresh", 300)], version="24")
    with caplog.at_level("INFO", logger=bc.logger.name):
        bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile 12")
    assert _rows(dst)[(".google.com", "__Secure-1PSIDTS")] == (b"user-stale", 100)
    assert "overwriting" in caplog.text and "versions differ" in caplog.text


def test_non_sqlite_copy_falls_back_to_overwrite(stores):
    src, dst = stores
    _store(src, [(".google.com", "SID", b"user-sid", 100)])
    dst.parent.mkdir(parents=True)
    dst.write_text("HALF-COPY-GARBAGE")
    bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile 12")
    assert _rows(dst) == {(".google.com", "SID"): (b"user-sid", 100)}


def test_other_auth_files_are_still_overwritten(stores):
    """Only the cookie store merges; Login Data / Web Data / Preferences keep the old
    overwrite semantics (they carry no rotating session state)."""
    src, dst = stores
    _store(src, [(".google.com", "SID", b"user-sid", 100)])
    _store(dst, [(".google.com", "SID", b"user-sid", 100)])
    (src.parent / "Preferences").write_text("user-prefs")
    (dst.parent / "Preferences").write_text("copy-prefs")
    bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile 12")
    assert (dst.parent / "Preferences").read_text() == "user-prefs"


def test_paths_with_uri_special_characters(tmp_path):
    src = tmp_path / "user 100%" / "Profile #12" / "Cookies"
    dst = tmp_path / "copy?x" / "Default" / "Cookies"
    _store(src, [(".x.com", "auth_token", b"new", 500)])
    _store(dst, [(".google.com", "__Secure-1PSIDTS", b"copy-fresh", 300)])
    assert bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), "Profile #12") == 0
    assert set(_rows(dst)) == {(".x.com", "auth_token"), (".google.com", "__Secure-1PSIDTS")}
