"""Real-profile cookie refresh merges rows, newest wins, instead of overwriting.

Google rotates its session cookies (``__Secure-*PSIDTS``) a few times an hour and treats
an older value as a stolen session. Once the copy-browser has browsed, the user's own
profile holds the STALE value; overwriting the copy with it on every launch signed the
copy out, and the user's re-login only restarted the clock (2026-09-09).
"""

import contextlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import browser_connect as bc
from hermes_cli import browser_connect_cookie_merge as cookie_merge

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


def _mirror(src: Path, dst: Path, profile: str = "Profile 12", merge: bool = True) -> dict[str, str]:
    """Re-sync ``src``'s profile into the copy that holds ``dst``; ``merge``: the copy is a completed
    snapshot of that same source profile (``snapshot_real_profile`` decides it from the marker)."""
    return bc._mirror_profile_auth(str(src.parents[1]), str(dst.parents[1]), profile, merge)


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
    assert _mirror(src, dst) == {}
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
    _mirror(src, dst)
    assert _rows(dst)[(".google.com", "__Secure-1PSIDTS")] == (b"user-relogin", 900)


def test_first_launch_still_copies_the_whole_store(stores):
    src, dst = stores
    _store(src, [(".google.com", "SID", b"user-sid", 100)])
    assert not dst.exists()
    _mirror(src, dst)
    assert _rows(dst) == {(".google.com", "SID"): (b"user-sid", 100)}


def test_schema_version_drift_falls_back_to_overwrite(stores, caplog):
    """A Chrome upgrade between launches changes the store version; merging across versions
    could write rows the new schema does not expect, so the copy is overwritten instead."""
    src, dst = stores
    _store(src, [(".google.com", "__Secure-1PSIDTS", b"user-stale", 100)], version="25")
    _store(dst, [(".google.com", "__Secure-1PSIDTS", b"copy-fresh", 300)], version="24")
    with caplog.at_level("INFO", logger=cookie_merge.logger.name):
        _mirror(src, dst)
    assert _rows(dst)[(".google.com", "__Secure-1PSIDTS")] == (b"user-stale", 100)
    assert "overwriting" in caplog.text and "versions differ" in caplog.text


def test_non_sqlite_copy_falls_back_to_the_lock_aware_copy(stores):
    """A copy that is not a cookie store is not merged; it takes the ordinary auth copy, which
    refuses to raw-copy over it, so the snapshot fails closed instead of launching signed out."""
    src, dst = stores
    _store(src, [(".google.com", "SID", b"user-sid", 100)])
    dst.parent.mkdir(parents=True)
    dst.write_text("HALF-COPY-GARBAGE")
    failed = _mirror(src, dst)
    assert set(failed) == {"Cookies"}
    assert dst.read_text() == "HALF-COPY-GARBAGE"


def test_other_auth_files_are_still_overwritten(stores):
    """Only the cookie store merges; Login Data / Web Data / Preferences keep the old
    overwrite semantics (they carry no rotating session state)."""
    src, dst = stores
    _store(src, [(".google.com", "SID", b"user-sid", 100)])
    _store(dst, [(".google.com", "SID", b"user-sid", 100)])
    (src.parent / "Preferences").write_text("user-prefs")
    (dst.parent / "Preferences").write_text("copy-prefs")
    _mirror(src, dst)
    assert (dst.parent / "Preferences").read_text() == "user-prefs"


def test_paths_with_uri_special_characters(tmp_path):
    src = tmp_path / "user 100%" / "Profile #12" / "Cookies"
    dst = tmp_path / "copy?x" / "Default" / "Cookies"
    _store(src, [(".x.com", "auth_token", b"new", 500)])
    _store(dst, [(".google.com", "__Secure-1PSIDTS", b"copy-fresh", 300)])
    assert _mirror(src, dst, "Profile #12") == {}
    assert set(_rows(dst)) == {(".x.com", "auth_token"), (".google.com", "__Secure-1PSIDTS")}


def test_held_copy_store_is_refused_within_its_deadline_and_left_intact(stores, monkeypatch):
    """A copy store another connection holds is reported locked (the snapshot then fails closed)
    within the auth deadline — never waited on past it, never overwritten through the lock."""
    src, dst = stores
    _store(src, [(".google.com", "__Secure-1PSIDTS", b"user-relogin", 900)])
    _store(dst, [(".google.com", "__Secure-1PSIDTS", b"copy-old", 300)])
    monkeypatch.setattr(bc, "_AUTH_BACKUP_DEADLINE_S", 0.3)
    with contextlib.closing(sqlite3.connect(dst)) as holder:
        holder.execute("begin exclusive")
        started = time.monotonic()
        failed = _mirror(src, dst)
        assert time.monotonic() - started < 5
        assert failed == {"Cookies": bc._AUTH_DB_LOCKED}
        holder.rollback()
    assert _rows(dst)[(".google.com", "__Secure-1PSIDTS")] == (b"copy-old", 300)


def test_without_merge_the_copy_store_is_overwritten(stores):
    """A copy that is not a snapshot of this source profile is another account's cookies: a merge
    would keep them signed in beside this profile's, so the store is replaced instead."""
    src, dst = stores
    _store(src, [(".google.com", "SID", b"this-account", 100)])
    _store(dst, [(".google.com", "SID", b"other-account", 900), (".x.com", "auth_token", b"other", 900)])
    assert _mirror(src, dst, merge=False) == {}
    assert _rows(dst) == {(".google.com", "SID"): (b"this-account", 100)}


def test_snapshot_merges_only_into_a_copy_of_the_same_source_profile(tmp_path, monkeypatch):
    """The first snapshot and a switch to another source profile copy the store; only a re-sync
    of the profile the completed copy came from merges into it."""
    user = tmp_path / "user"
    for profile in ("Profile 12", "Profile 7"):
        _store(user / profile / "Cookies", [(".google.com", "SID", profile.encode(), 100)])
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "home")

    def snapshot(profile):
        (user / "Local State").write_text(json.dumps({"profile": {"last_used": profile}}), encoding="utf-8")
        dst, err = bc.snapshot_real_profile("chrome", src=str(user))
        assert err is None
        return Path(dst) / "Default" / "Cookies"

    copy = snapshot("Profile 12")
    with sqlite3.connect(copy) as conn:  # the copy-browser rotated a cookie since
        conn.execute("UPDATE cookies SET encrypted_value = ?, last_update_utc = 500", (b"rotated",))
    assert _rows(snapshot("Profile 12")) == {(".google.com", "SID"): (b"rotated", 500)}
    assert _rows(snapshot("Profile 7")) == {(".google.com", "SID"): (b"Profile 7", 100)}
