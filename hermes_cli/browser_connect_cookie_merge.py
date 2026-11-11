"""Row-wise refresh of the real-profile copy's cookie store.

Every consented launch re-syncs the user's auth files into the hermes-owned copy. For the cookie
store a plain overwrite discards every cookie the copy-browser itself refreshed since the last
launch. Google rotates its session cookies (``__Secure-*PSIDTS``) a few times an hour and treats an
older value as a stolen session, so once the copy has browsed, the user's profile holds the STALE
value: overwriting with it signs the copy out, and the user's re-login only restarts the clock.
Merging row by row, newest ``last_update_utc`` wins, keeps the copy's rotations across relaunches
while a sign-in the user makes in their own browser still lands. Deletions do not propagate — a
sign-out in the user's browser leaves the copy signed in until the cookie expires or is removed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from pathlib import Path

from hermes_cli import browser_connect_prep as _prep

logger = logging.getLogger(__name__)

_COOKIE_STORE = "Cookies"


class CookieMergeUnsupported(Exception):
    """The two cookie stores cannot be merged row by row; the caller overwrites instead."""


def _unique_columns(conn: sqlite3.Connection, schema: str) -> list[str]:
    """Columns of the cookies table's UNIQUE index — Chromium's own cookie identity."""
    for _seq, name, unique, *_ in conn.execute(f"PRAGMA {schema}.index_list('cookies')"):
        if unique:
            cols = [row[2] for row in conn.execute(f"PRAGMA {schema}.index_info('{name}')")]
            if cols:
                return cols
    raise CookieMergeUnsupported("no unique index on cookies")


def _merge_rows(conn: sqlite3.Connection) -> None:
    versions = {schema: dict(conn.execute(f"SELECT key, value FROM {schema}.meta"))
                for schema in ("main", "src")}
    if versions["main"].get("version") != versions["src"].get("version"):
        raise CookieMergeUnsupported("cookie store versions differ")
    cols = {schema: [row[1] for row in conn.execute(f"PRAGMA {schema}.table_info('cookies')")]
            for schema in ("main", "src")}
    if not cols["main"] or cols["main"] != cols["src"] or "last_update_utc" not in cols["main"]:
        raise CookieMergeUnsupported("cookie table columns differ")
    key = _unique_columns(conn, "main")
    if key != _unique_columns(conn, "src"):
        raise CookieMergeUnsupported("cookie identity columns differ")
    column_list = ", ".join(f'"{c}"' for c in cols["main"])
    match = " AND ".join(f'd."{c}" IS s."{c}"' for c in key)
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO main.cookies ({column_list}) "
            f"SELECT {column_list} FROM src.cookies AS s "
            f"WHERE NOT EXISTS (SELECT 1 FROM main.cookies AS d "
            f"WHERE {match} AND d.last_update_utc >= s.last_update_utc)")


def merge_cookie_db(src_file: str, dst_file: str) -> str | None:
    """Merge ``src_file``'s cookies into the EXISTING copy ``dst_file``; None on success, else why.

    The source is ATTACHed read-only (never written) through a normal SQLite read, so committed
    WAL content is included, and the copy is written through SQLite rather than replaced. Lock
    waits are bounded like the auth backup's. Raises ``CookieMergeUnsupported`` when the stores
    differ in schema (a browser upgrade between launches) or either is not a cookie store.
    """
    from hermes_cli.browser_connect import _AUTH_BACKUP_DEADLINE_S, _AUTH_DB_LOCKED

    wait = _prep.remaining(_AUTH_BACKUP_DEADLINE_S)
    try:
        with contextlib.closing(sqlite3.connect(Path(dst_file).resolve().as_uri(), uri=True,
                                                timeout=wait)) as conn:
            conn.execute("ATTACH DATABASE ? AS src", (Path(src_file).resolve().as_uri() + "?mode=ro",))
            try:
                _merge_rows(conn)
            finally:
                conn.execute("DETACH DATABASE src")
        return None
    except sqlite3.OperationalError as e:
        if getattr(e, "sqlite_errorcode", None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            _prep.remaining()  # an exhausted call budget is a timeout, not a held store
            logger.debug("real-profile: cookie merge into %s blocked: %s", dst_file, e)
            return _AUTH_DB_LOCKED
        raise CookieMergeUnsupported(str(e)) from e
    except sqlite3.DatabaseError as e:
        raise CookieMergeUnsupported(str(e)) from e


def refresh_auth_file(src_file: str, dst_file: str) -> str | None:
    """Bring one auth file in the copy up to date; None on success, else why it failed. A cookie
    store that already exists in the copy is MERGED; everything else — and a store that cannot be
    merged — takes the facade's lock-aware copy."""
    from hermes_cli.browser_connect import _copy_auth_file

    if os.path.basename(src_file) == _COOKIE_STORE and os.path.isfile(dst_file):
        try:
            return merge_cookie_db(src_file, dst_file)
        except CookieMergeUnsupported as e:
            logger.info("real-profile: overwriting %s instead of merging (%s)", dst_file, e)
    return _copy_auth_file(src_file, dst_file)
