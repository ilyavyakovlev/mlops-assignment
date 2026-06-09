"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.

Each text-type column gets an inline comment showing up to 5 distinct
sample values so the model can use the correct case and encoding in
WHERE clauses (e.g. gender = 'M' not 'm', Admission = '-' not
'outpatient clinic').
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"

_TEXT_TYPES = {
    "TEXT", "VARCHAR", "CHAR", "CHARACTER", "NVARCHAR", "NCHAR",
    "CLOB", "STRING", "TINYTEXT", "MEDIUMTEXT", "LONGTEXT",
}


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _sample_values(conn: sqlite3.Connection, table: str, col: str, ctype: str) -> str:
    """Return an inline comment with ≤5 distinct sample values for text columns."""
    base = ctype.upper().split("(")[0].strip()
    if base not in _TEXT_TYPES:
        return ""
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(col)} FROM {_q(table)} "
            f"WHERE {_q(col)} IS NOT NULL LIMIT 5"
        ).fetchall()
        samples = [repr(str(r[0])) for r in rows if r[0] is not None]
        if not samples:
            return ""
        return "  -- e.g.: " + ", ".join(samples)
    except Exception:
        return ""


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                line += _sample_values(conn, t, name, ctype)
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                col_lines.append(
                    f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                )
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
