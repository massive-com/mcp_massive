import csv
import io
import math
import re
import sqlite3
import time
from typing import Any, cast
from datetime import datetime, timezone

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError as SQLParseError
from pydantic import BaseModel, Field

TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")

# SQLite system tables that must never be used as user table names.
_RESERVED_TABLE_NAMES: frozenset[str] = frozenset(
    {
        "sqlite_master",
        "sqlite_sequence",
        "sqlite_stat1",
        "sqlite_stat2",
        "sqlite_stat3",
        "sqlite_stat4",
    }
)

# Name-based denylist of dangerous SQLite functions.  These are real SQLite
# functions (or loadable-extension functions) that could escape the sandbox.
# Note: unrecognised functions are already blocked via the Anonymous node
# catch-all in _validate_sql, so we only need to list functions that sqlglot
# might give a typed Func subclass.
_BLOCKED_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "load_extension",  # loads arbitrary shared libraries
        "readfile",  # fileio extension: reads files from disk
        "writefile",  # fileio extension: writes files to disk
        "edit",  # fileio extension: opens editor on file
        "fts3_tokenizer",  # can be abused to call arbitrary C functions
    }
)

# Custom functions registered via _register_custom_functions, plus FTS5
# ranking/highlighting built-ins, that sqlglot parses as Anonymous nodes.
# These are safe and must be explicitly allowed through the AST validator.
_CUSTOM_ANONYMOUS_FUNCTIONS: frozenset[str] = frozenset(
    {
        "to_timestamp",
        "to_date",
        # FTS5 ranking and highlighting built-ins
        "bm25",
        "snippet",
        "highlight",
    }
)

# SQLite read-only PRAGMAs the authorizer must allow because FTS5 invokes
# them internally during MATCH queries.  Users cannot trigger these: the
# statement-type check in _validate_sql rejects any user-issued PRAGMA.
_AUTHORIZER_ALLOWED_PRAGMAS: frozenset[str] = frozenset(
    {
        "data_version",
    }
)

DEFAULT_MAX_TABLES = 50
DEFAULT_MAX_ROWS = 50_000
TTL_SECONDS = 3600
QUERY_TIMEOUT_SECONDS = 30
# Cells in a StoreSummary preview are capped tightly — the preview is meant
# to convey schema and shape, not content.  Long-text rows (e.g. filings)
# would otherwise blow up the call_api response in tokens.
PREVIEW_MAX_CELL_CHARS = 200

# Functions allowed through the SQLite authorizer.  This must include every
# custom function registered via _register_custom_functions as well as safe
# SQLite built-ins that queries may use.
_AUTHORIZER_ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        # Custom registered functions
        "sqrt",
        "ln",
        "exp",
        "power",
        "concat",
        "stddev",
        "stddev_samp",
        "corr",
        "to_timestamp",
        "to_date",
        # Standard SQLite aggregates / scalars
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "total",
        "group_concat",
        "coalesce",
        "nullif",
        "ifnull",
        "iif",
        "abs",
        "round",
        "typeof",
        "unicode",
        "zeroblob",
        "lower",
        "upper",
        "length",
        "substr",
        "substring",
        "trim",
        "ltrim",
        "rtrim",
        "replace",
        "instr",
        "hex",
        "quote",
        "char",
        "printf",
        "format",
        "like",
        "glob",
        "date",
        "time",
        "datetime",
        "julianday",
        "strftime",
        "unixepoch",
        "timediff",
        "cast",
        # Window functions
        "row_number",
        "rank",
        "dense_rank",
        "ntile",
        "lag",
        "lead",
        "first_value",
        "last_value",
        "nth_value",
        # NULL / type checking
        "likelihood",
        "likely",
        "unlikely",
        # CASE is not a function but some drivers report it
        "case",
        # FTS5 MATCH operator and ranking/highlighting built-ins
        "match",
        "bm25",
        "snippet",
        "highlight",
    }
)


def _select_only_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    dbname: str | None,
    source: str | None,
) -> int:
    """SQLite authorizer callback that only permits SELECT reads.

    Installed after table population so CREATE/INSERT for setup are
    unaffected.  Blocks INSERT, UPDATE, DELETE, CREATE, DROP, ATTACH,
    PRAGMA, and any function not in the allowlist.
    """
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        if arg2 is not None and arg2.lower() in _AUTHORIZER_ALLOWED_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    # FTS5 issues PRAGMA data_version internally to detect schema changes.
    # User-issued PRAGMAs are already blocked by _validate_sql, so this is
    # only reachable for trusted internal callers.
    if action == sqlite3.SQLITE_PRAGMA:
        if arg1 is not None and arg1.lower() in _AUTHORIZER_ALLOWED_PRAGMAS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def _truncate_cell(value: Any, max_chars: int) -> Any:
    """Return *value* unchanged unless its string form exceeds *max_chars*.

    When truncation fires, the returned string ends with a marker like
    ``... [truncated: N more chars]`` so an LLM caller can see how much
    was elided and decide whether to re-query with ``max_cell_chars=0``
    or a tighter ``WHERE`` clause.
    """
    if value is None:
        return None
    s = value if isinstance(value, str) else str(value)
    if len(s) <= max_chars:
        return value
    excess = len(s) - max_chars
    return f"{s[:max_chars]}... [truncated: {excess} more chars]"


class Table:
    """Column-oriented in-memory table (replaces pl.DataFrame)."""

    def __init__(self, columns: list[str], data: dict[str, list]):
        if columns:
            lengths = {col: len(data[col]) for col in columns if col in data}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                raise ValueError(
                    f"All columns must have the same length, got inconsistent lengths: "
                    f"{
                        dict(
                            sorted(
                                ((col, length) for col, length in lengths.items()),
                                key=lambda x: x[0],
                            )
                        )
                    }"
                )
        self.columns = columns
        self.data = data

    @classmethod
    def from_records(cls, records: list[dict]) -> "Table":
        """Build from list of flat dicts.

        Preserves insertion order of keys across all records.
        Fills missing keys with None.
        When two columns differ only in case (e.g. ``T`` and ``t``),
        the later column is renamed with a ``_2`` suffix to avoid
        SQLite case-insensitive collisions.
        """
        if not records:
            return cls([], {})
        # Collect all keys in insertion order (original casing)
        seen: set[str] = set()
        raw_columns: list[str] = []
        for rec in records:
            for key in rec:
                if key not in seen:
                    seen.add(key)
                    raw_columns.append(key)

        # Deduplicate case-insensitive collisions for SQLite compatibility
        ci_seen: set[str] = set()
        columns: list[str] = []
        col_map: dict[str, str] = {}  # original key -> final column name
        for col in raw_columns:
            if col.lower() in ci_seen:
                renamed = f"{col}_2"
                while renamed.lower() in ci_seen:
                    renamed += "_2"
                columns.append(renamed)
                ci_seen.add(renamed.lower())
                col_map[col] = renamed
            else:
                columns.append(col)
                ci_seen.add(col.lower())
                col_map[col] = col

        data: dict[str, list] = {col: [] for col in columns}
        for rec in records:
            for raw_col in raw_columns:
                data[col_map[raw_col]].append(rec.get(raw_col))
        return cls(columns, data)

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(self.data[self.columns[0]])

    def head(self, n: int) -> "Table":
        new_data = {col: vals[:n] for col, vals in self.data.items()}
        return Table(list(self.columns), new_data)

    def rows(self) -> list[tuple]:
        length = len(self)
        return [tuple(self.data[col][i] for col in self.columns) for i in range(length)]

    def write_csv(self, max_cell_chars: int = 0) -> str:
        """Serialize the table as CSV.

        If ``max_cell_chars > 0``, any cell whose string representation
        exceeds that length is truncated with a visible marker so the
        caller knows how many characters were omitted.  Useful for
        long FTS5 TEXT columns (e.g. 10-K risk factors) where a single
        row can be thousands of tokens.  ``max_cell_chars = 0`` leaves
        all cells untouched.
        """
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.columns)
        if max_cell_chars <= 0:
            for row in self.rows():
                writer.writerow(row)
        else:
            for row in self.rows():
                writer.writerow(tuple(_truncate_cell(v, max_cell_chars) for v in row))
        return buf.getvalue()

    def get_column(self, name: str) -> list:
        if name not in self.data:
            raise ValueError(f"Column '{name}' not found. Available: {self.columns}")
        return self.data[name]

    def with_column(self, name: str, values: list) -> "Table":
        new_columns = list(self.columns)
        new_data = {col: list(vals) for col, vals in self.data.items()}
        if name not in new_data:
            new_columns.append(name)
        new_data[name] = values
        return Table(new_columns, new_data)

    def sort(self, column: str) -> "Table":
        """Sort rows by a single column.

        None values sort last.  Mixed types are coerced to their string
        representation to avoid TypeError on comparison.
        """
        col_data = self.data[column]

        def _sort_key(i: int):
            v = col_data[i]
            if v is None:
                return (2,)  # sort last
            try:
                return (0, v)
            except TypeError:
                return (1, str(v))

        # Detect mixed non-None types and fall back to string comparison
        types_seen: set[type] = set()
        for v in col_data:
            if v is not None:
                types_seen.add(type(v))
        if len(types_seen) > 1:
            # Mixed types: compare as (is_none, type_name, str_value) for stable ordering
            def _mixed_key(i: int):
                v = col_data[i]
                if v is None:
                    return (2, "", "")
                return (0, type(v).__name__, str(v))

            indices = sorted(range(len(self)), key=_mixed_key)
        else:
            indices = sorted(
                range(len(self)), key=lambda i: (col_data[i] is None, col_data[i])
            )

        new_data = {col: [self.data[col][i] for i in indices] for col in self.columns}
        return Table(list(self.columns), new_data)

    def __getitem__(self, name: str) -> list:
        return self.get_column(name)

    def equals(self, other: "Table") -> bool:
        return self.columns == other.columns and self.data == other.data


class StoreSummary(BaseModel):
    table_name: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")
    row_count: int = Field(ge=0)
    columns: list[str]
    preview: str


def _infer_sqlite_affinity(values: list) -> str:
    """Infer SQLite column affinity from the first non-None value."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            return "INTEGER"
        if isinstance(v, int):
            return "INTEGER"
        if isinstance(v, float):
            return "REAL"
        return "TEXT"
    return "TEXT"


# FTS5 reserves these identifiers; a user column with this name collides
# with the virtual table's own hidden columns, so we fall back to a plain
# table when any column uses one of these names.
_FTS5_RESERVED_COLUMN_NAMES: frozenset[str] = frozenset({"rank", "rowid"})


def _create_and_populate_table(
    conn: sqlite3.Connection, name: str, table: Table
) -> None:
    """Create a SQLite table from a Table and bulk-insert rows.

    When at least one column has TEXT affinity and there are no
    FTS5-reserved column-name collisions, the table is created as an
    FTS5 virtual table with non-text columns marked ``UNINDEXED``.
    This lets users run ``WHERE {name} MATCH 'query'`` directly on
    the base table — no mirror, no JOIN.  FTS5 preserves SQLite's
    dynamic type for stored values, so numeric ORDER BY and
    SUM/AVG/MIN/MAX work as they would on a plain table.

    Falls back to a plain ``CREATE TABLE`` when:
      - the table has no TEXT columns, or
      - a column name collides with FTS5 reserved names (``rank``,
        ``rowid``) or the table's own name.
    """
    cols = table.columns
    text_cols = {c for c in cols if _infer_sqlite_affinity(table.data[c]) == "TEXT"}
    lower_names = {c.lower() for c in cols}
    reserved_collision = bool(lower_names & _FTS5_RESERVED_COLUMN_NAMES)
    name_collision = name.lower() in lower_names

    if text_cols and not reserved_collision and not name_collision:
        col_defs_parts = [
            f'"{c}"' if c in text_cols else f'"{c}" UNINDEXED' for c in cols
        ]
        col_defs = ", ".join(col_defs_parts)
        conn.execute(
            f'CREATE VIRTUAL TABLE "{name}" USING fts5('
            f"{col_defs}, tokenize='porter unicode61')"
        )
    else:
        affinities = [_infer_sqlite_affinity(table.data[c]) for c in cols]
        col_defs = ", ".join(f'"{c}" {a}' for c, a in zip(cols, affinities))
        conn.execute(f'CREATE TABLE "{name}" ({col_defs})')

    if len(table) > 0:
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(f'"{c}"' for c in cols)
        conn.executemany(
            f'INSERT INTO "{name}" ({col_list}) VALUES ({placeholders})',
            table.rows(),
        )


class _StddevAggregate:
    """SQLite custom aggregate for sample standard deviation (STDDEV / STDDEV_SAMP)."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: float | None) -> None:
        if value is not None:
            self.values.append(value)

    def finalize(self) -> float | None:
        n = len(self.values)
        if n < 2:
            return None
        mean = sum(self.values) / n
        variance = sum((x - mean) ** 2 for x in self.values) / (n - 1)
        return math.sqrt(variance)


class _CorrAggregate:
    """SQLite custom aggregate for Pearson correlation coefficient."""

    def __init__(self) -> None:
        self.xs: list[float] = []
        self.ys: list[float] = []

    def step(self, x: float | None, y: float | None) -> None:
        if x is not None and y is not None:
            self.xs.append(float(x))
            self.ys.append(float(y))

    def finalize(self) -> float | None:
        n = len(self.xs)
        if n < 2:
            return None
        mean_x = sum(self.xs) / n
        mean_y = sum(self.ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(self.xs, self.ys))
        var_x = sum((x - mean_x) ** 2 for x in self.xs)
        var_y = sum((y - mean_y) ** 2 for y in self.ys)
        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return None
        return cov / denom


def _epoch_to_seconds(value: float | int) -> float:
    """Normalize a numeric epoch timestamp (seconds, ms, or ns) to seconds.

    Heuristic based on digit count:
      - 10 digits → seconds   (unix epoch ~1.7e9)
      - 13 digits → ms        (most Massive API fields: ``t``, ``sip_timestamp``)
      - 16+ digits → ns       (snapshot ``last_updated`` fields)
    """
    v = float(value)
    abs_v = abs(v)
    if abs_v > 1e15:  # nanoseconds
        return v / 1e9
    if abs_v > 1e11:  # milliseconds
        return v / 1e3
    return v  # already seconds


def _to_timestamp(value: float | int | str | None) -> str | None:
    """Convert a timestamp to ISO-8601 UTC datetime string.

    Accepts epoch seconds/ms/ns (auto-detected by magnitude) or an
    ISO-8601 string (returned as-is after validation).
    """
    if value is None:
        return None

    if isinstance(value, str):
        # Already a string timestamp — return as-is if it looks like a date
        stripped = value.strip()
        if stripped and (stripped[0].isdigit() or stripped[0] == "-"):
            # Might be a numeric string — try parsing as number
            try:
                return _to_timestamp(float(stripped))
            except ValueError:
                pass
        return stripped  # ISO-8601 or date string — pass through

    return datetime.fromtimestamp(_epoch_to_seconds(value), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _to_date(value: float | int | str | None) -> str | None:
    """Convert a timestamp to a UTC date string (YYYY-MM-DD).

    Accepts epoch seconds/ms/ns (auto-detected by magnitude) or an
    ISO-8601 string.  Useful for cross-asset JOINs where crypto bars
    use midnight UTC and equity bars use 4 AM ET.
    """
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        # If it already looks like YYYY-MM-DD, return the date portion
        if re.compile(r"^\d{4}-\d{2}-\d{2}").match(stripped):
            return stripped[:10]
        # Numeric string
        try:
            return _to_date(float(stripped))
        except ValueError:
            return stripped

    return datetime.fromtimestamp(_epoch_to_seconds(value), tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )


def _register_custom_functions(conn: sqlite3.Connection) -> None:
    """Register math and string functions that SQLite lacks."""
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None else None)
    conn.create_function("LN", 1, lambda x: math.log(x) if x is not None else None)
    conn.create_function("EXP", 1, lambda x: math.exp(x) if x is not None else None)
    conn.create_function(
        "POWER",
        2,
        lambda x, y: math.pow(x, y) if x is not None and y is not None else None,
    )
    conn.create_function(
        "CONCAT", -1, lambda *args: "".join(str(a) for a in args if a is not None)
    )
    conn.create_function("TO_TIMESTAMP", 1, _to_timestamp)
    conn.create_function("TO_DATE", 1, _to_date)
    # The typeshed _AggregateProtocol uses narrow int types in its stubs,
    # but sqlite3 accepts any scalar at runtime.  Use cast(Any, ...) so
    # the type checker does not reject the class.
    conn.create_aggregate("STDDEV", 1, cast(Any, _StddevAggregate))
    conn.create_aggregate("STDDEV_SAMP", 1, cast(Any, _StddevAggregate))
    conn.create_aggregate("CORR", 2, cast(Any, _CorrAggregate))


def _rewrite_count_filter(tree: exp.Expression) -> exp.Expression:
    """Rewrite COUNT(*) FILTER (WHERE cond) to SUM(CASE WHEN cond THEN 1 ELSE 0 END).

    Uses sqlglot AST traversal so it handles arbitrarily nested conditions
    (e.g. WHERE (a > 1 OR b < 2)) that a regex would break on.
    """
    for node in tree.find_all(exp.Filter):
        # Only rewrite COUNT(*) FILTER — leave other aggregates alone
        agg = node.this
        if not isinstance(agg, exp.Count):
            continue
        where_clause = node.expression
        # exp.Filter wraps a Where node whose .this is the actual condition
        condition = (
            where_clause.this if isinstance(where_clause, exp.Where) else where_clause
        )
        replacement = exp.Sum(
            this=exp.Case(
                ifs=[exp.If(this=condition, true=exp.Literal.number(1))],
                default=exp.Literal.number(0),
            )
        )
        node.replace(replacement)
    return tree


def _preprocess_sql(sql: str) -> str:
    """Normalize common SQL dialect differences to SQLite-compatible syntax.

    Uses sqlglot's AST-based dialect transpilation to handle:
      - ILIKE -> LIKE (via LOWER wrapping)
      - CAST(... AS VARCHAR/DOUBLE) -> CAST(... AS TEXT/REAL)
      - STRING_AGG(...) -> GROUP_CONCAT(...)
      - ANY_VALUE(x) -> MAX(x)
      - COUNT(*) FILTER (WHERE cond) -> SUM(CASE WHEN cond THEN 1 ELSE 0 END)

    All rewrites go through the AST so they correctly handle nested
    expressions, subqueries, and keywords inside string literals.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        assert isinstance(tree, exp.Expression)
        tree = _rewrite_count_filter(tree)
        sql = tree.sql(dialect="sqlite")
    except SQLParseError:
        pass  # fall through with original SQL; _validate_sql will catch errors
    return sql


def _infer_dtype_label(values: list) -> str:
    """Infer a human-readable type label from column values."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            return "Boolean"
        if isinstance(v, int):
            return "Int64"
        if isinstance(v, float):
            return "Float64"
        return "String"
    return "String"


class DataFrameStore:
    """In-memory store for Tables with SQL query capability."""

    def __init__(
        self,
        max_tables: int = DEFAULT_MAX_TABLES,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self._max_tables = max_tables
        self._max_rows = max_rows
        self._tables: dict[str, tuple[Table, float]] = {}

    def store(self, name: str, records: list[dict]) -> StoreSummary:
        """Store a list of flat dicts as a named Table.

        Args:
            name: SQL-safe table name.
            records: List of flat dictionaries (rows).

        Returns:
            StoreSummary with table info and a 5-row preview.

        Raises:
            ValueError: If name is invalid, limits are exceeded, or records are empty.
        """
        if not TABLE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid table name '{name}'. "
                "Must match [a-zA-Z_][a-zA-Z0-9_]{{0,62}}."
            )

        if name.lower() in _RESERVED_TABLE_NAMES:
            raise ValueError(
                f"Table name '{name}' is reserved. Please choose a different name."
            )

        self._evict_expired()

        # Allow overwriting an existing table without counting against the limit
        if name not in self._tables and len(self._tables) >= self._max_tables:
            raise ValueError(
                f"Table limit reached ({self._max_tables}). "
                "Drop a table before storing a new one."
            )

        if not records:
            raise ValueError("Cannot store an empty record set.")

        if len(records) > self._max_rows:
            raise ValueError(
                f"Too many rows ({len(records)}). Maximum is {self._max_rows}."
            )

        table = Table.from_records(records)
        self._check_duplicate_columns(table)
        self._tables[name] = (table, time.time())

        preview_table = table.head(5)
        preview_csv = preview_table.write_csv(max_cell_chars=PREVIEW_MAX_CELL_CHARS)

        return StoreSummary(
            table_name=name,
            row_count=len(table),
            columns=table.columns,
            preview=preview_csv,
        )

    def query(self, sql: str, max_cell_chars: int = 0) -> str:
        """Execute a SQL SELECT query across all stored tables.

        Args:
            sql: SQL query string.
            max_cell_chars: If > 0, truncate output cells whose string
                form exceeds this length with a visible marker.  0
                (default) leaves cells untouched.

        Returns:
            CSV string of the query result.
        """
        return self._execute_sql(sql).write_csv(max_cell_chars=max_cell_chars)

    def show_tables(self) -> str:
        """List all stored tables with metadata."""
        self._evict_expired()
        if not self._tables:
            return "No tables stored."

        now = time.time()
        lines = ["table_name,rows,columns,age_seconds"]
        for name, (table, ts) in self._tables.items():
            age = int(now - ts)
            lines.append(f"{name},{len(table)},{len(table.columns)},{age}")
        return "\n".join(lines)

    def describe_table(self, name: str) -> str:
        """Describe a single table's schema.

        Args:
            name: Table name.

        Returns:
            Column names, dtypes, and row count.
        """
        self._evict_expired()
        if name not in self._tables:
            raise ValueError(f"Table '{name}' not found.")
        table, _ts = self._tables[name]
        lines = [f"Table: {name} ({len(table)} rows)", "column,dtype"]
        for col_name in table.columns:
            dtype = _infer_dtype_label(table.data[col_name])
            lines.append(f"{col_name},{dtype}")
        return "\n".join(lines)

    def drop_table(self, name: str) -> str:
        """Remove a table from the store.

        Args:
            name: Table name.

        Returns:
            Confirmation message.
        """
        if name not in self._tables:
            raise ValueError(f"Table '{name}' not found.")
        del self._tables[name]
        return f"Table '{name}' dropped."

    def get_table(self, name: str) -> Table:
        """Retrieve a stored Table by name.

        Args:
            name: Table name.

        Returns:
            The stored Table.

        Raises:
            ValueError: If the table does not exist.
        """
        self._evict_expired()
        if name not in self._tables:
            raise ValueError(f"Table '{name}' not found.")
        table, _ts = self._tables[name]
        return table

    def store_table(self, name: str, table: Table) -> StoreSummary:
        """Store an existing Table under the given name.

        Args:
            name: SQL-safe table name.
            table: Table to store.

        Returns:
            StoreSummary with table info and a 5-row preview.

        Raises:
            ValueError: If name is invalid or limits are exceeded.
        """
        if not TABLE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid table name '{name}'. "
                "Must match [a-zA-Z_][a-zA-Z0-9_]{{0,62}}."
            )

        if name.lower() in _RESERVED_TABLE_NAMES:
            raise ValueError(
                f"Table name '{name}' is reserved. Please choose a different name."
            )

        self._evict_expired()

        if name not in self._tables and len(self._tables) >= self._max_tables:
            raise ValueError(
                f"Table limit reached ({self._max_tables}). "
                "Drop a table before storing a new one."
            )

        if len(table) > self._max_rows:
            raise ValueError(
                f"Too many rows ({len(table)}). Maximum is {self._max_rows}."
            )

        self._check_duplicate_columns(table)
        self._tables[name] = (table, time.time())

        preview_table = table.head(5)
        preview_csv = preview_table.write_csv(max_cell_chars=PREVIEW_MAX_CELL_CHARS)

        return StoreSummary(
            table_name=name,
            row_count=len(table),
            columns=table.columns,
            preview=preview_csv,
        )

    def query_table(self, sql: str) -> Table:
        """Execute a SQL SELECT query and return the result as a Table.

        Args:
            sql: SQL query string.

        Returns:
            Table of the query result.
        """
        return self._execute_sql(sql)

    @staticmethod
    def _check_duplicate_columns(table: Table) -> None:
        """Raise ValueError if the Table has duplicate column names."""
        cols = table.columns
        if len(cols) != len(set(cols)):
            seen: set[str] = set()
            dupes: list[str] = []
            for c in cols:
                if c in seen and c not in dupes:
                    dupes.append(c)
                seen.add(c)
            raise ValueError(
                f"DataFrame has duplicate column names: {dupes}. "
                "Rename columns to be unique before storing."
            )

    @staticmethod
    def _validate_sql(sql: str, allowed_tables: set[str]) -> str:
        """Validate that *sql* is a read-only SELECT over registered tables.

        Uses sqlglot to parse the SQL into an AST and structurally verify:

        1. It is a single SELECT / UNION / EXCEPT / INTERSECT statement.
        2. It contains no blocked functions (filesystem I/O, etc.).
        3. Every table reference resolves to either a CTE defined in the
           query or one of the *allowed_tables*.

        Args:
            sql: Raw SQL string.
            allowed_tables: Set of table names the query is permitted to
                reference (i.e. the keys of ``self._tables``).

        Returns:
            The cleaned SQL string.

        Raises:
            ValueError: If the query fails any of the above checks.
        """
        stripped = sql.strip().rstrip(";").strip()
        if not stripped:
            raise ValueError("Empty SQL query.")

        try:
            statements = sqlglot.parse(stripped, dialect="sqlite")
        except SQLParseError as exc:
            raise ValueError(f"SQL parse error: {exc}") from exc

        # Filter out None entries that sqlglot may return for trailing
        # semicolons / empty segments.
        statements = [s for s in statements if s is not None]

        if len(statements) != 1:
            raise ValueError(
                f"Exactly one SQL statement is allowed, got {len(statements)}."
            )

        stmt = statements[0]

        # Must be a read-only query.  exp.Query covers Select (including
        # WITH ... SELECT) and set operations (Union, Except, Intersect).
        if not isinstance(stmt, exp.Query):
            raise ValueError(
                f"Only SELECT queries are allowed. Got: {type(stmt).__name__}"
            )

        # ---- Function denylist -------------------------------------------
        for node in stmt.find_all(exp.Func):
            if isinstance(node, (exp.Anonymous, exp.AnonymousAggFunc)):
                # Anonymous nodes are functions sqlglot doesn't recognise —
                # block them unless they are our own custom functions.
                if node.name.lower() not in _CUSTOM_ANONYMOUS_FUNCTIONS:
                    raise ValueError(f"Function not allowed: {node.name}")
            # Safety net: check SQL name against the explicit denylist
            # even if sqlglot gives them a typed Func subclass.
            try:
                sql_name = type(node).sql_name().lower()
            except (NotImplementedError, AttributeError):
                continue
            if sql_name in _BLOCKED_FUNC_NAMES:
                raise ValueError(f"Function not allowed: {sql_name}")

        # ---- Table reference allowlist -----------------------------------
        # Collect CTE names defined within the query itself; these are valid
        # table references that do not need to exist in _tables.
        cte_names = {cte.alias for cte in stmt.find_all(exp.CTE)}

        for table_node in stmt.find_all(exp.Table):
            name = table_node.name
            # Table-valued functions produce Table nodes with an empty name;
            # skip those — the function check above already handles them.
            if not name:
                continue
            # Qualified references like schema.table must be rejected —
            # they are never registered user tables.
            if table_node.db or table_node.catalog:
                raise ValueError(
                    f"Schema-qualified table references are not allowed: "
                    f"{table_node.sql(dialect='sqlite')}"
                )
            if name not in allowed_tables and name not in cte_names:
                raise ValueError(
                    f"Table not found: '{name}'. "
                    f"Available tables: {sorted(allowed_tables) or '(none)'}"
                )

        return stripped

    def _execute_sql(self, sql: str) -> Table:
        """Execute SQL against stored tables using SQLite.

        An in-memory SQLite connection is created per query with
        ``trusted_schema = OFF`` for safety. A progress handler
        enforces the query timeout.

        Args:
            sql: SQL query string (must be a SELECT query).

        Returns:
            Table of the query result.
        """
        self._evict_expired()
        cleaned_sql = self._validate_sql(sql, set(self._tables.keys()))
        preprocessed = _preprocess_sql(cleaned_sql)

        conn = sqlite3.connect(":memory:")
        try:
            # ---- Phase 1: setup (needs write access) ----
            conn.execute("PRAGMA trusted_schema = OFF")
            conn.execute("PRAGMA cell_size_check = ON")
            _register_custom_functions(conn)

            for name, (table, _ts) in self._tables.items():
                _create_and_populate_table(conn, name, table)

            # ---- Phase 2: lock down before running user SQL ----
            conn.execute("PRAGMA query_only = ON")

            # Disable extension loading (no-op if already compiled out)
            if hasattr(conn, "enable_load_extension"):
                conn.enable_load_extension(False)

            # Python 3.12+ setconfig for additional hardening
            _setconfig = getattr(conn, "setconfig", None)
            if _setconfig is not None:
                for attr, val in (
                    ("SQLITE_DBCONFIG_DEFENSIVE", True),
                    ("SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION", False),
                    ("SQLITE_DBCONFIG_ENABLE_TRIGGER", False),
                    ("SQLITE_DBCONFIG_ENABLE_VIEW", False),
                    ("SQLITE_DBCONFIG_TRUSTED_SCHEMA", False),
                ):
                    flag = getattr(sqlite3, attr, None)
                    if flag is not None:
                        _setconfig(flag, val)

            # Authorizer: allowlist of permitted operations and functions
            conn.set_authorizer(_select_only_authorizer)

            # Timeout via progress handler
            deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

            def _check_timeout() -> int:
                if time.monotonic() > deadline:
                    return 1  # non-zero interrupts the query
                return 0

            conn.set_progress_handler(_check_timeout, 1000)

            try:
                cursor = conn.execute(preprocessed)
            except sqlite3.DatabaseError as exc:
                msg = str(exc)
                if "interrupted" in msg.lower():
                    raise TimeoutError(
                        f"Query exceeded {QUERY_TIMEOUT_SECONDS}s timeout."
                    ) from None
                if "not authorized" in msg.lower() or "authorization" in msg.lower():
                    raise ValueError(f"Query blocked by authorizer: {msg}") from exc
                # Convert "no such table" to a helpful message
                if "no such table" in msg:
                    raise ValueError(
                        f"Table not found in query. {msg}. "
                        f"Available tables: {sorted(self._tables.keys()) or '(none)'}"
                    ) from exc
                raise ValueError(f"SQL error: {msg}") from exc

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                # Return empty Table with correct column names
                return Table(columns, {col: [] for col in columns})

            # Build column-oriented dict
            col_data: dict[str, list] = {col: [] for col in columns}
            for row in rows:
                for col, val in zip(columns, row):
                    col_data[col].append(val)

            return Table(columns, col_data)
        finally:
            conn.close()

    def _evict_expired(self) -> None:
        """Remove tables that have exceeded their TTL."""
        now = time.time()
        expired = [
            name
            for name, (_table, ts) in self._tables.items()
            if now - ts > TTL_SECONDS
        ]
        for name in expired:
            del self._tables[name]
