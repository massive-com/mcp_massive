import csv
import io
import math
import re
import sqlite3
import time
from typing import Any, cast

import sqlglot
from sqlglot import exp
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

DEFAULT_MAX_TABLES = 50
DEFAULT_MAX_ROWS = 50_000
TTL_SECONDS = 3600
QUERY_TIMEOUT_SECONDS = 30

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
    return sqlite3.SQLITE_DENY


class Table:
    """Column-oriented in-memory table (replaces pl.DataFrame)."""

    def __init__(self, columns: list[str], data: dict[str, list]):
        if columns:
            lengths = {col: len(data[col]) for col in columns if col in data}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                raise ValueError(
                    f"All columns must have the same length, got inconsistent lengths: "
                    f"{dict(sorted(((col, length) for col, length in lengths.items()), key=lambda x: x[0]))}"
                )
        self.columns = columns
        self.data = data

    @classmethod
    def from_records(cls, records: list[dict]) -> "Table":
        """Build from list of flat dicts.

        Preserves insertion order of keys across all records.
        Fills missing keys with None.
        """
        if not records:
            return cls([], {})
        # Collect all keys in insertion order
        seen: set[str] = set()
        columns: list[str] = []
        for rec in records:
            for key in rec:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        data: dict[str, list] = {col: [] for col in columns}
        for rec in records:
            for col in columns:
                data[col].append(rec.get(col))
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

    def write_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.columns)
        for row in self.rows():
            writer.writerow(row)
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


def _create_and_populate_table(
    conn: sqlite3.Connection, name: str, table: Table
) -> None:
    """Create a SQLite table from a Table and bulk-insert rows."""
    cols = table.columns
    affinities = [_infer_sqlite_affinity(table.data[c]) for c in cols]
    col_defs = ", ".join(f'"{c}" {a}' for c, a in zip(cols, affinities))
    conn.execute(f'CREATE TABLE "{name}" ({col_defs})')
    if len(table) > 0:
        placeholders = ", ".join("?" for _ in cols)
        conn.executemany(
            f'INSERT INTO "{name}" VALUES ({placeholders})',
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
    # The typeshed _AggregateProtocol uses narrow int types in its stubs,
    # but sqlite3 accepts any scalar at runtime.  Use cast(Any, ...) so
    # the type checker does not reject the class.
    conn.create_aggregate("STDDEV", 1, cast(Any, _StddevAggregate))
    conn.create_aggregate("STDDEV_SAMP", 1, cast(Any, _StddevAggregate))


def _preprocess_sql(sql: str) -> str:
    """Normalize common SQL dialect differences to SQLite-compatible syntax."""
    # ILIKE -> LIKE (SQLite LIKE is case-insensitive for ASCII by default)
    sql = re.sub(r"\bILIKE\b", "LIKE", sql, flags=re.IGNORECASE)
    # CAST(... AS VARCHAR) -> CAST(... AS TEXT)
    sql = re.sub(r"\bVARCHAR\b", "TEXT", sql, flags=re.IGNORECASE)
    # CAST(... AS DOUBLE) -> CAST(... AS REAL)
    sql = re.sub(r"\bDOUBLE\b", "REAL", sql, flags=re.IGNORECASE)
    # STRING_AGG(expr, sep ORDER BY ...) -> GROUP_CONCAT(expr, sep)
    # This is a simplified rewrite — strips ORDER BY inside STRING_AGG
    sql = re.sub(
        r"\bSTRING_AGG\s*\(\s*([^,]+),\s*([^)]*?)\s+ORDER\s+BY\s+[^)]+\)",
        r"GROUP_CONCAT(\1, \2)",
        sql,
        flags=re.IGNORECASE,
    )
    # STRING_AGG without ORDER BY -> GROUP_CONCAT
    sql = re.sub(r"\bSTRING_AGG\b", "GROUP_CONCAT", sql, flags=re.IGNORECASE)
    # ANY_VALUE(x) -> MIN(x) — both return an arbitrary value from the group
    sql = re.sub(r"\bANY_VALUE\s*\(", "MIN(", sql, flags=re.IGNORECASE)
    # COUNT(*) FILTER (WHERE cond) -> SUM(CASE WHEN cond THEN 1 ELSE 0 END)
    sql = re.sub(
        r"\bCOUNT\s*\(\s*\*\s*\)\s*FILTER\s*\(\s*WHERE\s+(.+?)\)",
        r"SUM(CASE WHEN \1 THEN 1 ELSE 0 END)",
        sql,
        flags=re.IGNORECASE,
    )
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
        preview_csv = preview_table.write_csv()

        return StoreSummary(
            table_name=name,
            row_count=len(table),
            columns=table.columns,
            preview=preview_csv,
        )

    def query(self, sql: str) -> str:
        """Execute a SQL SELECT query across all stored tables.

        Args:
            sql: SQL query string.

        Returns:
            CSV string of the query result.
        """
        return self._execute_sql(sql).write_csv()

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

    # Keep old name as alias for backward compatibility during transition
    get_dataframe = get_table

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
        preview_csv = preview_table.write_csv()

        return StoreSummary(
            table_name=name,
            row_count=len(table),
            columns=table.columns,
            preview=preview_csv,
        )

    # Keep old name as alias for backward compatibility during transition
    store_dataframe = store_table

    def query_table(self, sql: str) -> Table:
        """Execute a SQL SELECT query and return the result as a Table.

        Args:
            sql: SQL query string.

        Returns:
            Table of the query result.
        """
        return self._execute_sql(sql)

    # Keep old name as alias for backward compatibility during transition
    query_df = query_table

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
        except sqlglot.errors.ParseError as exc:
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
                # block them since they may be dangerous extensions.
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
