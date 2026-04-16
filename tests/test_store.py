import time
from unittest.mock import patch

import math
import pytest

from mcp_massive.store import (
    DataFrameStore,
    Table,
    TABLE_NAME_RE,
    _infer_sqlite_affinity,
    _infer_dtype_label,
    _preprocess_sql,
)


# ---------------------------------------------------------------------------
# Table class unit tests
# ---------------------------------------------------------------------------


class TestTable:
    def test_from_records_basic(self):
        t = Table.from_records([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert t.columns == ["a", "b"]
        assert t["a"] == [1, 3]
        assert t["b"] == [2, 4]

    def test_from_records_missing_keys_filled_with_none(self):
        t = Table.from_records([{"a": 1}, {"b": 2}, {"a": 3, "b": 4}])
        assert t.columns == ["a", "b"]
        assert t["a"] == [1, None, 3]
        assert t["b"] == [None, 2, 4]

    def test_from_records_empty_list(self):
        t = Table.from_records([])
        assert t.columns == []
        assert len(t) == 0

    def test_from_records_preserves_casing(self):
        t = Table.from_records([{"Name": "Alice", "AGE": 30}])
        assert t.columns == ["Name", "AGE"]
        assert t["Name"] == ["Alice"]
        assert t["AGE"] == [30]

    def test_from_records_deduplicates_case_insensitive_columns(self):
        """Massive.com returns both T (ticker) and t (timestamp)."""
        t = Table.from_records([{"T": "AAPL", "v": 100.0, "t": 1704067200000}])
        assert t.columns == ["T", "v", "t_2"]
        assert t["T"] == ["AAPL"]
        assert t["v"] == [100.0]
        assert t["t_2"] == [1704067200000]

    def test_from_records_dedup_multiple_collisions(self):
        t = Table.from_records([{"a": 1, "A": 2, "A_2": 3}])
        # a preserved, A collides -> A_2, A_2 collides with A_2 -> A_2_2
        assert t.columns == ["a", "A_2", "A_2_2"]
        assert t["a"] == [1]
        assert t["A_2"] == [2]
        assert t["A_2_2"] == [3]

    def test_len(self):
        t = Table(["x"], {"x": [1, 2, 3]})
        assert len(t) == 3

    def test_len_empty(self):
        t = Table([], {})
        assert len(t) == 0

    def test_head_normal(self):
        t = Table(["x"], {"x": [1, 2, 3, 4, 5]})
        h = t.head(3)
        assert h["x"] == [1, 2, 3]
        assert len(h) == 3

    def test_head_exceeds_length(self):
        t = Table(["x"], {"x": [1, 2]})
        h = t.head(10)
        assert h["x"] == [1, 2]

    def test_head_zero(self):
        t = Table(["x"], {"x": [1, 2]})
        h = t.head(0)
        assert len(h) == 0

    def test_rows(self):
        t = Table(["a", "b"], {"a": [1, 2], "b": [3, 4]})
        assert t.rows() == [(1, 3), (2, 4)]

    def test_write_csv_basic(self):
        t = Table(["a", "b"], {"a": [1, 2], "b": ["x", "y"]})
        csv = t.write_csv()
        assert csv == "a,b\n1,x\n2,y\n"

    def test_write_csv_with_special_characters(self):
        t = Table(["name"], {"name": ["has,comma", 'has"quote', "normal"]})
        csv = t.write_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        # csv module should quote fields with commas and escape quotes
        assert '"has,comma"' in lines[1]
        assert '"' in lines[2]  # quote should be escaped

    def test_write_csv_with_none(self):
        t = Table(["x"], {"x": [1, None, 3]})
        result = t.write_csv()
        # None renders as empty string in CSV
        assert "1" in result
        assert "3" in result
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_get_column_missing_raises(self):
        t = Table(["a"], {"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            t.get_column("missing")

    def test_with_column_adds_new(self):
        t = Table(["a"], {"a": [1, 2]})
        t2 = t.with_column("b", [3, 4])
        assert t2.columns == ["a", "b"]
        assert t2["b"] == [3, 4]
        # Original unchanged
        assert "b" not in t.data

    def test_with_column_replaces_existing(self):
        t = Table(["a", "b"], {"a": [1, 2], "b": [3, 4]})
        t2 = t.with_column("b", [5, 6])
        assert t2.columns == ["a", "b"]
        assert t2["b"] == [5, 6]
        # Original unchanged
        assert t["b"] == [3, 4]

    def test_sort_basic(self):
        t = Table(["x", "y"], {"x": [3, 1, 2], "y": ["c", "a", "b"]})
        s = t.sort("x")
        assert s["x"] == [1, 2, 3]
        assert s["y"] == ["a", "b", "c"]

    def test_sort_with_none_values(self):
        t = Table(["x"], {"x": [3, None, 1, None, 2]})
        s = t.sort("x")
        # None values sort last
        assert s["x"] == [1, 2, 3, None, None]

    def test_sort_mixed_types_does_not_crash(self):
        """Sort with mixed types should not raise TypeError."""
        t = Table(["x"], {"x": [1, "b", 2, "a"]})
        # Should not crash — mixed types are handled gracefully
        s = t.sort("x")
        assert len(s) == 4

    def test_equals_true(self):
        t1 = Table(["a", "b"], {"a": [1], "b": [2]})
        t2 = Table(["a", "b"], {"a": [1], "b": [2]})
        assert t1.equals(t2)

    def test_equals_different_data(self):
        t1 = Table(["a"], {"a": [1]})
        t2 = Table(["a"], {"a": [2]})
        assert not t1.equals(t2)

    def test_equals_different_column_order(self):
        t1 = Table(["a", "b"], {"a": [1], "b": [2]})
        t2 = Table(["b", "a"], {"a": [1], "b": [2]})
        assert not t1.equals(t2)

    def test_getitem(self):
        t = Table(["x"], {"x": [10, 20]})
        assert t["x"] == [10, 20]

    def test_mismatched_column_lengths_raises(self):
        with pytest.raises(ValueError, match="inconsistent lengths"):
            Table(["a", "b"], {"a": [1, 2, 3], "b": [4, 5]})

    def test_consistent_column_lengths_ok(self):
        """No error when all columns have the same length."""
        t = Table(["a", "b"], {"a": [1, 2], "b": [3, 4]})
        assert len(t) == 2


# ---------------------------------------------------------------------------
# Type inference helpers
# ---------------------------------------------------------------------------


class TestInferSqliteAffinity:
    def test_int(self):
        assert _infer_sqlite_affinity([1, 2, 3]) == "INTEGER"

    def test_float(self):
        assert _infer_sqlite_affinity([1.0, 2.0]) == "REAL"

    def test_bool(self):
        assert _infer_sqlite_affinity([True, False]) == "INTEGER"

    def test_string(self):
        assert _infer_sqlite_affinity(["a", "b"]) == "TEXT"

    def test_all_none_defaults_to_text(self):
        assert _infer_sqlite_affinity([None, None]) == "TEXT"

    def test_leading_none_skipped(self):
        assert _infer_sqlite_affinity([None, None, 42]) == "INTEGER"

    def test_bool_before_int(self):
        """Bool is a subclass of int; ensure bool is detected first."""
        assert _infer_sqlite_affinity([True, 1, 2]) == "INTEGER"


class TestInferDtypeLabel:
    def test_int(self):
        assert _infer_dtype_label([1, 2]) == "Int64"

    def test_float(self):
        assert _infer_dtype_label([1.0]) == "Float64"

    def test_bool(self):
        assert _infer_dtype_label([True]) == "Boolean"

    def test_string(self):
        assert _infer_dtype_label(["x"]) == "String"

    def test_all_none(self):
        assert _infer_dtype_label([None, None]) == "String"

    def test_leading_none(self):
        assert _infer_dtype_label([None, 3.14]) == "Float64"


class TestTableNameValidation:
    """Test the table name regex and store validation."""

    @pytest.mark.parametrize(
        "name",
        ["prices", "my_table", "_private", "A", "a1", "table_123", "a" * 63],
    )
    def test_valid_names(self, name):
        assert TABLE_NAME_RE.match(name)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "1starts_with_digit",
            "has space",
            "has-dash",
            "has.dot",
            "a" * 64,  # too long
            "table; DROP TABLE x",  # SQL injection
            "table'name",
            'table"name',
        ],
    )
    def test_invalid_names(self, name):
        assert not TABLE_NAME_RE.match(name)


class TestDataFrameStore:
    def _sample_records(self, n=10):
        return [
            {"ticker": f"T{i}", "price": float(i), "volume": i * 100} for i in range(n)
        ]

    def test_store_and_retrieve(self):
        s = DataFrameStore()
        summary = s.store("prices", self._sample_records(5))
        assert summary.table_name == "prices"
        assert summary.row_count == 5
        assert "ticker" in summary.columns
        assert "price" in summary.columns
        assert "volume" in summary.columns
        assert len(summary.preview.strip().split("\n")) <= 6  # header + 5 rows

    def test_store_overwrite_existing(self):
        s = DataFrameStore()
        s.store("t", self._sample_records(3))
        summary = s.store("t", self._sample_records(7))
        assert summary.row_count == 7

    def test_store_invalid_name(self):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="Invalid table name"):
            s.store("1bad", self._sample_records(1))

    def test_store_empty_records(self):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="empty record set"):
            s.store("t", [])

    def test_store_max_tables(self):
        s = DataFrameStore(max_tables=5)
        for i in range(5):
            s.store(f"t{i}", [{"x": 1}])
        with pytest.raises(ValueError, match="Table limit reached"):
            s.store("one_more", [{"x": 1}])

    def test_store_max_tables_overwrite_ok(self):
        s = DataFrameStore(max_tables=5)
        for i in range(5):
            s.store(f"t{i}", [{"x": 1}])
        # Overwriting an existing table should work
        s.store("t0", [{"x": 2}])

    def test_store_default_max_tables(self):
        s = DataFrameStore()
        assert s._max_tables == 50

    def test_store_custom_max_rows(self):
        s = DataFrameStore(max_rows=10)
        with pytest.raises(ValueError, match="Too many rows"):
            s.store("big", [{"x": i} for i in range(11)])

    def test_store_max_rows(self):
        s = DataFrameStore()
        records = [{"x": i} for i in range(50_001)]
        with pytest.raises(ValueError, match="Too many rows"):
            s.store("big", records)

    def test_store_max_rows_boundary(self):
        s = DataFrameStore()
        records = [{"x": i} for i in range(50_000)]
        summary = s.store("big", records)
        assert summary.row_count == 50_000

    def test_preview_capped_at_5(self):
        s = DataFrameStore()
        summary = s.store("t", self._sample_records(100))
        # Preview CSV: header + up to 5 data rows
        lines = summary.preview.strip().split("\n")
        assert len(lines) == 6  # header + 5

    def test_preview_fewer_than_5(self):
        s = DataFrameStore()
        summary = s.store("t", self._sample_records(2))
        lines = summary.preview.strip().split("\n")
        assert len(lines) == 3  # header + 2

    def test_show_tables_empty(self):
        s = DataFrameStore()
        assert s.show_tables() == "No tables stored."

    def test_show_tables(self):
        s = DataFrameStore()
        s.store("prices", [{"x": 1}])
        s.store("volume", [{"a": 1}, {"a": 2}])
        result = s.show_tables()
        lines = result.strip().split("\n")
        assert lines[0] == "table_name,rows,columns,age_seconds"
        # Parse data lines into a dict keyed by table name.
        rows = {line.split(",")[0]: line.split(",") for line in lines[1:]}
        assert "prices" in rows
        assert rows["prices"][1] == "1"  # 1 row
        assert rows["prices"][2] == "1"  # 1 column
        assert "volume" in rows
        assert rows["volume"][1] == "2"  # 2 rows
        assert rows["volume"][2] == "1"  # 1 column

    def test_describe_table(self):
        s = DataFrameStore()
        s.store("t", [{"ticker": "AAPL", "price": 150.0}])
        result = s.describe_table("t")
        assert "Table: t (1 rows)" in result
        assert "ticker" in result
        assert "price" in result

    def test_describe_missing_table(self):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="not found"):
            s.describe_table("nope")

    def test_drop_table(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}])
        result = s.drop_table("t")
        assert "dropped" in result
        assert s.show_tables() == "No tables stored."

    def test_drop_missing_table(self):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="not found"):
            s.drop_table("nope")

    def test_ttl_eviction(self):
        s = DataFrameStore()
        s.store("old", [{"x": 1}])
        # Manually backdate the timestamp
        df, _ts = s._tables["old"]
        s._tables["old"] = (df, time.time() - 3601)
        s._evict_expired()
        assert "old" not in s._tables

    def test_ttl_eviction_keeps_fresh(self):
        s = DataFrameStore()
        s.store("fresh", [{"x": 1}])
        s._evict_expired()
        assert "fresh" in s._tables

    def test_eviction_before_query(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}])
        df, _ts = s._tables["t"]
        s._tables["t"] = (df, time.time() - 3601)
        # Query should evict and fail to find the table
        with pytest.raises(Exception):
            s.query("SELECT * FROM t")


class TestStoreQuery:
    def test_simple_select(self):
        s = DataFrameStore()
        s.store("t", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        df = s.query_table("SELECT * FROM t")
        assert df.columns == ["a", "b"]
        assert len(df) == 2
        assert df["a"] == [1, 3]
        assert df["b"] == [2, 4]

    def test_query_returns_csv(self):
        """Verify that query() returns valid CSV with header and data rows."""
        s = DataFrameStore()
        s.store("t", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        csv = s.query("SELECT * FROM t")
        lines = csv.strip().split("\n")
        assert lines[0] == "a,b"
        assert lines[1] == "1,2"
        assert lines[2] == "3,4"

    def test_where_clause(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}, {"x": 3}])
        df = s.query_table("SELECT * FROM t WHERE x > 1")
        assert len(df) == 2
        assert sorted(df["x"]) == [2, 3]

    def test_join_across_tables(self):
        s = DataFrameStore()
        s.store(
            "prices",
            [{"ticker": "AAPL", "price": 150.0}, {"ticker": "GOOG", "price": 2800.0}],
        )
        s.store(
            "volume", [{"ticker": "AAPL", "vol": 1000}, {"ticker": "GOOG", "vol": 500}]
        )
        df = s.query_table(
            "SELECT p.ticker, p.price, v.vol "
            "FROM prices p JOIN volume v ON p.ticker = v.ticker "
            "ORDER BY p.ticker"
        )
        assert len(df) == 2
        assert df["ticker"] == ["AAPL", "GOOG"]
        assert df["price"] == [150.0, 2800.0]
        assert df["vol"] == [1000, 500]

    def test_aggregation(self):
        s = DataFrameStore()
        s.store("t", [{"g": "a", "v": 10}, {"g": "a", "v": 20}, {"g": "b", "v": 30}])
        df = s.query_table("SELECT g, SUM(v) AS total FROM t GROUP BY g ORDER BY g")
        assert len(df) == 2
        assert df["g"] == ["a", "b"]
        assert df["total"] == [30, 30]

    def test_empty_result(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}])
        df = s.query_table("SELECT * FROM t WHERE x > 100")
        assert len(df) == 0

    def test_invalid_sql(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}])
        with pytest.raises(Exception):
            s.query("NOT VALID SQL AT ALL")

    def test_count(self):
        s = DataFrameStore()
        s.store("t", [{"x": i} for i in range(10)])
        df = s.query_table("SELECT COUNT(*) AS cnt FROM t")
        assert df["cnt"][0] == 10


class TestGetDataFrame:
    def test_get_existing(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1, "y": 2}])
        df = s.get_table("t")
        assert df.columns == ["x", "y"]
        assert len(df) == 1

    def test_get_missing(self):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="not found"):
            s.get_table("nope")


class TestStoreTable:
    def test_store_and_retrieve(self):
        s = DataFrameStore()
        tbl = Table(["a", "b"], {"a": [1, 2, 3], "b": [4, 5, 6]})
        summary = s.store_table("t", tbl)
        assert summary.table_name == "t"
        assert summary.row_count == 3
        assert "a" in summary.columns

    def test_store_overwrites(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}])
        tbl = Table(["x"], {"x": [10, 20]})
        summary = s.store_table("t", tbl)
        assert summary.row_count == 2

    def test_store_invalid_name(self):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="Invalid table name"):
            s.store_table("1bad", Table(["x"], {"x": [1]}))


class TestQueryTable:
    def test_returns_table(self):
        s = DataFrameStore()
        s.store("t", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        tbl = s.query_table("SELECT * FROM t WHERE a > 1")
        assert isinstance(tbl, Table)
        assert len(tbl) == 1
        assert tbl["a"][0] == 3

    def test_empty_result(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}])
        tbl = s.query_table("SELECT * FROM t WHERE x > 100")
        assert isinstance(tbl, Table)
        assert len(tbl) == 0


class TestDuplicateColumnGuardrails:
    def test_unaliased_duplicate_expressions_succeed(self):
        """SQLite auto-aliases duplicate expressions like min(x), max(x)."""
        s = DataFrameStore()
        s.store("t", [{"x": 1, "g": "a"}, {"x": 2, "g": "a"}, {"x": 3, "g": "b"}])
        df = s.query_table("SELECT MIN(x), MAX(x) FROM t")
        assert len(df) == 1

    def test_aliased_expressions_work(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}, {"x": 3}])
        df = s.query_table("SELECT MIN(x) AS min_x, MAX(x) AS max_x FROM t")
        assert df.columns == ["min_x", "max_x"]
        assert df["min_x"][0] == 1
        assert df["max_x"][0] == 3

    def test_check_duplicate_columns_raises(self):
        s = DataFrameStore()
        tbl = Table(["a", "a"], {"a": [1, 2]})
        with pytest.raises(ValueError, match="duplicate column names"):
            s._check_duplicate_columns(tbl)

    def test_check_duplicate_columns_passes_for_unique(self):
        s = DataFrameStore()
        tbl = Table(["a", "b", "c"], {"a": [1], "b": [2], "c": [3]})
        # Should not raise
        s._check_duplicate_columns(tbl)

    def test_store_and_query_case_insensitive_columns(self):
        """End-to-end: store records with T/t columns, query via SQL."""
        s = DataFrameStore()
        records = [
            {
                "T": "AAPL",
                "v": 45000000.0,
                "vw": 150.5,
                "o": 149.0,
                "c": 151.0,
                "h": 152.0,
                "l": 148.0,
                "t": 1704067200000,
                "n": 500000,
            },
        ]
        result = s.store("prices", records)
        assert result.row_count == 1

        csv = s.query("SELECT T, v, t_2 FROM prices")
        assert csv == "T,v,t_2\nAAPL,45000000.0,1704067200000\n"


class TestScalarSubqueryRewrite:
    def test_scalar_subquery_in_select_is_rewritten(self):
        s = DataFrameStore()
        s.store("t", [{"t": 1, "c": 10.0}, {"t": 2, "c": 20.0}])
        df = s.query_table(
            "SELECT (SELECT c FROM t ORDER BY t ASC LIMIT 1) AS first_close,"
            " (SELECT c FROM t ORDER BY t DESC LIMIT 1) AS last_close FROM t"
        )
        assert df["first_close"][0] == 10.0
        assert df["last_close"][0] == 20.0

    def test_scalar_subquery_with_aggregates(self):
        s = DataFrameStore()
        s.store("t", [{"t": 1, "c": 10.0}, {"t": 2, "c": 20.0}, {"t": 3, "c": 30.0}])
        df = s.query_table(
            "SELECT COUNT(*) AS n,"
            " (SELECT c FROM t ORDER BY t ASC LIMIT 1) AS first_close,"
            " (SELECT c FROM t ORDER BY t DESC LIMIT 1) AS last_close"
            " FROM t"
        )
        assert df["n"][0] == 3
        assert df["first_close"][0] == 10.0
        assert df["last_close"][0] == 30.0

    def test_scalar_subquery_with_where(self):
        s = DataFrameStore()
        s.store("t", [{"t": 1, "c": 10.0}, {"t": 2, "c": 20.0}, {"t": 3, "c": 30.0}])
        df = s.query_table(
            "SELECT 'X' AS ticker,"
            " (SELECT c FROM t WHERE t >= 2 ORDER BY t ASC LIMIT 1) AS first_close,"
            " COUNT(*) AS n"
            " FROM t WHERE t >= 2"
        )
        assert df["first_close"][0] == 20.0
        assert df["n"][0] == 2

    def test_scalar_subquery_complex_aggregates(self):
        """Matches the exact pattern LLMs generate: string literals + STDDEV + CASE."""
        s = DataFrameStore()
        s.store(
            "prices",
            [
                {"t": 1, "c": 100.0, "daily_return": 0.01},
                {"t": 2, "c": 105.0, "daily_return": -0.02},
                {"t": 3, "c": 110.0, "daily_return": 0.03},
            ],
        )
        df = s.query_table(
            "SELECT 'TICK' AS ticker, '5Y' AS period,"
            " (SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_close,"
            " (SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS last_close,"
            " COUNT(*) AS n_days,"
            " STDDEV(daily_return) * SQRT(252.0) AS volatility,"
            " SQRT(AVG(CASE WHEN daily_return < 0"
            " THEN daily_return * daily_return ELSE 0.0 END))"
            " * SQRT(252.0) AS downside_dev"
            " FROM prices WHERE daily_return IS NOT NULL"
        )
        assert df["ticker"][0] == "TICK"
        assert df["first_close"][0] == 100.0
        assert df["last_close"][0] == 110.0
        assert df["n_days"][0] == 3
        # volatility = STDDEV([0.01,-0.02,0.03]) * SQRT(252) ≈ 0.39950
        assert abs(df["volatility"][0] - 0.3994996871087636) < 1e-10
        # downside_dev = SQRT(AVG([0, 0.0004, 0])) * SQRT(252) ≈ 0.18330
        assert abs(df["downside_dev"][0] - 0.18330302779823363) < 1e-10

    def test_scalar_subquery_with_existing_cte(self):
        """Query that already has a WITH clause plus scalar subqueries."""
        s = DataFrameStore()
        s.store("t", [{"t": 1, "c": 10.0}, {"t": 2, "c": 20.0}, {"t": 3, "c": 30.0}])
        df = s.query_table(
            "WITH filtered AS (SELECT * FROM t WHERE c > 5) "
            "SELECT COUNT(*) AS n,"
            " (SELECT c FROM filtered ORDER BY t ASC LIMIT 1) AS first_c"
            " FROM filtered"
        )
        assert df["n"][0] == 3
        assert df["first_c"][0] == 10.0

    def test_scalar_subquery_with_nested_parens(self):
        """Subquery expression contains nested function calls with parens."""
        s = DataFrameStore()
        s.store("t", [{"t": 1, "v": 4.0}, {"t": 2, "v": 9.0}])
        df = s.query_table(
            "SELECT (SELECT SQRT(v) FROM t ORDER BY t ASC LIMIT 1) AS root_v,"
            " COUNT(*) AS n FROM t"
        )
        assert df["root_v"][0] == 2.0
        assert df["n"][0] == 2

    def test_scalar_subquery_with_string_containing_select(self):
        """String literal containing 'SELECT' should not confuse the parser."""
        s = DataFrameStore()
        s.store("t", [{"t": 1, "c": 10.0}, {"t": 2, "c": 20.0}])
        df = s.query_table(
            "SELECT 'not a SELECT' AS label,"
            " (SELECT c FROM t ORDER BY t ASC LIMIT 1) AS first_c"
            " FROM t"
        )
        assert df["label"][0] == "not a SELECT"
        assert df["first_c"][0] == 10.0

    def test_no_rewrite_for_from_subquery(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        df = s.query_table("SELECT * FROM (SELECT x, y FROM t WHERE x > 1) AS sub")
        assert len(df) == 1
        assert df["x"][0] == 3
        assert df["y"][0] == 4

    def test_no_rewrite_for_where_subquery(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}, {"x": 3}])
        df = s.query_table("SELECT * FROM t WHERE x IN (SELECT x FROM t WHERE x > 1)")
        assert sorted(df["x"]) == [2, 3]

    def test_no_rewrite_for_plain_query(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}])
        df = s.query_table("SELECT SUM(x) AS total FROM t")
        assert df["total"][0] == 3

    def test_scalar_subquery_arithmetic_in_cte(self):
        """Subqueries used in arithmetic inside a CTE body."""
        s = DataFrameStore()
        s.store("t", [{"t": 1, "c": 100.0}, {"t": 2, "c": 120.0}, {"t": 3, "c": 150.0}])
        df = s.query_table(
            "WITH stats AS ("
            " SELECT (SELECT c FROM t ORDER BY t DESC LIMIT 1)"
            " / (SELECT c FROM t ORDER BY t ASC LIMIT 1) - 1 AS total_return,"
            " COUNT(*) AS n"
            " FROM t"
            ") SELECT ROUND(total_return * 100, 2) AS pct, n FROM stats"
        )
        assert df["pct"][0] == 50.0  # (150/100 - 1) * 100
        assert df["n"][0] == 3

    def test_scalar_subquery_arithmetic_in_cte_with_aggregates(self):
        """Full realistic pattern: CTE with subquery arithmetic + STDDEV + CASE."""
        s = DataFrameStore()
        s.store(
            "prices",
            [
                {"t": 1, "c": 100.0, "daily_return": 0.01},
                {"t": 2, "c": 105.0, "daily_return": -0.02},
                {"t": 3, "c": 110.0, "daily_return": 0.03},
            ],
        )
        df = s.query_table(
            "WITH stats AS ("
            " SELECT 'TICK' AS ticker,"
            " (SELECT c FROM prices ORDER BY t DESC LIMIT 1)"
            " / (SELECT c FROM prices ORDER BY t ASC LIMIT 1) - 1 AS total_return,"
            " COUNT(*) AS n,"
            " STDDEV(daily_return) AS vol,"
            " SQRT(AVG(CASE WHEN daily_return < 0"
            " THEN daily_return * daily_return ELSE 0 END)) AS downside"
            " FROM prices WHERE daily_return IS NOT NULL"
            ") SELECT ticker, ROUND(total_return * 100, 2) AS pct,"
            " ROUND(vol * SQRT(252) * 100, 2) AS ann_vol FROM stats"
        )
        assert df["ticker"][0] == "TICK"
        assert df["pct"][0] == 10.0  # (110/100 - 1) * 100
        # ann_vol = ROUND(STDDEV([0.01,-0.02,0.03]) * SQRT(252) * 100, 2)
        assert df["ann_vol"][0] == 39.95
        assert "n" not in df.columns  # n not selected in outer


class TestStddevAggregate:
    """Verify STDDEV / STDDEV_SAMP custom aggregates return correct values."""

    def test_stddev_correct_value(self):
        s = DataFrameStore()
        s.store(
            "t",
            [
                {"v": 2.0},
                {"v": 4.0},
                {"v": 4.0},
                {"v": 4.0},
                {"v": 5.0},
                {"v": 5.0},
                {"v": 7.0},
                {"v": 9.0},
            ],
        )
        df = s.query_table("SELECT STDDEV(v) AS sd FROM t")
        # Sample stddev of [2,4,4,4,5,5,7,9]: mean=5, var=32/7, sd≈2.1381

        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        expected = math.sqrt(sum((x - 5.0) ** 2 for x in vals) / 7)
        assert abs(df["sd"][0] - expected) < 1e-10

    def test_stddev_samp_same_as_stddev(self):
        s = DataFrameStore()
        s.store("t", [{"v": 1.0}, {"v": 3.0}, {"v": 5.0}])
        df = s.query_table("SELECT STDDEV(v) AS sd, STDDEV_SAMP(v) AS ss FROM t")
        assert abs(df["sd"][0] - df["ss"][0]) < 1e-15

    def test_stddev_single_row_returns_null(self):
        s = DataFrameStore()
        s.store("t", [{"v": 42.0}])
        df = s.query_table("SELECT STDDEV(v) AS sd FROM t")
        assert df["sd"][0] is None

    def test_stddev_no_rows_returns_null(self):
        s = DataFrameStore()
        s.store("t", [{"v": 1.0}])
        df = s.query_table("SELECT STDDEV(v) AS sd FROM t WHERE v > 999")
        assert df["sd"][0] is None

    def test_stddev_ignores_nulls(self):
        s = DataFrameStore()
        s.store("t", [{"v": 1.0}, {"v": None}, {"v": 3.0}])
        df = s.query_table("SELECT STDDEV(v) AS sd FROM t")
        # Sample stddev of [1, 3] with n-1 = sqrt((1+1)/1) = sqrt(2)
        import math

        expected = math.sqrt(2.0)
        assert abs(df["sd"][0] - expected) < 1e-10


class TestRewriteQueryCorrectness:
    """Verify that scalar-subquery rewriting produces semantically correct results.

    Each test compares rewritten query output against known-correct values to
    catch regressions where the rewriter silently changes query semantics.
    """

    def _store(self):
        s = DataFrameStore()
        s.store(
            "prices",
            [
                {"t": 1, "c": 100.0, "v": 1000},
                {"t": 2, "c": 110.0, "v": 1500},
                {"t": 3, "c": 105.0, "v": 1200},
                {"t": 4, "c": 120.0, "v": 800},
                {"t": 5, "c": 115.0, "v": 900},
            ],
        )
        return s

    # ---- Row count preservation ----

    def test_row_count_not_inflated_by_cross_join(self):
        """CROSS JOIN with a single-row CTE must not multiply rows."""
        s = self._store()
        df = s.query_table(
            "SELECT t, c, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c "
            "FROM prices"
        )
        assert len(df) == 5, f"Expected 5 rows, got {len(df)}"

    def test_row_count_preserved_with_where(self):
        """Filtering + scalar subquery must not change the filtered row count."""
        s = self._store()
        df = s.query_table(
            "SELECT t, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c "
            "FROM prices WHERE t >= 3"
        )
        assert len(df) == 3, f"Expected 3 rows (t=3,4,5), got {len(df)}"

    def test_row_count_preserved_with_multiple_subqueries(self):
        """Multiple scalar subqueries should not compound row inflation."""
        s = self._store()
        df = s.query_table(
            "SELECT t, c, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c, "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS last_c, "
            "(SELECT MIN(v) FROM prices) AS min_v "
            "FROM prices"
        )
        assert len(df) == 5

    # ---- Exact scalar subquery values ----

    def test_first_and_last_values_are_correct(self):
        s = self._store()
        df = s.query_table(
            "SELECT "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c, "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS last_c "
            "FROM prices"
        )
        assert df["first_c"][0] == 100.0
        assert df["last_c"][0] == 115.0

    def test_scalar_min_max_values(self):
        s = self._store()
        df = s.query_table(
            "SELECT "
            "(SELECT MIN(c) FROM prices) AS min_c, "
            "(SELECT MAX(c) FROM prices) AS max_c "
            "FROM prices"
        )
        assert df["min_c"][0] == 100.0
        assert df["max_c"][0] == 120.0

    # ---- Aggregation correctness ----

    def test_count_not_inflated(self):
        """COUNT(*) alongside scalar subqueries must reflect actual row count."""
        s = self._store()
        df = s.query_table(
            "SELECT COUNT(*) AS n, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c "
            "FROM prices"
        )
        assert len(df) == 1
        assert df["n"][0] == 5
        assert df["first_c"][0] == 100.0

    def test_sum_not_inflated(self):
        """SUM must not be multiplied by CROSS JOIN fan-out."""
        s = self._store()
        df = s.query_table(
            "SELECT SUM(v) AS total_v, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c "
            "FROM prices"
        )
        assert df["total_v"][0] == 5400  # 1000+1500+1200+800+900
        assert df["first_c"][0] == 100.0

    def test_avg_not_affected(self):
        """AVG must not be affected by the rewrite."""
        s = self._store()
        df = s.query_table(
            "SELECT AVG(c) AS avg_c, "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS last_c "
            "FROM prices"
        )
        expected_avg = (100.0 + 110.0 + 105.0 + 120.0 + 115.0) / 5
        assert abs(df["avg_c"][0] - expected_avg) < 0.001
        assert df["last_c"][0] == 115.0

    def test_count_with_where_and_subquery(self):
        s = self._store()
        df = s.query_table(
            "SELECT COUNT(*) AS n, "
            "(SELECT MIN(c) FROM prices) AS min_c "
            "FROM prices WHERE t >= 3"
        )
        assert df["n"][0] == 3  # rows t=3,4,5
        assert df["min_c"][0] == 100.0  # min over all rows (subquery is unfiltered)

    # ---- Arithmetic with scalar subqueries ----

    def test_return_calculation(self):
        """(last / first - 1) must produce the correct return."""
        s = self._store()
        df = s.query_table(
            "SELECT "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) "
            "/ (SELECT c FROM prices ORDER BY t ASC LIMIT 1) - 1 AS total_return "
            "FROM prices"
        )
        expected = 115.0 / 100.0 - 1  # 0.15
        assert abs(df["total_return"][0] - expected) < 0.001

    def test_arithmetic_in_cte_body(self):
        """Arithmetic involving subqueries inside a CTE must compute correctly."""
        s = self._store()
        df = s.query_table(
            "WITH stats AS ("
            " SELECT (SELECT c FROM prices ORDER BY t DESC LIMIT 1)"
            " / (SELECT c FROM prices ORDER BY t ASC LIMIT 1) - 1 AS ret,"
            " COUNT(*) AS n"
            " FROM prices"
            ") SELECT ROUND(ret * 100, 1) AS pct, n FROM stats"
        )
        assert df["pct"][0] == 15.0  # (115/100 - 1) * 100
        assert df["n"][0] == 5

    # ---- GROUP BY correctness ----

    def test_group_by_with_scalar_subquery(self):
        """GROUP BY must still partition correctly when a scalar subquery is present."""
        s = DataFrameStore()
        s.store(
            "data",
            [
                {"g": "a", "v": 10},
                {"g": "a", "v": 20},
                {"g": "b", "v": 30},
                {"g": "b", "v": 40},
                {"g": "b", "v": 50},
            ],
        )
        df = s.query_table(
            "SELECT g, SUM(v) AS total, "
            "(SELECT MIN(v) FROM data) AS global_min "
            "FROM data GROUP BY g"
        )
        # Sort in Python to avoid conflating with the ORDER BY bug.
        df = df.sort("g")
        assert len(df) == 2
        assert df["g"] == ["a", "b"]
        assert df["total"] == [30, 120]
        assert df["global_min"] == [10, 10]

    def test_group_by_not_broken_in_cte(self):
        """GROUP BY inside a CTE with scalar subqueries must group correctly."""
        s = DataFrameStore()
        s.store(
            "data",
            [
                {"g": "x", "v": 1},
                {"g": "x", "v": 2},
                {"g": "y", "v": 10},
            ],
        )
        df = s.query_table(
            "WITH agg AS ("
            " SELECT g, SUM(v) AS total,"
            " (SELECT MAX(v) FROM data) AS peak"
            " FROM data GROUP BY g"
            ") SELECT g, total, peak FROM agg ORDER BY g"
        )
        assert df["g"] == ["x", "y"]
        assert df["total"] == [3, 10]
        assert df["peak"] == [10, 10]

    # ---- String literal + mixed expression correctness ----

    def test_string_literal_preserved(self):
        """String literals in SELECT must survive rewriting intact."""
        s = self._store()
        df = s.query_table(
            "SELECT 'hello' AS label, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c "
            "FROM prices"
        )
        assert df["label"][0] == "hello"
        assert df["first_c"][0] == 100.0

    def test_multiple_string_literals_and_subqueries(self):
        s = self._store()
        df = s.query_table(
            "SELECT 'AAPL' AS ticker, '5Y' AS period, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c, "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS last_c, "
            "COUNT(*) AS n "
            "FROM prices"
        )
        assert df["ticker"][0] == "AAPL"
        assert df["period"][0] == "5Y"
        assert df["first_c"][0] == 100.0
        assert df["last_c"][0] == 115.0
        assert df["n"][0] == 5

    # ---- Equivalence: rewritten vs hand-written CTE ----

    def test_rewritten_matches_manual_cte(self):
        """The rewritten query must produce the same result as a hand-written CTE."""
        s = self._store()
        # Query that triggers rewriting
        rewritten_df = s.query_table(
            "SELECT COUNT(*) AS n, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c, "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS last_c "
            "FROM prices"
        )
        # Equivalent hand-written CTE (no rewriting needed)
        manual_df = s.query_table(
            "WITH sq0 AS (SELECT c AS fc FROM prices ORDER BY t ASC LIMIT 1), "
            "sq1 AS (SELECT c AS lc FROM prices ORDER BY t DESC LIMIT 1) "
            "SELECT COUNT(*) AS n, MIN(sq0.fc) AS first_c, MIN(sq1.lc) AS last_c "
            "FROM prices CROSS JOIN sq0 CROSS JOIN sq1"
        )
        assert rewritten_df["n"][0] == manual_df["n"][0]
        assert rewritten_df["first_c"][0] == manual_df["first_c"][0]
        assert rewritten_df["last_c"][0] == manual_df["last_c"][0]

    def test_rewritten_cte_matches_manual_cte(self):
        """CTE-body rewrite must match hand-written equivalent."""
        s = self._store()
        rewritten_df = s.query_table(
            "WITH stats AS ("
            " SELECT (SELECT c FROM prices ORDER BY t DESC LIMIT 1)"
            " / (SELECT c FROM prices ORDER BY t ASC LIMIT 1) - 1 AS ret"
            " FROM prices"
            ") SELECT ret FROM stats"
        )
        manual_df = s.query_table(
            "WITH fc AS (SELECT c AS v FROM prices ORDER BY t ASC LIMIT 1), "
            "lc AS (SELECT c AS v FROM prices ORDER BY t DESC LIMIT 1), "
            "stats AS (SELECT lc.v / fc.v - 1 AS ret FROM prices CROSS JOIN fc CROSS JOIN lc) "
            "SELECT ret FROM stats"
        )
        assert abs(rewritten_df["ret"][0] - manual_df["ret"][0]) < 1e-9

    # ---- Edge cases ----

    def test_single_row_table(self):
        """Scalar subquery on a single-row table must work correctly."""
        s = DataFrameStore()
        s.store("t", [{"x": 42}])
        df = s.query_table("SELECT (SELECT x FROM t LIMIT 1) AS val FROM t")
        assert len(df) == 1
        assert df["val"][0] == 42

    def test_subquery_with_different_table(self):
        """Scalar subquery referencing a different table than the main FROM."""
        s = DataFrameStore()
        s.store("a", [{"x": 1}, {"x": 2}, {"x": 3}])
        s.store("b", [{"y": 100}])
        df = s.query_table(
            "SELECT x, (SELECT y FROM b LIMIT 1) AS b_val FROM a ORDER BY x"
        )
        assert len(df) == 3
        assert df["x"] == [1, 2, 3]
        assert df["b_val"] == [100, 100, 100]

    def test_subquery_with_null_values(self):
        """NULLs in data must not be lost or mishandled during rewriting."""
        s = DataFrameStore()
        s.store("t", [{"x": 1, "y": None}, {"x": 2, "y": 10.0}])
        df = s.query_table(
            "SELECT x, y, (SELECT MIN(x) FROM t) AS min_x FROM t ORDER BY x"
        )
        assert len(df) == 2
        assert df["min_x"] == [1, 1]
        assert df["y"][0] is None
        assert df["y"][1] == 10.0

    def test_no_data_loss_all_columns_present(self):
        """All original columns and subquery-derived columns must appear in output."""
        s = self._store()
        df = s.query_table(
            "SELECT t, c, v, (SELECT MAX(c) FROM prices) AS peak FROM prices ORDER BY t"
        )
        assert df.columns == ["t", "c", "v", "peak"]
        assert df["t"] == [1, 2, 3, 4, 5]
        assert df["c"] == [100.0, 110.0, 105.0, 120.0, 115.0]
        assert df["v"] == [1000, 1500, 1200, 800, 900]
        assert df["peak"] == [120.0] * 5

    def test_order_by_preserved_after_rewrite(self):
        """ORDER BY must still sort correctly after rewriting."""
        s = self._store()
        df = s.query_table(
            "SELECT t, c, "
            "(SELECT MIN(c) FROM prices) AS floor "
            "FROM prices ORDER BY c DESC"
        )
        assert df["c"] == [120.0, 115.0, 110.0, 105.0, 100.0]
        assert df["floor"] == [100.0] * 5

    def test_distinct_with_scalar_subquery(self):
        """DISTINCT must still deduplicate correctly."""
        s = DataFrameStore()
        s.store("t", [{"g": "a", "v": 1}, {"g": "a", "v": 2}, {"g": "b", "v": 3}])
        df = s.query_table("SELECT DISTINCT g, (SELECT MAX(v) FROM t) AS max_v FROM t")
        # Sort in Python to avoid conflating with the ORDER BY bug.
        df = df.sort("g")
        assert len(df) == 2
        assert df["g"] == ["a", "b"]
        assert df["max_v"] == [3, 3]

    def test_realistic_financial_query(self):
        """Full realistic query pattern that LLMs generate for financial analysis."""
        s = DataFrameStore()
        s.store(
            "prices",
            [
                {"t": 1, "c": 100.0, "ret": 0.0},
                {"t": 2, "c": 105.0, "ret": 0.05},
                {"t": 3, "c": 102.0, "ret": -0.02857},
                {"t": 4, "c": 110.0, "ret": 0.07843},
                {"t": 5, "c": 108.0, "ret": -0.01818},
            ],
        )
        df = s.query_table(
            "SELECT 'AAPL' AS ticker, "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS open_price, "
            "(SELECT c FROM prices ORDER BY t DESC LIMIT 1) AS close_price, "
            "COUNT(*) AS trading_days, "
            "AVG(ret) AS avg_return, "
            "MIN(c) AS low, "
            "MAX(c) AS high "
            "FROM prices WHERE ret IS NOT NULL"
        )
        assert len(df) == 1
        assert df["ticker"][0] == "AAPL"
        assert df["open_price"][0] == 100.0
        assert df["close_price"][0] == 108.0
        assert df["trading_days"][0] == 5
        assert df["low"][0] == 100.0
        assert df["high"][0] == 110.0
        expected_avg = (0.0 + 0.05 - 0.02857 + 0.07843 - 0.01818) / 5
        assert abs(df["avg_return"][0] - expected_avg) < 0.0001

    def test_subquery_inside_case_inside_aggregate(self):
        """Subqueries nested inside CASE WHEN inside aggregates (e.g. MIN/MAX)."""
        s = DataFrameStore()
        s.store(
            "btc_30d",
            [
                {
                    "t": 1,
                    "o": 100.0,
                    "h": 110.0,
                    "l": 95.0,
                    "c": 105.0,
                    "v": 1000,
                    "n": 50,
                },
                {
                    "t": 2,
                    "o": 105.0,
                    "h": 115.0,
                    "l": 100.0,
                    "c": 110.0,
                    "v": 1500,
                    "n": 75,
                },
                {
                    "t": 3,
                    "o": 110.0,
                    "h": 120.0,
                    "l": 105.0,
                    "c": 115.0,
                    "v": 1200,
                    "n": 60,
                },
            ],
        )
        df = s.query_table(
            "SELECT "
            "MIN(l) AS btc_30d_low, "
            "MAX(h) AS btc_30d_high, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM btc_30d) THEN o ELSE NULL END) AS btc_start_price, "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM btc_30d) THEN c ELSE NULL END) AS btc_end_price, "
            "SUM(v) AS total_volume, "
            "SUM(n) AS total_trades "
            "FROM btc_30d"
        )
        assert df["btc_30d_low"][0] == 95.0
        assert df["btc_30d_high"][0] == 120.0
        assert df["btc_start_price"][0] == 100.0
        assert df["btc_end_price"][0] == 115.0
        assert df["total_volume"][0] == 3700
        assert df["total_trades"][0] == 185

    def test_subquery_inside_case_with_group_by(self):
        """Subqueries inside CASE inside aggregates with GROUP BY."""
        s = DataFrameStore()
        s.store(
            "data",
            [
                {"g": "a", "t": 1, "v": 10},
                {"g": "a", "t": 2, "v": 20},
                {"g": "b", "t": 1, "v": 30},
                {"g": "b", "t": 3, "v": 40},
            ],
        )
        df = s.query_table(
            "SELECT g, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM data) THEN v ELSE NULL END) AS first_v, "
            "SUM(v) AS total "
            "FROM data GROUP BY g"
        )
        df = df.sort("g")
        assert len(df) == 2
        assert df["g"] == ["a", "b"]
        assert df["first_v"] == [10, 30]
        assert df["total"] == [30, 70]


class TestRewriterRobustness:
    """Comprehensive robustness and accuracy tests for the query rewriter.

    These exercise patterns commonly generated by LLMs for financial analysis
    and edge cases that could silently corrupt results.
    """

    def _ohlcv_store(self):
        s = DataFrameStore()
        s.store(
            "prices",
            [
                {"t": 1, "o": 100.0, "h": 105.0, "l": 98.0, "c": 103.0, "v": 1000},
                {"t": 2, "o": 103.0, "h": 108.0, "l": 101.0, "c": 107.0, "v": 1200},
                {"t": 3, "o": 107.0, "h": 110.0, "l": 104.0, "c": 105.0, "v": 800},
                {"t": 4, "o": 105.0, "h": 112.0, "l": 103.0, "c": 111.0, "v": 1500},
                {"t": 5, "o": 111.0, "h": 115.0, "l": 109.0, "c": 113.0, "v": 1100},
            ],
        )
        return s

    # ---- CASE WHEN + subquery across different aggregate functions ----

    def test_sum_case_when_subquery(self):
        """SUM(CASE WHEN t = (SELECT ...) THEN v END) pattern."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "SUM(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN v ELSE 0 END) AS first_vol, "
            "SUM(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN v ELSE 0 END) AS last_vol "
            "FROM prices"
        )
        assert df["first_vol"][0] == 1000
        assert df["last_vol"][0] == 1100

    def test_avg_case_when_subquery(self):
        """AVG with CASE WHEN referencing a subquery comparison."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "AVG(CASE WHEN t >= (SELECT MAX(t) FROM prices) - 1 THEN c ELSE NULL END) AS recent_avg "
            "FROM prices"
        )
        # t >= 4: c values are 111.0, 113.0 -> avg = 112.0
        assert abs(df["recent_avg"][0] - 112.0) < 0.001

    def test_count_case_when_subquery(self):
        """COUNT with CASE WHEN checking against a subquery value."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "COUNT(CASE WHEN c > (SELECT AVG(c) FROM prices) THEN 1 ELSE NULL END) AS above_avg "
            "FROM prices"
        )
        # avg c = (103+107+105+111+113)/5 = 107.8; c > 107.8: 111, 113 -> 2
        assert df["above_avg"][0] == 2

    def test_max_case_when_subquery(self):
        """MAX(CASE WHEN ...) with subquery — different aggregate than MIN."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MAX(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN h ELSE NULL END) AS first_high "
            "FROM prices"
        )
        assert df["first_high"][0] == 105.0

    def test_case_without_else(self):
        """CASE WHEN ... THEN ... END (no ELSE) defaults to NULL."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS first_o, "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) AS last_c "
            "FROM prices"
        )
        assert df["first_o"][0] == 100.0
        assert df["last_c"][0] == 113.0

    def test_arithmetic_combining_case_subqueries(self):
        """Arithmetic using multiple CASE+subquery results in one expression."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) "
            "- MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN c END) AS price_change "
            "FROM prices"
        )
        assert df["price_change"][0] == 10.0  # 113 - 103

    # ---- Mixed standalone subquery + CASE-embedded subquery ----

    def test_mixed_standalone_and_case_subqueries(self):
        """Standalone scalar subquery + CASE-embedded subquery + aggregate."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "(SELECT c FROM prices ORDER BY t ASC LIMIT 1) AS first_c, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o ELSE NULL END) AS first_o, "
            "COUNT(*) AS n "
            "FROM prices"
        )
        assert df["first_c"][0] == 103.0
        assert df["first_o"][0] == 100.0
        assert df["n"][0] == 5

    def test_mixed_standalone_and_case_with_group_by(self):
        """Standalone subquery + CASE-embedded subquery + GROUP BY."""
        s = DataFrameStore()
        s.store(
            "data",
            [
                {"g": "a", "t": 1, "v": 10},
                {"g": "a", "t": 2, "v": 20},
                {"g": "b", "t": 1, "v": 30},
                {"g": "b", "t": 3, "v": 40},
            ],
        )
        df = s.query_table(
            "SELECT g, "
            "(SELECT MAX(v) FROM data) AS global_max, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM data) THEN v ELSE NULL END) AS first_v "
            "FROM data GROUP BY g"
        )
        df = df.sort("g")
        assert df["global_max"] == [40, 40]
        assert df["first_v"] == [10, 30]

    # ---- Non-aggregate queries with CASE + subquery ----

    def test_non_aggregate_case_subquery(self):
        """SELECT t, CASE WHEN t = (subquery) ... — no aggregates."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT t, "
            "CASE WHEN t = (SELECT MIN(t) FROM prices) THEN 'first' "
            "WHEN t = (SELECT MAX(t) FROM prices) THEN 'last' "
            "ELSE 'mid' END AS label "
            "FROM prices ORDER BY t"
        )
        assert len(df) == 5
        assert df["label"] == ["first", "mid", "mid", "mid", "last"]

    def test_non_aggregate_case_subquery_preserves_all_rows(self):
        """CROSS JOIN with single-row CTEs must not collapse rows."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT t, c, "
            "c - (SELECT MIN(c) FROM prices) AS above_min "
            "FROM prices ORDER BY t"
        )
        assert len(df) == 5
        # MIN(c) = 103.0
        assert df["above_min"] == [0.0, 4.0, 2.0, 8.0, 10.0]

    # ---- ORDER BY / LIMIT on the inline path ----

    def test_inline_path_preserves_order_by(self):
        """ORDER BY must still work after inline CROSS JOIN rewrite."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(l) AS low, "
            "MAX(h) AS high, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS first_o "
            "FROM prices ORDER BY low"
        )
        assert len(df) == 1
        assert df["low"][0] == 98.0
        assert df["first_o"][0] == 100.0

    def test_inline_path_preserves_limit(self):
        """LIMIT must still work after inline CROSS JOIN rewrite."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(l) AS low, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS first_o "
            "FROM prices LIMIT 1"
        )
        assert len(df) == 1
        assert df["first_o"][0] == 100.0

    # ---- Percentage / ratio patterns ----

    def test_percentage_of_total_per_group(self):
        """SUM(v) * 100 / (SELECT SUM(v)) — percentage of total per group."""
        s = DataFrameStore()
        s.store(
            "sales",
            [
                {"region": "east", "amount": 100},
                {"region": "east", "amount": 200},
                {"region": "west", "amount": 300},
                {"region": "west", "amount": 400},
            ],
        )
        df = s.query_table(
            "SELECT region, SUM(amount) AS total, "
            "ROUND(SUM(amount) * 100.0 / (SELECT SUM(amount) FROM sales), 1) AS pct "
            "FROM sales GROUP BY region"
        )
        df = df.sort("region")
        assert df["total"] == [300, 700]
        assert df["pct"] == [30.0, 70.0]

    def test_ratio_to_global_max_per_row(self):
        """c / (SELECT MAX(c)) per row — ratio pattern."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT t, "
            "ROUND(c * 100.0 / (SELECT MAX(c) FROM prices), 1) AS pct_of_max "
            "FROM prices ORDER BY t"
        )
        assert len(df) == 5
        # MAX(c) = 113.0
        expected = [
            round(c * 100.0 / 113.0, 1) for c in [103.0, 107.0, 105.0, 111.0, 113.0]
        ]
        assert df["pct_of_max"] == expected

    # ---- Cross-table subquery patterns ----

    def test_cross_table_subquery_in_case(self):
        """CASE WHEN referencing a subquery from a different table."""
        s = DataFrameStore()
        s.store("btc", [{"t": 1, "c": 50000.0}, {"t": 2, "c": 52000.0}])
        s.store("eth", [{"t": 1, "c": 3000.0}, {"t": 2, "c": 3200.0}])
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN btc.t = (SELECT MIN(t) FROM btc) THEN btc.c END) AS btc_start, "
            "MIN(CASE WHEN btc.t = (SELECT MAX(t) FROM btc) THEN btc.c END) AS btc_end "
            "FROM btc"
        )
        assert df["btc_start"][0] == 50000.0
        assert df["btc_end"][0] == 52000.0

    def test_subquery_references_different_table_in_case(self):
        """Subquery inside CASE references a different table than FROM."""
        s = DataFrameStore()
        s.store("orders", [{"id": 1, "amount": 100}, {"id": 2, "amount": 200}])
        s.store("thresholds", [{"min_val": 150}])
        df = s.query_table(
            "SELECT "
            "COUNT(CASE WHEN amount > (SELECT min_val FROM thresholds LIMIT 1) "
            "THEN 1 END) AS above_threshold "
            "FROM orders"
        )
        assert df["above_threshold"][0] == 1  # only id=2 (200 > 150)

    # ---- COALESCE / function wrapping ----

    def test_coalesce_wrapping_scalar_subquery(self):
        """COALESCE((SELECT ...), 0) — function wrapping subquery."""
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}])
        df = s.query_table(
            "SELECT COALESCE((SELECT MIN(x) FROM t), 0) AS min_x, COUNT(*) AS n FROM t"
        )
        assert df["min_x"][0] == 1
        assert df["n"][0] == 2

    def test_coalesce_with_case_subquery(self):
        """COALESCE wrapping a CASE+subquery expression inside an aggregate."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(COALESCE(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END, 0)) "
            "AS first_o "
            "FROM prices"
        )
        assert df["first_o"][0] == 0.0  # MIN(100, 0, 0, 0, 0) = 0

    # ---- NULL handling ----

    def test_null_in_data_with_case_subquery(self):
        """NULL values in data must not corrupt CASE+subquery results."""
        s = DataFrameStore()
        s.store(
            "t",
            [
                {"ts": 1, "val": None},
                {"ts": 2, "val": 10.0},
                {"ts": 3, "val": 20.0},
            ],
        )
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN ts = (SELECT MIN(ts) FROM t) THEN val END) AS first_val, "
            "MIN(CASE WHEN ts = (SELECT MAX(ts) FROM t) THEN val END) AS last_val "
            "FROM t"
        )
        assert df["first_val"][0] is None  # val is NULL at ts=1
        assert df["last_val"][0] == 20.0

    def test_subquery_returns_null(self):
        """Subquery returning NULL must not crash the rewriter."""
        s = DataFrameStore()
        s.store("t", [{"x": None}, {"x": None}])
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN x = (SELECT MIN(x) FROM t) THEN 1 ELSE 0 END) AS result "
            "FROM t"
        )
        # MIN(x) is NULL, x = NULL is always false in SQL -> all get ELSE 0
        assert df["result"][0] == 0

    # ---- Single-row edge case ----

    def test_single_row_case_subquery(self):
        """CASE+subquery on a single-row table."""
        s = DataFrameStore()
        s.store("t", [{"ts": 1, "o": 50.0, "c": 55.0}])
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN ts = (SELECT MIN(ts) FROM t) THEN o END) AS start_p, "
            "MIN(CASE WHEN ts = (SELECT MAX(ts) FROM t) THEN c END) AS end_p "
            "FROM t"
        )
        assert df["start_p"][0] == 50.0
        assert df["end_p"][0] == 55.0

    # ---- Subquery with its own WHERE clause ----

    def test_subquery_with_where_inside_case(self):
        """The subquery inside CASE has its own WHERE filter."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices WHERE c > 105) "
            "THEN c END) AS first_above_105 "
            "FROM prices"
        )
        # prices where c > 105: t=2(107), t=4(111), t=5(113). MIN(t) = 2. c at t=2 = 107
        assert df["first_above_105"][0] == 107.0

    # ---- Duplicate subquery values ----

    def test_same_subquery_used_multiple_times(self):
        """The same subquery value appears in multiple CASE expressions."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS first_o, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN c END) AS first_c, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN h END) AS first_h "
            "FROM prices"
        )
        assert df["first_o"][0] == 100.0
        assert df["first_c"][0] == 103.0
        assert df["first_h"][0] == 105.0

    # ---- Realistic financial analysis queries ----

    def test_full_crypto_ohlcv_analysis(self):
        """Exact pattern from the user's bug report with realistic data."""
        s = DataFrameStore()
        s.store(
            "btc_30d",
            [
                {
                    "t": 1,
                    "o": 42000.0,
                    "h": 43500.0,
                    "l": 41000.0,
                    "c": 43000.0,
                    "v": 5.5,
                    "n": 15000,
                },
                {
                    "t": 2,
                    "o": 43000.0,
                    "h": 44000.0,
                    "l": 42500.0,
                    "c": 43800.0,
                    "v": 6.2,
                    "n": 17000,
                },
                {
                    "t": 3,
                    "o": 43800.0,
                    "h": 45000.0,
                    "l": 43000.0,
                    "c": 44500.0,
                    "v": 7.1,
                    "n": 20000,
                },
            ],
        )
        df = s.query_table(
            "SELECT "
            "MIN(l) AS btc_30d_low, "
            "MAX(h) AS btc_30d_high, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM btc_30d) THEN o ELSE NULL END) AS btc_start_price, "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM btc_30d) THEN c ELSE NULL END) AS btc_end_price, "
            "SUM(v) AS total_volume, "
            "SUM(n) AS total_trades "
            "FROM btc_30d"
        )
        assert df["btc_30d_low"][0] == 41000.0
        assert df["btc_30d_high"][0] == 45000.0
        assert df["btc_start_price"][0] == 42000.0
        assert df["btc_end_price"][0] == 44500.0
        assert abs(df["total_volume"][0] - 18.8) < 0.001
        assert df["total_trades"][0] == 52000

    def test_multi_asset_sequential_queries(self):
        """Multiple assets queried in sequence — the full user workflow."""
        s = DataFrameStore()
        for asset, base in [("btc", 40000.0), ("eth", 3000.0), ("sol", 100.0)]:
            s.store(
                f"{asset}_30d",
                [
                    {
                        "t": i,
                        "o": base + i * 10,
                        "h": base + i * 15,
                        "l": base + i * 5,
                        "c": base + i * 12,
                        "v": 100 + i,
                        "n": 50 + i,
                    }
                    for i in range(1, 6)
                ],
            )
        for asset in ["btc", "eth", "sol"]:
            tbl = f"{asset}_30d"
            df = s.query_table(
                f"SELECT "
                f"MIN(l) AS low, MAX(h) AS high, "
                f"MIN(CASE WHEN t = (SELECT MIN(t) FROM {tbl}) THEN o ELSE NULL END) AS start_price, "
                f"MIN(CASE WHEN t = (SELECT MAX(t) FROM {tbl}) THEN c ELSE NULL END) AS end_price, "
                f"SUM(v) AS total_volume, "
                f"SUM(n) AS total_trades "
                f"FROM {tbl}"
            )
            assert len(df) == 1
            assert df["start_price"][0] is not None
            assert df["end_price"][0] is not None
            assert df["low"][0] < df["high"][0]
            assert df["total_volume"][0] > 0

    def test_financial_return_with_case_extraction(self):
        """Calculate return using CASE WHEN to extract first/last close."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN c END) AS start_c, "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) AS end_c, "
            "ROUND((MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) "
            " - MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN c END)) "
            " / MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN c END) * 100, 2) "
            "AS return_pct "
            "FROM prices"
        )
        assert df["start_c"][0] == 103.0
        assert df["end_c"][0] == 113.0
        expected_return = round((113.0 - 103.0) / 103.0 * 100, 2)
        assert abs(df["return_pct"][0] - expected_return) < 0.01

    def test_case_subquery_with_string_literal_and_aggregates(self):
        """String literals + CASE+subquery + aggregates — a common LLM pattern."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT 'AAPL' AS ticker, '5D' AS period, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS open_price, "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) AS close_price, "
            "COUNT(*) AS trading_days, "
            "MIN(l) AS period_low, "
            "MAX(h) AS period_high "
            "FROM prices"
        )
        assert df["ticker"][0] == "AAPL"
        assert df["period"][0] == "5D"
        assert df["open_price"][0] == 100.0
        assert df["close_price"][0] == 113.0
        assert df["trading_days"][0] == 5
        assert df["period_low"][0] == 98.0
        assert df["period_high"][0] == 115.0

    def test_case_subquery_with_where_on_main_query(self):
        """WHERE clause on the main query combined with CASE+subquery."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT "
            "COUNT(*) AS n, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices WHERE t >= 3) THEN c END) AS start_c "
            "FROM prices WHERE t >= 3"
        )
        # Rows where t >= 3: t=3,4,5. MIN(t) in subquery (also t >= 3) = 3.
        # c at t=3 = 105.0
        assert df["n"][0] == 3
        assert df["start_c"][0] == 105.0

    def test_case_subquery_in_cte_body(self):
        """CASE+subquery inside a CTE body, then queried from outer SELECT."""
        s = self._ohlcv_store()
        df = s.query_table(
            "WITH stats AS ("
            " SELECT "
            " MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS start_price,"
            " MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) AS end_price,"
            " COUNT(*) AS n"
            " FROM prices"
            ") SELECT start_price, end_price,"
            " ROUND((end_price - start_price) / start_price * 100, 2) AS return_pct"
            " FROM stats"
        )
        assert df["start_price"][0] == 100.0
        assert df["end_price"][0] == 113.0
        expected = round((113.0 - 100.0) / 100.0 * 100, 2)
        assert abs(df["return_pct"][0] - expected) < 0.01

    # ---- Aggregate correctness verification ----

    def test_sum_not_inflated_with_case_subquery(self):
        """SUM must not be multiplied by CROSS JOIN with CTEs."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT SUM(v) AS total_vol, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS first_o "
            "FROM prices"
        )
        assert df["total_vol"][0] == 5600  # 1000+1200+800+1500+1100
        assert df["first_o"][0] == 100.0

    def test_count_not_inflated_with_case_subquery(self):
        """COUNT must still reflect actual row count."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT COUNT(*) AS n, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN o END) AS first_o, "
            "MIN(CASE WHEN t = (SELECT MAX(t) FROM prices) THEN c END) AS last_c "
            "FROM prices"
        )
        assert df["n"][0] == 5
        assert df["first_o"][0] == 100.0
        assert df["last_c"][0] == 113.0

    def test_avg_not_affected_with_case_subquery(self):
        """AVG must not be skewed by the rewrite."""
        s = self._ohlcv_store()
        df = s.query_table(
            "SELECT AVG(c) AS avg_c, "
            "MIN(CASE WHEN t = (SELECT MIN(t) FROM prices) THEN c END) AS first_c "
            "FROM prices"
        )
        expected_avg = (103.0 + 107.0 + 105.0 + 111.0 + 113.0) / 5
        assert abs(df["avg_c"][0] - expected_avg) < 0.001
        assert df["first_c"][0] == 103.0


class TestSQLFeatureCoverage:
    """Test coverage for SQL features advertised in query_data docstring.

    Covers: window functions, ILIKE/LIKE, set operations, join variants,
    HAVING, EXISTS/NOT EXISTS, BETWEEN/IN, CAST, string functions, chained
    CTEs, OFFSET, OR conditions, and realistic LLM-generated combo patterns.
    """

    def _make_store(self):
        s = DataFrameStore()
        s.store(
            "employees",
            [
                {"name": "Alice", "dept": "eng", "salary": 100, "hired": 1},
                {"name": "Bob", "dept": "eng", "salary": 120, "hired": 2},
                {"name": "Carol", "dept": "sales", "salary": 90, "hired": 3},
                {"name": "Dave", "dept": "sales", "salary": 110, "hired": 4},
                {"name": "Eve", "dept": "eng", "salary": 130, "hired": 5},
            ],
        )
        return s

    # ---- Window Functions ----

    def test_row_number(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees"
        )
        assert len(df) == 5
        # Find Eve's row number
        eve_idx = df["name"].index("Eve")
        assert df["rn"][eve_idx] == 1

    def test_rank_dense_rank_with_ties(self):
        s = DataFrameStore()
        s.store(
            "scores",
            [
                {"name": "A", "score": 100},
                {"name": "B", "score": 90},
                {"name": "C", "score": 90},
                {"name": "D", "score": 80},
            ],
        )
        df = s.query_table(
            "SELECT name, score, "
            "RANK() OVER (ORDER BY score DESC) AS rnk, "
            "DENSE_RANK() OVER (ORDER BY score DESC) AS drnk "
            "FROM scores ORDER BY score DESC, name"
        )
        assert df["rnk"] == [1, 2, 2, 4]
        assert df["drnk"] == [1, 2, 2, 3]

    def test_lag_lead(self):
        s = DataFrameStore()
        s.store("ts", [{"t": 1, "v": 10}, {"t": 2, "v": 20}, {"t": 3, "v": 15}])
        df = s.query_table(
            "SELECT t, v, "
            "LAG(v) OVER (ORDER BY t) AS prev_v, "
            "LEAD(v) OVER (ORDER BY t) AS next_v "
            "FROM ts ORDER BY t"
        )
        assert df["prev_v"][0] is None
        assert df["prev_v"][1] == 10
        assert df["next_v"][1] == 15
        assert df["next_v"][2] is None

    def test_running_total(self):
        s = DataFrameStore()
        s.store("ts", [{"t": 1, "v": 10}, {"t": 2, "v": 20}, {"t": 3, "v": 30}])
        df = s.query_table(
            "SELECT t, v, "
            "SUM(v) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running "
            "FROM ts ORDER BY t"
        )
        assert df["running"] == [10, 30, 60]

    def test_partition_by(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT name, dept, salary, "
            "SUM(salary) OVER (PARTITION BY dept) AS dept_total "
            "FROM employees ORDER BY name"
        )
        alice_idx = df["name"].index("Alice")
        assert df["dept_total"][alice_idx] == 350  # eng: 100+120+130

    def test_ntile(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT name, salary, NTILE(3) OVER (ORDER BY salary) AS bucket "
            "FROM employees ORDER BY salary"
        )
        assert len(df) == 5
        buckets = df["bucket"]
        assert min(buckets) == 1
        assert max(buckets) == 3

    # ---- ILIKE / LIKE ----

    def test_ilike(self):
        """ILIKE is preprocessed to LIKE (SQLite LIKE is case-insensitive for ASCII)."""
        s = self._make_store()
        df = s.query_table("SELECT name FROM employees WHERE name ILIKE '%ali%'")
        assert len(df) == 1
        assert df["name"][0] == "Alice"

    def test_like(self):
        s = self._make_store()
        df = s.query_table("SELECT name FROM employees WHERE name LIKE 'B%'")
        assert len(df) == 1
        assert df["name"][0] == "Bob"

    # ---- Set Operations ----

    def test_union_all(self):
        s = DataFrameStore()
        s.store("a", [{"x": 1}, {"x": 2}])
        s.store("b", [{"x": 2}, {"x": 3}])
        df = s.query_table("SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x")
        assert df["x"] == [1, 2, 2, 3]

    def test_union_dedup(self):
        s = DataFrameStore()
        s.store("a", [{"x": 1}, {"x": 2}])
        s.store("b", [{"x": 2}, {"x": 3}])
        df = s.query_table("SELECT x FROM a UNION SELECT x FROM b ORDER BY x")
        assert df["x"] == [1, 2, 3]

    def test_except_intersect(self):
        s = DataFrameStore()
        s.store("a", [{"x": 1}, {"x": 2}, {"x": 3}])
        s.store("b", [{"x": 2}, {"x": 3}, {"x": 4}])
        df_except = s.query_table("SELECT x FROM a EXCEPT SELECT x FROM b")
        assert df_except["x"] == [1]
        df_intersect = s.query_table(
            "SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY x"
        )
        assert df_intersect["x"] == [2, 3]

    # ---- Join Variants ----

    def test_left_join_with_nulls(self):
        s = DataFrameStore()
        s.store("orders", [{"id": 1, "cust": "A"}, {"id": 2, "cust": "B"}])
        s.store("payments", [{"order_id": 1, "amount": 50}])
        df = s.query_table(
            "SELECT o.id, o.cust, p.amount "
            "FROM orders o LEFT JOIN payments p ON o.id = p.order_id "
            "ORDER BY o.id"
        )
        assert len(df) == 2
        assert df["amount"][0] == 50
        assert df["amount"][1] is None

    def test_full_outer_join(self):
        s = DataFrameStore()
        s.store("a", [{"k": 1, "va": 10}, {"k": 2, "va": 20}])
        s.store("b", [{"k": 2, "vb": 200}, {"k": 3, "vb": 300}])
        df = s.query_table(
            "SELECT a.k AS ak, b.k AS bk, va, vb "
            "FROM a FULL OUTER JOIN b ON a.k = b.k "
            "ORDER BY COALESCE(a.k, b.k)"
        )
        assert len(df) == 3
        # k=1: va=10, vb=NULL; k=2: matched; k=3: va=NULL, vb=300
        assert df["va"][0] == 10
        assert df["vb"][0] is None
        assert df["vb"][2] == 300
        assert df["va"][2] is None

    def test_multi_table_join(self):
        s = DataFrameStore()
        s.store("customers", [{"cid": 1, "name": "Alice"}, {"cid": 2, "name": "Bob"}])
        s.store(
            "orders",
            [
                {"oid": 10, "cid": 1, "product_id": 100},
                {"oid": 11, "cid": 2, "product_id": 101},
            ],
        )
        s.store(
            "products",
            [
                {"pid": 100, "pname": "Widget"},
                {"pid": 101, "pname": "Gadget"},
            ],
        )
        df = s.query_table(
            "SELECT c.name, p.pname "
            "FROM customers c "
            "JOIN orders o ON c.cid = o.cid "
            "JOIN products p ON o.product_id = p.pid "
            "ORDER BY c.name"
        )
        assert len(df) == 2
        assert df["name"] == ["Alice", "Bob"]
        assert df["pname"] == ["Widget", "Gadget"]

    # ---- HAVING ----

    def test_having(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT dept, SUM(salary) AS total "
            "FROM employees GROUP BY dept HAVING SUM(salary) > 300"
        )
        assert len(df) == 1
        assert df["dept"][0] == "eng"
        assert df["total"][0] == 350

    # ---- EXISTS / NOT EXISTS ----

    def test_exists(self):
        s = DataFrameStore()
        s.store("orders", [{"id": 1, "cust": "A"}, {"id": 2, "cust": "B"}])
        s.store("payments", [{"order_id": 1, "amount": 50}])
        df = s.query_table(
            "SELECT id, cust FROM orders o "
            "WHERE EXISTS (SELECT 1 FROM payments p WHERE p.order_id = o.id)"
        )
        assert len(df) == 1
        assert df["cust"][0] == "A"

    def test_not_exists(self):
        s = DataFrameStore()
        s.store("orders", [{"id": 1, "cust": "A"}, {"id": 2, "cust": "B"}])
        s.store("payments", [{"order_id": 1, "amount": 50}])
        df = s.query_table(
            "SELECT id, cust FROM orders o "
            "WHERE NOT EXISTS (SELECT 1 FROM payments p WHERE p.order_id = o.id)"
        )
        assert len(df) == 1
        assert df["cust"][0] == "B"

    # ---- BETWEEN / IN ----

    def test_between(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT name FROM employees WHERE salary BETWEEN 100 AND 120 ORDER BY name"
        )
        assert df["name"] == ["Alice", "Bob", "Dave"]

    def test_in_literal_list(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT name FROM employees WHERE salary IN (90, 130) ORDER BY name"
        )
        assert df["name"] == ["Carol", "Eve"]

    # ---- CAST ----

    def test_cast(self):
        """CAST AS VARCHAR/DOUBLE are preprocessed to TEXT/REAL."""
        s = DataFrameStore()
        s.store("t", [{"x": 42}])
        df = s.query_table(
            "SELECT CAST(x AS VARCHAR) AS xs, CAST(x AS DOUBLE) AS xd FROM t"
        )
        assert df["xs"][0] == "42"
        assert df["xd"][0] == 42.0

    # ---- String Functions ----

    def test_string_functions(self):
        s = DataFrameStore()
        s.store("t", [{"s": "Hello"}])
        df = s.query_table(
            "SELECT UPPER(s) AS u, LOWER(s) AS l, "
            "CONCAT(s, ' World') AS c, LENGTH(s) AS n FROM t"
        )
        assert df["u"][0] == "HELLO"
        assert df["l"][0] == "hello"
        assert df["c"][0] == "Hello World"
        assert df["n"][0] == 5

    # ---- Chained CTEs ----

    def test_chained_ctes(self):
        s = self._make_store()
        df = s.query_table(
            "WITH eng AS ("
            "  SELECT name, salary FROM employees WHERE dept = 'eng'"
            "), ranked AS ("
            "  SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM eng"
            ") SELECT name, salary FROM ranked WHERE rn = 1"
        )
        assert len(df) == 1
        assert df["name"][0] == "Eve"
        assert df["salary"][0] == 130

    # ---- SQLite-compatible feature tests ----

    def test_filter_clause_via_preprocessor(self):
        """COUNT(*) FILTER (WHERE ...) is preprocessed to SUM(CASE WHEN ...)."""
        s = self._make_store()
        df = s.query_table(
            "SELECT COUNT(*) FILTER (WHERE dept = 'eng') AS eng_count, "
            "COUNT(*) AS total FROM employees"
        )
        assert df["eng_count"][0] == 3
        assert df["total"][0] == 5

    def test_filter_clause_nested_parens(self):
        """COUNT(*) FILTER with nested parentheses in the condition."""
        s = self._make_store()
        df = s.query_table(
            "SELECT COUNT(*) FILTER (WHERE (salary > 100 OR dept = 'sales')) AS cnt "
            "FROM employees"
        )
        # salary > 100: Bob(120), Dave(110), Eve(130) = 3
        # dept = 'sales': Carol(90), Dave(110) = 2
        # union: Bob, Carol, Dave, Eve = 4
        assert df["cnt"][0] == 4

    def test_group_concat(self):
        """GROUP_CONCAT replaces STRING_AGG (via preprocessor)."""
        s = self._make_store()
        df = s.query_table(
            "SELECT dept, STRING_AGG(name, ', ' ORDER BY name) AS names "
            "FROM employees GROUP BY dept ORDER BY dept"
        )
        # GROUP_CONCAT does not guarantee ordering, so check contents as sets
        assert df["dept"] == ["eng", "sales"]
        eng_names = set(df["names"][0].split(", "))
        assert eng_names == {"Alice", "Bob", "Eve"}
        sales_names = set(df["names"][1].split(", "))
        assert sales_names == {"Carol", "Dave"}

    def test_top_n_per_group_via_cte(self):
        """Top-N per group using CTE + ROW_NUMBER (replaces QUALIFY test)."""
        s = self._make_store()
        df = s.query_table(
            "WITH ranked AS ("
            "  SELECT name, dept, salary, "
            "  ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
            "  FROM employees"
            ") SELECT name, dept, salary FROM ranked WHERE rn = 1 ORDER BY dept"
        )
        assert len(df) == 2
        assert df["name"] == ["Eve", "Dave"]

    # ---- OFFSET ----

    def test_limit_offset(self):
        s = self._make_store()
        df = s.query_table("SELECT name FROM employees ORDER BY hired LIMIT 2 OFFSET 2")
        assert len(df) == 2
        assert df["name"] == ["Carol", "Dave"]

    # ---- OR Conditions ----

    def test_or_conditions(self):
        s = self._make_store()
        df = s.query_table(
            "SELECT name FROM employees WHERE salary > 125 OR salary < 95 ORDER BY name"
        )
        assert df["name"] == ["Carol", "Eve"]

    # ---- Realistic LLM-Generated Combo Patterns ----

    def test_window_cte_combo_moving_average(self):
        s = DataFrameStore()
        s.store(
            "ts",
            [
                {"t": 1, "v": 10},
                {"t": 2, "v": 20},
                {"t": 3, "v": 30},
                {"t": 4, "v": 40},
                {"t": 5, "v": 50},
            ],
        )
        df = s.query_table(
            "WITH windowed AS ("
            "  SELECT t, v, "
            "  AVG(v) OVER (ORDER BY t ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS ma "
            "  FROM ts"
            ") SELECT t, v, ROUND(ma, 2) AS ma FROM windowed ORDER BY t"
        )
        assert len(df) == 5
        # t=1: avg(10,20)=15, t=2: avg(10,20,30)=20, t=3: avg(20,30,40)=30
        assert df["ma"][0] == 15.0
        assert df["ma"][1] == 20.0
        assert df["ma"][2] == 30.0

    def test_top_n_per_group(self):
        s = self._make_store()
        df = s.query_table(
            "WITH ranked AS ("
            "  SELECT name, dept, salary, "
            "  ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
            "  FROM employees"
            ") SELECT name, dept, salary FROM ranked WHERE rn <= 2 ORDER BY dept, salary DESC"
        )
        # eng top 2: Eve(130), Bob(120); sales top 2: Dave(110), Carol(90)
        assert len(df) == 4
        assert df["name"] == ["Eve", "Bob", "Dave", "Carol"]


class TestPreprocessNestedEdgeCases:
    """Tests for _preprocess_sql handling of nested/complex SQL.

    These cover edge cases where a regex-based STRING_AGG rewrite would
    mangle valid queries containing nested function calls, CASE expressions,
    or subqueries as arguments.  The AST-based approach handles them all.
    """

    def _make_store(self):
        s = DataFrameStore()
        s.store(
            "employees",
            [
                {"name": "Alice", "dept": "eng", "salary": 100, "hired": 1},
                {"name": "Bob", "dept": "eng", "salary": 120, "hired": 2},
                {"name": "Carol", "dept": "sales", "salary": 90, "hired": 3},
                {"name": "Dave", "dept": "sales", "salary": 110, "hired": 4},
                {"name": "Eve", "dept": "eng", "salary": 130, "hired": 5},
            ],
        )
        return s

    # ---- STRING_AGG with nested expressions (previously broken by regex) ----

    def test_string_agg_with_coalesce_nested_function(self):
        """STRING_AGG with COALESCE (commas in first arg break regex [^,]+)."""
        s = self._make_store()
        df = s.query_table(
            "SELECT dept, STRING_AGG(COALESCE(name, 'Unknown'), ', ' ORDER BY name) AS names "
            "FROM employees GROUP BY dept ORDER BY dept"
        )
        assert df["dept"] == ["eng", "sales"]
        eng_names = set(df["names"][0].split(", "))
        assert eng_names == {"Alice", "Bob", "Eve"}

    def test_string_agg_with_case_in_list(self):
        """STRING_AGG with CASE WHEN ... IN (1, 2) — commas inside IN break regex."""
        s = DataFrameStore()
        s.store(
            "items",
            [
                {"id": 1, "label": "a"},
                {"id": 2, "label": "b"},
                {"id": 3, "label": "c"},
            ],
        )
        df = s.query_table(
            "SELECT STRING_AGG("
            "  CASE WHEN id IN (1, 2) THEN label ELSE 'other' END, ', '"
            "  ORDER BY label"
            ") AS result FROM items"
        )
        # id=1→'a', id=2→'b', id=3→'other'
        labels = set(df["result"][0].split(", "))
        assert labels == {"a", "b", "other"}

    def test_string_agg_with_case_expression(self):
        """STRING_AGG with a CASE expression as the value arg."""
        s = self._make_store()
        df = s.query_table(
            "SELECT dept, STRING_AGG("
            "  CASE WHEN salary > 100 THEN name ELSE 'low' END, ', '"
            "  ORDER BY name"
            ") AS labels "
            "FROM employees GROUP BY dept ORDER BY dept"
        )
        assert df["dept"] == ["eng", "sales"]
        eng_labels = set(df["labels"][0].split(", "))
        assert eng_labels == {"low", "Bob", "Eve"}

    def test_string_agg_without_order_by(self):
        """Plain STRING_AGG without ORDER BY should rewrite to GROUP_CONCAT."""
        s = self._make_store()
        df = s.query_table(
            "SELECT dept, STRING_AGG(name, ', ') AS names "
            "FROM employees GROUP BY dept ORDER BY dept"
        )
        assert df["dept"] == ["eng", "sales"]
        assert len(df["names"][0].split(", ")) == 3
        assert len(df["names"][1].split(", ")) == 2

    def test_string_agg_in_subquery(self):
        """STRING_AGG inside a subquery."""
        s = self._make_store()
        df = s.query_table(
            "SELECT sub.dept, sub.names FROM ("
            "  SELECT dept, STRING_AGG(name, ', ' ORDER BY name) AS names "
            "  FROM employees GROUP BY dept"
            ") sub ORDER BY sub.dept"
        )
        assert df["dept"] == ["eng", "sales"]
        eng_names = set(df["names"][0].split(", "))
        assert eng_names == {"Alice", "Bob", "Eve"}

    def test_string_agg_concat_nested(self):
        """STRING_AGG wrapping CONCAT (commas break regex [^,]+)."""
        s = self._make_store()
        df = s.query_table(
            "SELECT dept, STRING_AGG(CONCAT(name, ':' , dept), ', ' ORDER BY name) AS pairs "
            "FROM employees GROUP BY dept ORDER BY dept"
        )
        assert df["dept"] == ["eng", "sales"]
        eng_pairs = set(df["pairs"][0].split(", "))
        assert eng_pairs == {"Alice:eng", "Bob:eng", "Eve:eng"}

    # ---- Keyword-in-string-literal preservation ----

    def test_preprocess_preserves_double_in_string_literal(self):
        """DOUBLE inside a string literal must not become REAL."""
        result = _preprocess_sql("SELECT * FROM t WHERE col = 'DOUBLE value'")
        assert "'DOUBLE value'" in result

    def test_preprocess_preserves_ilike_in_string_literal(self):
        """ILIKE inside a string literal must not become LIKE."""
        result = _preprocess_sql("SELECT * FROM t WHERE col = 'uses ILIKE syntax'")
        assert "'uses ILIKE syntax'" in result

    def test_preprocess_preserves_varchar_in_string_literal(self):
        """VARCHAR inside a string literal must not become TEXT."""
        result = _preprocess_sql("SELECT * FROM t WHERE col = 'type is VARCHAR'")
        assert "'type is VARCHAR'" in result

    # ---- Direct _preprocess_sql output validation ----

    def test_preprocess_string_agg_coalesce_output(self):
        """_preprocess_sql must produce balanced parens for STRING_AGG(COALESCE(...))."""
        sql = "SELECT STRING_AGG(COALESCE(a, b), ', ' ORDER BY a) FROM t"
        result = _preprocess_sql(sql)
        assert "GROUP_CONCAT" in result.upper()
        assert "STRING_AGG" not in result.upper()
        assert result.count("(") == result.count(")")

    def test_preprocess_any_value_rewrite(self):
        """ANY_VALUE should be rewritten to an aggregate (MIN or MAX)."""
        result = _preprocess_sql("SELECT ANY_VALUE(name) FROM t GROUP BY dept")
        upper = result.upper()
        assert "ANY_VALUE" not in upper
        assert "MIN" in upper or "MAX" in upper

    def test_preprocess_ilike_rewrite(self):
        """ILIKE should be rewritten to a case-insensitive form."""
        result = _preprocess_sql("SELECT * FROM t WHERE name ILIKE '%foo%'")
        upper = result.upper()
        # sqlglot rewrites to LOWER(name) LIKE LOWER('%foo%')
        assert "ILIKE" not in upper
        assert "LIKE" in upper

    def test_preprocess_cast_varchar_to_text(self):
        """CAST(... AS VARCHAR) should become CAST(... AS TEXT)."""
        result = _preprocess_sql("SELECT CAST(x AS VARCHAR) FROM t")
        assert "TEXT" in result.upper()
        assert "VARCHAR" not in result.upper()

    def test_preprocess_cast_double_to_real(self):
        """CAST(... AS DOUBLE) should become CAST(... AS REAL)."""
        result = _preprocess_sql("SELECT CAST(x AS DOUBLE) FROM t")
        assert "REAL" in result.upper()
        # DOUBLE should not appear (except possibly in a different context)
        assert "DOUBLE" not in result.upper()


class TestSQLSecurityValidation:
    """Security tests: ensure SQL injection / filesystem access is blocked."""

    def _store_with_data(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}])
        return s

    # ---- Statement-type validation ----

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE evil AS SELECT 1",
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET x = 1",
            "DELETE FROM t",
            "DROP TABLE t",
            "ALTER TABLE t ADD COLUMN y INT",
            "ATTACH '/tmp/db.sqlite' AS stolen",
            "COPY t TO '/tmp/out.csv'",
            "COPY (SELECT 1) TO '/tmp/out.csv'",
            "INSTALL httpfs",
            "LOAD httpfs",
            "PRAGMA database_list",
            "SET enable_external_access = true",
            "CALL pragma_version()",
        ],
    )
    def test_non_select_statements_blocked(self, sql):
        s = self._store_with_data()
        with pytest.raises(ValueError):
            s.query(sql)

    def test_empty_query_rejected(self):
        s = self._store_with_data()
        with pytest.raises(ValueError, match="Empty SQL query"):
            s.query("")
        with pytest.raises(ValueError, match="Empty SQL query"):
            s.query("   ")

    def test_multiple_statements_blocked(self):
        s = self._store_with_data()
        with pytest.raises(ValueError):
            s.query("SELECT 1; DROP TABLE t")

    # ---- Dangerous functions error (they don't exist in SQLite) ----

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM read_csv('/etc/passwd')",
            "SELECT * FROM read_text('/etc/passwd')",
            "SELECT * FROM read_parquet('/tmp/data.parquet')",
            "SELECT * FROM glob('/Users/*/.ssh/*')",
        ],
    )
    def test_filesystem_functions_error(self, sql):
        """Functions like read_csv don't exist in SQLite and will error."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query(sql)

    def test_getenv_errors(self):
        """getenv() does not exist in SQLite."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query("SELECT getenv('HOME')")

    # ---- ATTACH / PRAGMA blocked by validation ----

    def test_attach_blocked(self):
        """ATTACH is blocked by keyword validation."""
        s = self._store_with_data()
        with pytest.raises(ValueError):
            s.query("ATTACH DATABASE ':memory:' AS other")

    def test_pragma_blocked(self):
        """PRAGMA is blocked by keyword validation."""
        s = self._store_with_data()
        with pytest.raises(ValueError):
            s.query("PRAGMA table_info(t)")

    # ---- Table reference validation ----

    def test_unregistered_table_blocked(self):
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query("SELECT * FROM nonexistent_table")

    def test_cte_references_not_rejected_as_unregistered(self):
        """CTE-defined names must not trigger the table allowlist check."""
        s = self._store_with_data()
        df = s.query_table(
            "WITH doubled AS (SELECT x * 2 AS x2 FROM t) "
            "SELECT x2 FROM doubled ORDER BY x2"
        )
        assert df["x2"] == [2, 4]

    # ---- Parser edge-cases that must not false-positive ----

    def test_trailing_semicolon_allowed(self):
        """Trailing semicolon is stripped by validation."""
        s = self._store_with_data()
        df = s.query_table("SELECT x FROM t;")
        assert len(df) == 2

    def test_semicolons_in_string_literals_allowed(self):
        """Semicolons inside string literals must not be treated as statement separators."""
        s = self._store_with_data()
        df = s.query_table("SELECT 'hello;world' AS label FROM t")
        assert df["label"][0] == "hello;world"

    def test_standard_functions_still_allowed(self):
        """Common SQL functions (avg, count, sum, etc.) must not be blocked."""
        s = self._store_with_data()
        df = s.query_table(
            "SELECT COUNT(*) AS cnt, SUM(x) AS total, AVG(x) AS avg_x FROM t"
        )
        assert df["cnt"][0] == 2
        assert df["total"][0] == 3

    # ---- Schema-qualified table references ----

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM main.sqlite_master",
            "SELECT * FROM main.t",
            "SELECT * FROM temp.sqlite_master",
            "SELECT sql FROM pragma_table_info('t')",
        ],
    )
    def test_schema_qualified_references_blocked(self, sql):
        """Schema-qualified refs (main.X, temp.X) must be rejected."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query(sql)

    # ---- System table access via subqueries / UNION ----

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM (SELECT * FROM sqlite_master)",
            "SELECT x FROM t UNION SELECT sql FROM sqlite_master",
            "SELECT x FROM t UNION ALL SELECT name FROM sqlite_master",
            "SELECT x FROM t WHERE x IN (SELECT rowid FROM sqlite_master)",
        ],
    )
    def test_system_table_access_via_subquery_or_union(self, sql):
        """sqlite_master must be blocked even inside subqueries and UNIONs."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query(sql)

    # ---- SQLite-specific dangerous functions ----

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT load_extension('/tmp/evil.so')",
            "SELECT load_extension('/tmp/evil.so', 'entry')",
            "SELECT readfile('/etc/passwd')",
            "SELECT writefile('/tmp/pwned', 'data')",
            "SELECT fts3_tokenizer('simple')",
        ],
    )
    def test_sqlite_dangerous_functions_blocked(self, sql):
        """SQLite extension functions that could escape the sandbox must be blocked."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query(sql)

    # ---- Anonymous / unrecognised function catch-all ----

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT some_unknown_func(x) FROM t",
            "SELECT my_custom_udf(1, 2, 3)",
        ],
    )
    def test_anonymous_functions_blocked(self, sql):
        """Unrecognised function names must be rejected by the catch-all."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query(sql)

    # ---- False-positive avoidance ----

    def test_keywords_in_string_literals_not_blocked(self):
        """SQL keywords inside string literals must not trigger rejection."""
        s = self._store_with_data()
        df = s.query_table("SELECT 'DROP TABLE t; DELETE FROM t' AS val FROM t")
        assert df["val"][0] == "DROP TABLE t; DELETE FROM t"

    def test_keywords_in_column_aliases_not_blocked(self):
        """Column aliases that look like keywords must not trigger rejection."""
        s = self._store_with_data()
        df = s.query_table("SELECT x AS delete_count FROM t")
        assert df["delete_count"] == [1, 2]

    # ---- Subquery table references in various positions ----

    def test_unregistered_table_in_subquery_blocked(self):
        """Unregistered tables inside subqueries must be caught."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query("SELECT * FROM (SELECT * FROM secret_table)")

    def test_unregistered_table_in_where_subquery_blocked(self):
        """Unregistered tables in WHERE subqueries must be caught."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query("SELECT * FROM t WHERE x IN (SELECT y FROM secret_table)")

    def test_unregistered_table_in_join_blocked(self):
        """Unregistered tables in JOINs must be caught."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query("SELECT * FROM t JOIN secret_table s ON t.x = s.y")

    # ---- CTE edge cases ----

    def test_cte_cannot_mask_system_table_access(self):
        """A CTE named after a system table should not allow querying the real system table."""
        s = self._store_with_data()
        # The CTE itself references t (allowed), so this should work —
        # the CTE name 'sqlite_master' just shadows, not accesses, the real one.
        df = s.query_table(
            "WITH sqlite_master AS (SELECT x FROM t) SELECT * FROM sqlite_master"
        )
        assert df["x"] == [1, 2]

    def test_cte_body_referencing_unregistered_table_blocked(self):
        """The body of a CTE must also be validated for table references."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query("WITH leaked AS (SELECT * FROM sqlite_master) SELECT * FROM leaked")

    # ---- Multiple statement bypass attempts ----

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; SELECT 2",
            "SELECT x FROM t; ATTACH ':memory:' AS db2",
            "SELECT x FROM t;\nDROP TABLE t",
        ],
    )
    def test_multiple_statements_various_forms(self, sql):
        """Multiple statements in any form must be rejected."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            s.query(sql)


class TestSQLiteDefenseInDepth:
    """Tests that SQLite-level safeguards catch attacks even if sqlglot is bypassed.

    These tests patch _validate_sql to be a no-op, proving that the
    authorizer, query_only pragma, and connection config provide
    independent protection.
    """

    def _store_with_data(self):
        s = DataFrameStore()
        s.store("t", [{"x": 1}, {"x": 2}])
        return s

    def _query_bypassing_sqlglot(self, store, sql):
        """Run a query with sqlglot validation disabled."""
        with patch.object(
            DataFrameStore,
            "_validate_sql",
            return_value=sql.strip().rstrip(";").strip(),
        ):
            return store.query(sql)

    def _query_table_bypassing_sqlglot(self, store, sql):
        """Run a query_table with sqlglot validation disabled."""
        with patch.object(
            DataFrameStore,
            "_validate_sql",
            return_value=sql.strip().rstrip(";").strip(),
        ):
            return store.query_table(sql)

    # ---- Authorizer blocks write statements ----

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO t VALUES (99)",
            "UPDATE t SET x = 99",
            "DELETE FROM t",
            "DROP TABLE t",
            "CREATE TABLE evil (a TEXT)",
        ],
    )
    def test_authorizer_blocks_writes(self, sql):
        """Write operations are blocked by the authorizer even without sqlglot."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            self._query_bypassing_sqlglot(s, sql)

    def test_authorizer_blocks_attach(self):
        """ATTACH is blocked by the authorizer even without sqlglot."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            self._query_bypassing_sqlglot(s, "ATTACH ':memory:' AS other")

    def test_authorizer_blocks_unlisted_function(self):
        """Functions not in the allowlist are blocked by the authorizer."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            self._query_bypassing_sqlglot(s, "SELECT load_extension('/tmp/evil.so')")

    # ---- query_only pragma blocks writes ----

    def test_query_only_blocks_insert(self):
        """PRAGMA query_only prevents INSERT even if authorizer were absent."""
        s = self._store_with_data()
        with pytest.raises((ValueError, Exception)):
            self._query_bypassing_sqlglot(s, "INSERT INTO t VALUES (99)")

    # ---- Allowed queries still work with all safeguards active ----

    def test_select_still_works(self):
        """Normal SELECT queries work with all safeguards active."""
        s = self._store_with_data()
        result = self._query_table_bypassing_sqlglot(s, "SELECT x FROM t ORDER BY x")
        assert result["x"] == [1, 2]

    def test_aggregate_functions_work(self):
        """Aggregate functions in the allowlist work through the authorizer."""
        s = self._store_with_data()
        result = self._query_table_bypassing_sqlglot(
            s, "SELECT COUNT(*) AS cnt, SUM(x) AS total FROM t"
        )
        assert result["cnt"][0] == 2
        assert result["total"][0] == 3

    def test_custom_functions_work(self):
        """Custom registered functions (SQRT, etc.) work through the authorizer."""
        s = self._store_with_data()
        result = self._query_table_bypassing_sqlglot(
            s, "SELECT SQRT(x) AS root FROM t WHERE x = 4"
        )
        # x=4 doesn't exist; test with existing data
        result = self._query_table_bypassing_sqlglot(
            s, "SELECT POWER(x, 2) AS sq FROM t ORDER BY x"
        )
        assert result["sq"] == [1.0, 4.0]

    def test_window_functions_work(self):
        """Window functions work through the authorizer."""
        s = self._store_with_data()
        result = self._query_table_bypassing_sqlglot(
            s, "SELECT x, ROW_NUMBER() OVER (ORDER BY x) AS rn FROM t"
        )
        assert result["rn"] == [1, 2]


class TestCustomTimestampFunctions:
    """Tests for to_timestamp() and to_date() SQL functions."""

    def _make_store(self):
        s = DataFrameStore()
        s.store(
            "data",
            [
                {"t_ms": 1710460800000, "t_ns": 1710460800000000000, "t_s": 1710460800},
                {"t_ms": 1710374400000, "t_ns": 1710374400000000000, "t_s": 1710374400},
            ],
        )
        return s

    def test_to_timestamp_epoch_ms(self):
        s = self._make_store()
        result = s.query("SELECT to_timestamp(t_ms) as ts FROM data LIMIT 1")
        assert "2024-03-15T00:00:00Z" in result

    def test_to_timestamp_epoch_ns(self):
        s = self._make_store()
        result = s.query("SELECT to_timestamp(t_ns) as ts FROM data LIMIT 1")
        assert "2024-03-15" in result

    def test_to_timestamp_epoch_seconds(self):
        s = self._make_store()
        result = s.query("SELECT to_timestamp(t_s) as ts FROM data LIMIT 1")
        assert "2024-03-15T00:00:00Z" in result

    def test_to_timestamp_null(self):
        s = DataFrameStore()
        s.store("nulls", [{"t": None}])
        result = s.query("SELECT to_timestamp(t) as ts FROM nulls")
        assert "ts" in result  # header exists
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 row

    def test_to_date_epoch_ms(self):
        s = self._make_store()
        result = s.query("SELECT to_date(t_ms) as d FROM data LIMIT 1")
        assert "2024-03-15" in result

    def test_to_date_epoch_ns(self):
        s = self._make_store()
        result = s.query("SELECT to_date(t_ns) as d FROM data LIMIT 1")
        assert "2024-03-15" in result

    def test_to_date_cross_asset_join(self):
        """to_date() enables JOINs across assets with different timestamp offsets."""
        s = DataFrameStore()
        # Crypto midnight UTC vs equity 4am ET (4-hour offset)
        s.store("btc", [{"t": 1710460800000, "close": 71250.0}])
        s.store("spy", [{"t": 1710475200000, "close": 512.5}])

        # Raw JOIN fails
        raw = s.query("SELECT * FROM btc JOIN spy ON btc.t = spy.t")
        raw_rows = raw.strip().split("\n")
        assert len(raw_rows) <= 2  # header only or header + 0 rows

        # to_date JOIN succeeds
        result = s.query(
            "SELECT to_date(btc.t) as d, btc.close as btc, spy.close as spy "
            "FROM btc JOIN spy ON to_date(btc.t) = to_date(spy.t)"
        )
        assert "71250" in result
        assert "512.5" in result

    def test_to_date_iso_string_passthrough(self):
        s = DataFrameStore()
        s.store("opts", [{"expiry": "2026-04-17T00:00:00Z", "strike": 260}])
        result = s.query("SELECT to_date(expiry) as d, strike FROM opts")
        assert "2026-04-17" in result


class TestCorrAggregate:
    """Tests for the CORR() SQL aggregate function."""

    def test_perfect_positive_correlation(self):
        s = DataFrameStore()
        s.store("data", [{"x": i, "y": i * 2} for i in range(10)])
        result = s.query("SELECT CORR(x, y) as c FROM data")
        assert "1.0" in result

    def test_perfect_negative_correlation(self):
        s = DataFrameStore()
        s.store("data", [{"x": i, "y": -i} for i in range(10)])
        result = s.query("SELECT CORR(x, y) as c FROM data")
        assert "-1.0" in result

    def test_zero_correlation(self):
        """Orthogonal data should produce near-zero correlation."""
        s = DataFrameStore()
        s.store("data", [
            {"x": 1, "y": 0}, {"x": -1, "y": 0},
            {"x": 0, "y": 1}, {"x": 0, "y": -1},
        ])
        result = s.query("SELECT CORR(x, y) as c FROM data")
        val = float(result.strip().split("\n")[1])
        assert abs(val) < 0.01

    def test_corr_with_nulls(self):
        """NULL pairs should be skipped."""
        s = DataFrameStore()
        s.store("data", [
            {"x": 1, "y": 2}, {"x": None, "y": 5},
            {"x": 3, "y": 6}, {"x": 4, "y": None},
            {"x": 5, "y": 10},
        ])
        # Should only use the 3 complete pairs: (1,2), (3,6), (5,10)
        result = s.query("SELECT CORR(x, y) as c FROM data")
        val = float(result.strip().split("\n")[1])
        assert val > 0.99  # strong positive

    def test_corr_insufficient_data(self):
        """Fewer than 2 complete pairs should return NULL."""
        s = DataFrameStore()
        s.store("data", [{"x": 1, "y": 2}])
        result = s.query("SELECT CORR(x, y) as c FROM data")
        lines = result.strip().split("\n")
        assert lines[1].strip('"') == ""  # NULL


class TestReservedTableNames:
    """Tests for _RESERVED_TABLE_NAMES blocking."""

    @pytest.mark.parametrize(
        "name",
        [
            "sqlite_master",
            "sqlite_sequence",
            "sqlite_stat1",
        ],
    )
    def test_cannot_store_reserved_name(self, name):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="reserved"):
            s.store(name, [{"x": 1}])

    @pytest.mark.parametrize(
        "name",
        ["sqlite_master", "sqlite_sequence"],
    )
    def test_cannot_store_table_reserved_name(self, name):
        s = DataFrameStore()
        with pytest.raises(ValueError, match="reserved"):
            s.store_table(name, Table(["x"], {"x": [1]}))

    @pytest.mark.parametrize(
        "name",
        ["prices", "my_data", "agg_results", "ticker_info"],
    )
    def test_normal_names_still_work(self, name):
        s = DataFrameStore()
        summary = s.store(name, [{"x": 1}])
        assert summary.table_name == name


class TestQueryTimeout:
    """Tests for the query execution timeout."""

    def test_fast_query_succeeds(self):
        s = DataFrameStore()
        s.store("t", [{"x": i} for i in range(10)])
        df = s.query_table("SELECT SUM(x) AS total FROM t")
        assert df["total"][0] == 45

    def test_timeout_raises(self):
        s = DataFrameStore()
        # Store enough data to make a cross-join expensive
        s.store("t", [{"x": i} for i in range(500)])

        with patch("mcp_massive.store.QUERY_TIMEOUT_SECONDS", 0.001):
            with pytest.raises((TimeoutError, Exception)):
                # A 4-way cross join on 500 rows = 500^4 = 62.5 billion rows
                s.query(
                    "SELECT COUNT(*) FROM t CROSS JOIN t t2 CROSS JOIN t t3 CROSS JOIN t t4"
                )
