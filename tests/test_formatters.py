import json
import csv
import io

import pytest

from mcp_massive.formatters import json_to_csv, _flatten_dict


class TestFlattenDict:
    """Tests for the _flatten_dict helper function."""

    def test_flat_dict(self):
        """Test that flat dictionaries are unchanged."""
        input_dict = {"a": 1, "b": 2, "c": 3}
        result = _flatten_dict(input_dict)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_nested_dict(self):
        """Test flattening nested dictionaries."""
        input_dict = {"outer": {"inner": "value"}}
        result = _flatten_dict(input_dict)
        assert result == {"outer_inner": "value"}

    def test_deeply_nested_dict(self):
        """Test flattening deeply nested dictionaries."""
        input_dict = {"level1": {"level2": {"level3": "value"}}}
        result = _flatten_dict(input_dict)
        assert result == {"level1_level2_level3": "value"}

    def test_dict_with_list(self):
        """Test that lists are converted to strings."""
        input_dict = {"items": [1, 2, 3], "names": ["alice", "bob"]}
        result = _flatten_dict(input_dict)
        assert result == {"items": "[1, 2, 3]", "names": "['alice', 'bob']"}

    def test_mixed_nested_structure(self):
        """Test flattening mixed nested structures."""
        input_dict = {
            "simple": "value",
            "nested": {"field1": 100, "field2": "text"},
            "list_field": ["a", "b"],
        }
        result = _flatten_dict(input_dict)
        assert result == {
            "simple": "value",
            "nested_field1": 100,
            "nested_field2": "text",
            "list_field": "['a', 'b']",
        }

    def test_empty_dict(self):
        """Test flattening empty dictionary."""
        result = _flatten_dict({})
        assert result == {}

    def test_nested_empty_dict(self):
        """Test flattening dictionary with empty nested dict."""
        input_dict = {"outer": {}}
        result = _flatten_dict(input_dict)
        assert result == {}

    def test_none_values(self):
        """Test that None values are preserved."""
        input_dict = {"field": None, "nested": {"inner": None}}
        result = _flatten_dict(input_dict)
        assert result == {"field": None, "nested_inner": None}


class TestJsonToCsvStdlib:
    """Tests for the json_to_csv_stdlib function."""

    def test_options_contract_response(self):
        """Test conversion of real options contract API response."""
        json_input = {
            "results": [
                {
                    "break_even_price": 442.65,
                    "day": {
                        "change": 4.173,
                        "change_percent": 1.072,
                        "close": 393.613,
                        "high": 393.667,
                        "last_updated": 1759204800000000000,
                        "low": 383.038,
                        "open": 392.638,
                        "previous_close": 389.44,
                        "volume": 3,
                        "vwap": 388.41833,
                    },
                    "details": {
                        "contract_type": "call",
                        "exercise_style": "american",
                        "expiration_date": "2025-10-03",
                        "shares_per_contract": 100,
                        "strike_price": 50,
                        "ticker": "O:TSLA251003C00050000",
                    },
                    "greeks": {},
                    "open_interest": 1,
                    "underlying_asset": {
                        "change_to_break_even": -2.22,
                        "last_updated": 1759262178000773000,
                        "price": 444.865,
                        "ticker": "TSLA",
                        "timeframe": "REAL-TIME",
                    },
                    "fmv": 393.416,
                    "fmv_last_updated": 1759262177475530000,
                }
            ]
        }

        results = json_to_csv(json_input)

        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # Check flattened fields
        assert row["break_even_price"] == "442.65"
        assert row["day_change"] == "4.173"
        assert row["day_close"] == "393.613"
        assert row["details_contract_type"] == "call"
        assert row["details_ticker"] == "O:TSLA251003C00050000"
        assert row["underlying_asset_ticker"] == "TSLA"
        assert row["fmv"] == "393.416"

    def test_multiple_options_contracts(self):
        """Test conversion of multiple options contracts."""
        json_input = {
            "results": [
                {
                    "break_even_price": 442.65,
                    "details": {"strike_price": 50, "ticker": "O:TSLA251003C00050000"},
                    "open_interest": 1,
                },
                {
                    "break_even_price": 442.825,
                    "details": {"strike_price": 60, "ticker": "O:TSLA251003C00060000"},
                    "open_interest": 1,
                },
            ]
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["details_strike_price"] == "50"
        assert rows[1]["details_strike_price"] == "60"

    def test_market_status_response(self):
        """Test conversion of market status API response."""
        json_input = {
            "afterHours": False,
            "currencies": {"crypto": "open", "fx": "open"},
            "earlyHours": False,
            "exchanges": {"nasdaq": "open", "nyse": "open", "otc": "open"},
            "indicesGroups": {"s_and_p": "open", "dow_jones": "open"},
            "market": "open",
            "serverTime": "2025-10-01T11:36:28-04:00",
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # Check flattened nested fields
        assert row["afterHours"] == "False"
        assert row["currencies_crypto"] == "open"
        assert row["currencies_fx"] == "open"
        assert row["exchanges_nasdaq"] == "open"
        assert row["exchanges_nyse"] == "open"
        assert row["market"] == "open"
        assert row["serverTime"] == "2025-10-01T11:36:28-04:00"

    def test_json_string_input(self):
        """Test that JSON strings are properly parsed."""
        json_string = '{"results": [{"ticker": "AAPL", "price": 150.5}]}'
        results = json_to_csv(json_string)

        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["price"] == "150.5"

    def test_json_dict_input(self):
        """Test that dict input works directly."""
        json_dict = {"results": [{"ticker": "AAPL", "price": 150.5}]}
        results = json_to_csv(json_dict)

        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["price"] == "150.5"

    def test_list_input_without_results_key(self):
        """Test that a list without 'results' wrapper is handled."""
        json_input = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"

    def test_single_object_without_results_key(self):
        """Test that a single object is wrapped in a list."""
        json_input = {"ticker": "AAPL", "price": 150.5}
        results = json_to_csv(json_input)

        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["price"] == "150.5"

    def test_empty_results_list(self):
        """Test handling of empty results list."""
        json_input = {"results": []}
        results = json_to_csv(json_input)
        assert results == ""

    def test_empty_list(self):
        """Test handling of empty list input."""
        results = json_to_csv([])
        assert results == ""

    def test_inconsistent_fields_across_rows(self):
        """Test that rows with different fields are handled correctly."""
        json_input = {
            "results": [
                {"ticker": "AAPL", "price": 150.5, "volume": 1000},
                {"ticker": "GOOGL", "price": 2800.0},
                {"ticker": "MSFT", "extra_field": "value"},
            ]
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 3

        headers = reader.fieldnames
        assert "ticker" in headers
        assert "price" in headers
        assert "volume" in headers
        assert "extra_field" in headers

        # Missing fields should be empty strings
        assert rows[1]["volume"] == ""
        assert rows[0]["extra_field"] == ""

    def test_special_characters_in_values(self):
        """Test handling of special characters in CSV values."""
        json_input = {
            "results": [
                {
                    "name": "Company, Inc.",
                    "description": 'A company with "quotes"',
                    "note": "Line 1\nLine 2",
                }
            ]
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        # CSV library should handle escaping
        assert "Company, Inc." in rows[0]["name"]
        assert (
            '"quotes"' in rows[0]["description"] or "quotes" in rows[0]["description"]
        )

    def test_numeric_types_preserved(self):
        """Test that various numeric types are handled."""
        json_input = {
            "results": [
                {
                    "integer": 42,
                    "float": 3.14159,
                    "negative": -100,
                    "scientific": 1.23e-4,
                    "large": 1759204800000000000,
                }
            ]
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["integer"] == "42"
        assert row["float"] == "3.14159"
        assert row["negative"] == "-100"
        assert "1.23e-04" in row["scientific"] or "0.000123" in row["scientific"]
        assert row["large"] == "1759204800000000000"

    def test_boolean_values(self):
        """Test handling of boolean values."""
        json_input = {"results": [{"active": True, "archived": False}]}

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["active"] == "True"
        assert rows[0]["archived"] == "False"

    def test_null_values(self):
        """Test handling of null/None values."""
        json_input = {
            "results": [{"field1": "value", "field2": None, "field3": "another"}]
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["field1"] == "value"
        assert rows[0]["field2"] == "" or rows[0]["field2"] == "None"
        assert rows[0]["field3"] == "another"

    def test_empty_nested_dict(self):
        """Test handling of empty nested dictionaries."""
        json_input = {"results": [{"ticker": "AAPL", "greeks": {}, "price": 150.5}]}

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["price"] == "150.5"
        # Empty dict should not add any columns

    def test_list_with_nested_objects(self):
        """Test that lists containing objects are stringified."""
        json_input = {
            "results": [{"id": 1, "tags": [{"name": "tag1"}, {"name": "tag2"}]}]
        }

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        # Lists are converted to strings
        assert "tag1" in rows[0]["tags"]
        assert "tag2" in rows[0]["tags"]

    def test_invalid_json_string(self):
        """Test that invalid JSON string raises appropriate error."""
        with pytest.raises(json.JSONDecodeError):
            json_to_csv("not valid json {")

    def test_csv_output_format(self):
        """Test that output is valid CSV with proper headers."""
        json_input = {
            "results": [
                {"col1": "value1", "col2": "value2"},
                {"col1": "value3", "col2": "value4"},
            ]
        }

        results = json_to_csv(json_input)
        lines = [result.strip() for result in results.strip().split("\n")]

        # Should have header + 2 data rows
        assert len(lines) == 3
        assert lines[0] == "col1,col2"
        assert lines[1] == "value1,value2"
        assert lines[2] == "value3,value4"

    def test_field_ordering_consistency(self):
        """Test that field ordering is consistent across rows."""
        json_input = {
            "results": [
                {"z": 1, "a": 2, "m": 3},
                {"a": 4, "z": 5, "m": 6},
                {"m": 7, "z": 8, "a": 9},
            ]
        }

        results = json_to_csv(json_input)
        lines = [result.strip() for result in results.strip().split("\n")]

        # Header should maintain first-seen order
        header = lines[0]
        fields = header.split(",")
        assert fields == ["z", "a", "m"]

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        json_input = {"results": [{"name": "Café", "symbol": "€", "emoji": "🚀"}]}

        results = json_to_csv(json_input)
        reader = csv.DictReader(io.StringIO(results))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["name"] == "Café"
        assert rows[0]["symbol"] == "€"
        assert rows[0]["emoji"] == "🚀"
