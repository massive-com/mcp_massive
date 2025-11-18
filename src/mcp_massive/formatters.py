import json
import csv
import io
from typing import Any


def json_to_csv(json_input: str | dict) -> str:
    """
    Convert JSON to flattened CSV format.

    Args:
        json_input: JSON string or dict. If the JSON has a 'results' key containing
                   a list, it will be extracted. Otherwise, the entire structure
                   will be wrapped in a list for processing.

    Returns:
        CSV string with headers and flattened rows
    """
    # Parse JSON if it's a string
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty CSV
            return ""
    else:
        data = json_input

    if isinstance(data, dict) and "results" in data:
        results_value = data["results"]
        # Handle both list and single object responses
        if isinstance(results_value, list):
            records = results_value
        elif isinstance(results_value, dict):
            # Single object response (e.g., get_last_trade returns results as object)
            records = [results_value]
        else:
            records = [results_value]
    elif isinstance(data, dict) and "last" in data:
        # Handle responses with "last" key (e.g., get_last_trade, get_last_quote)
        records = [data["last"]] if isinstance(data["last"], dict) else [data]
    elif isinstance(data, list):
        records = data
    else:
        records = [data]

    # Only flatten dict records, skip non-dict items
    flattened_records = []
    for record in records:
        if isinstance(record, dict):
            flattened_records.append(_flatten_dict(record))
        else:
            # If it's not a dict, wrap it in a dict with a 'value' key
            flattened_records.append({"value": str(record)})

    if not flattened_records:
        return ""

    # Get all unique keys across all records (for consistent column ordering)
    all_keys = []
    seen = set()
    for record in flattened_records:
        if isinstance(record, dict):
            for key in record.keys():
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened_records)

    return output.getvalue()


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """
    Flatten a nested dictionary by joining keys with separator.

    Args:
        d: Dictionary to flatten
        parent_key: Key from parent level (for recursion)
        sep: Separator to use between nested keys

    Returns:
        Flattened dictionary with no nested structures
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # Recursively flatten nested dicts
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))

    return dict(items)
