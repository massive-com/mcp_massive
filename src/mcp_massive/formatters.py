import json
import csv
import io
from typing import Any


def strip_response_metadata(json_text: str, exclude_keys: set) -> str:
    """Strip metadata keys from a JSON response string.

    Parses the JSON, removes top-level keys in exclude_keys, and re-serializes.
    """
    data = json.loads(json_text)
    if isinstance(data, dict):
        for key in exclude_keys:
            data.pop(key, None)
    return json.dumps(data)


def extract_records(data: str | dict | list) -> list[dict]:
    """Extract and flatten records from raw JSON input.

    Takes raw JSON input (string or parsed), extracts the records list
    (handling 'results', 'last', list, and single-object cases), flattens
    each record via _flatten_dict, and returns a list of flat dicts.

    Args:
        data: JSON string, dict, or list.

    Returns:
        List of flattened dictionaries.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict) and "results" in data:
        results_value = data["results"]
        if isinstance(results_value, list):
            records = results_value
        elif isinstance(results_value, dict):
            records = [results_value]
        else:
            records = [results_value]
    elif isinstance(data, dict) and "last" in data:
        records = [data["last"]] if isinstance(data["last"], dict) else [data]
    elif isinstance(data, list):
        records = data
    else:
        records = [data]

    flattened_records = []
    for record in records:
        if isinstance(record, dict):
            flattened_records.append(_flatten_dict(record))
        else:
            flattened_records.append({"value": str(record)})

    return flattened_records


def json_to_csv(json_input: str | dict | list) -> str:
    """
    Convert JSON to flattened CSV format.

    Args:
        json_input: JSON string or dict. If the JSON has a 'results' key containing
                   a list, it will be extracted. Otherwise, the entire structure
                   will be wrapped in a list for processing.

    Returns:
        CSV string with headers and flattened rows
    """
    flattened_records = extract_records(json_input)

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
