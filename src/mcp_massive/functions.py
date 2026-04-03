"""Finance functions for post-processing Tables.

Provides a closed registry of pre-defined functions (Greeks, returns, technicals)
that can be applied to Tables via the ``apply`` parameter on ``call_api`` and
``query_data``.  No arbitrary code execution — only named functions with
column/literal inputs.
"""

import math
import re
import sqlite3
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .index import _expand_query
from .store import Table

_VALID_OPTION_TYPES = frozenset({"call", "put"})
_OUTPUT_COL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")
MAX_APPLY_STEPS = 20


# ---------------------------------------------------------------------------
# Parameter model
# ---------------------------------------------------------------------------


class ParamKind(Enum):
    """How a parameter value is interpreted."""

    COLUMN = "column"  # string → column name
    LITERAL = "literal"  # numeric literal only
    COL_OR_LIT = "col_or_lit"  # string → column, number → literal
    LITERAL_STR = "literal_str"  # string treated as literal (not column ref)


class ParamDef(BaseModel):
    name: str = Field(min_length=1)
    kind: ParamKind
    required: bool = True
    default: Any = None
    description: str = ""


class FunctionDef(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str
    params: list[ParamDef]
    output_dtype: str  # e.g. "Float64" — for display only
    impl: Callable  # (table, resolved_inputs) -> np.ndarray
    search_text: str = ""

    @model_validator(mode="after")
    def _build_search_text(self) -> "FunctionDef":
        parts = [
            self.name,
            self.name,
            self.name,
            self.category,
            self.category,
            self.description,
        ]
        for p in self.params:
            parts.append(p.name)
            if p.description:
                parts.append(p.description)
        self.search_text = " ".join(parts)
        return self

    def signature(self) -> str:
        """Human-readable signature string."""
        params = []
        for p in self.params:
            s = p.name
            if not p.required:
                s += "?"
            params.append(s)
        return f"{self.name}({', '.join(params)}) -> {self.output_dtype}"

    def full_description(self) -> str:
        """Full description with signature and param docs for search results."""
        lines = [
            f"{self.name}({', '.join(p.name + ('?' if not p.required else '') for p in self.params)}) -> {self.output_dtype}",
            self.description,
        ]
        for p in self.params:
            kind_hint = p.kind.value
            default_hint = f", default {p.default!r}" if p.default is not None else ""
            req = "required" if p.required else "optional"
            desc = f" — {p.description}" if p.description else ""
            lines.append(f"  {p.name} ({kind_hint}, {req}{default_hint}){desc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

FUNCTION_REGISTRY: dict[str, FunctionDef] = {}


def _register(func_def: FunctionDef) -> FunctionDef:
    FUNCTION_REGISTRY[func_def.name] = func_def
    return func_def


# ---------------------------------------------------------------------------
# Normal CDF / PDF
# ---------------------------------------------------------------------------

_SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF using math.erfc."""
    return 0.5 * np.vectorize(math.erfc)(-x / _SQRT_2)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def resolve_input(
    table: Table, value: Any, kind: ParamKind
) -> np.ndarray | float | str:
    """Resolve a single input value to a numpy array or scalar.

    - COLUMN: string → np.array(table[value], dtype=float64)
    - LITERAL: must be numeric
    - COL_OR_LIT: string → column as array, number → literal
    - LITERAL_STR: string stays as literal string
    """
    if kind == ParamKind.COLUMN:
        if not isinstance(value, str):
            raise ValueError(
                f"COLUMN param expects string column name, got {type(value).__name__}"
            )
        if value not in table.data:
            raise ValueError(f"Column '{value}' not found. Available: {table.columns}")
        return np.array(table.get_column(value), dtype=np.float64)
    elif kind == ParamKind.LITERAL:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"LITERAL param expects numeric value, got {type(value).__name__}: {value!r}"
            )
        return float(value)
    elif kind == ParamKind.COL_OR_LIT:
        if isinstance(value, str):
            if value not in table.data:
                raise ValueError(
                    f"Column '{value}' not found. Available: {table.columns}"
                )
            return np.array(table.get_column(value), dtype=np.float64)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise ValueError(
                f"COL_OR_LIT param expects string or number, got {type(value).__name__}"
            )
    elif kind == ParamKind.LITERAL_STR:
        return str(value)
    else:
        raise ValueError(f"Unknown ParamKind: {kind}")


def _to_numpy(val: np.ndarray | float, length: int) -> np.ndarray:
    """Convert a resolved input (array or scalar) to a numpy array."""
    if isinstance(val, (int, float)):
        return np.full(length, val, dtype=np.float64)
    return val.astype(np.float64)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------


def _validate_option_type(inputs: dict[str, Any]) -> str:
    """Extract and validate option_type from resolved inputs."""
    ot = inputs.get("option_type", "call")
    if ot not in _VALID_OPTION_TYPES:
        raise ValueError(f"Invalid option_type '{ot}'. Must be 'call' or 'put'.")
    return ot


def _bs_d1d2(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
):
    """Compute d1 and d2 for Black-Scholes."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ---------------------------------------------------------------------------
# Greeks implementations
# ---------------------------------------------------------------------------


def _impl_bs_price(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    n = len(table)
    S = _to_numpy(inputs["S"], n)
    K = _to_numpy(inputs["K"], n)
    T = _to_numpy(inputs["T"], n)
    r = _to_numpy(inputs["r"], n)
    sigma = _to_numpy(inputs["sigma"], n)
    option_type = _validate_option_type(inputs)

    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    if option_type == "call":
        price = S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return price


def _impl_bs_delta(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    n = len(table)
    S = _to_numpy(inputs["S"], n)
    K = _to_numpy(inputs["K"], n)
    T = _to_numpy(inputs["T"], n)
    r = _to_numpy(inputs["r"], n)
    sigma = _to_numpy(inputs["sigma"], n)
    option_type = _validate_option_type(inputs)

    d1, _d2 = _bs_d1d2(S, K, T, r, sigma)
    if option_type == "call":
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0
    return delta


def _impl_bs_gamma(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    n = len(table)
    S = _to_numpy(inputs["S"], n)
    K = _to_numpy(inputs["K"], n)
    T = _to_numpy(inputs["T"], n)
    r = _to_numpy(inputs["r"], n)
    sigma = _to_numpy(inputs["sigma"], n)

    d1, _d2 = _bs_d1d2(S, K, T, r, sigma)
    gamma = _norm_pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def _impl_bs_theta(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    n = len(table)
    S = _to_numpy(inputs["S"], n)
    K = _to_numpy(inputs["K"], n)
    T = _to_numpy(inputs["T"], n)
    r = _to_numpy(inputs["r"], n)
    sigma = _to_numpy(inputs["sigma"], n)
    option_type = _validate_option_type(inputs)

    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    common = -(S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == "call":
        theta = common - r * K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        theta = common + r * K * np.exp(-r * T) * _norm_cdf(-d2)
    # Return daily theta (divide annual by 365)
    return theta / 365.0


def _impl_bs_vega(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    n = len(table)
    S = _to_numpy(inputs["S"], n)
    K = _to_numpy(inputs["K"], n)
    T = _to_numpy(inputs["T"], n)
    r = _to_numpy(inputs["r"], n)
    sigma = _to_numpy(inputs["sigma"], n)

    d1, _d2 = _bs_d1d2(S, K, T, r, sigma)
    vega = S * _norm_pdf(d1) * np.sqrt(T)
    # Per 1% change in volatility
    return vega / 100.0


def _impl_bs_rho(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    n = len(table)
    S = _to_numpy(inputs["S"], n)
    K = _to_numpy(inputs["K"], n)
    T = _to_numpy(inputs["T"], n)
    r = _to_numpy(inputs["r"], n)
    sigma = _to_numpy(inputs["sigma"], n)
    option_type = _validate_option_type(inputs)

    _d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * _norm_cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * _norm_cdf(-d2)
    # Per 1% rate change
    return rho / 100.0


# ---------------------------------------------------------------------------
# Returns implementations (numpy)
# ---------------------------------------------------------------------------


def _impl_simple_return(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"]  # already a np.ndarray via resolve_input
    if len(arr) == 0:
        return arr
    shifted = np.empty_like(arr)
    shifted[0] = np.nan
    shifted[1:] = arr[:-1]
    result = (arr - shifted) / shifted
    return result


def _impl_log_return(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"].astype(np.float64)
    if len(arr) == 0:
        return arr
    shifted = np.empty_like(arr)
    shifted[0] = np.nan
    shifted[1:] = arr[:-1]
    result = np.log(arr / shifted)
    return result


def _impl_cumulative_return(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"]
    if len(arr) == 0:
        return arr
    shifted = np.empty_like(arr)
    shifted[0] = np.nan
    shifted[1:] = arr[:-1]
    pct = (arr - shifted) / shifted
    # First value is NaN — set 1+pct[0] = 1 so cumprod starts clean,
    # then mark result[0] = NaN afterward.
    one_plus = 1.0 + pct
    one_plus[0] = 1.0  # neutral element for cumprod
    result = np.cumprod(one_plus) - 1.0
    result[0] = np.nan
    return result


# ---------------------------------------------------------------------------
# Risk-adjusted returns implementations (numpy)
# ---------------------------------------------------------------------------


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with NaN padding for incomplete windows."""
    n = len(arr)
    result = np.full(n, np.nan)
    cumsum = np.zeros(n + 1)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + (arr[i] if not np.isnan(arr[i]) else 0.0)
    for i in range(window - 1, n):
        # Check if any value in window is NaN
        has_nan = False
        for j in range(i - window + 1, i + 1):
            if np.isnan(arr[j]):
                has_nan = True
                break
        if not has_nan:
            result[i] = (cumsum[i + 1] - cumsum[i - window + 1]) / window
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling sample standard deviation with NaN padding.

    Uses sample std (ddof=1), so windows smaller than 2 are undefined and
    return all-NaN without invoking numpy std (avoids runtime warnings).
    """
    n = len(arr)
    result = np.full(n, np.nan)
    if window < 2:
        return result
    for i in range(window - 1, n):
        window_data = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(window_data)):
            continue
        result[i] = np.std(window_data, ddof=1)
    return result


def _impl_sharpe_ratio(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"]
    window = int(inputs["window"])
    rf = float(inputs.get("rf", 0.0))
    if window < 2:
        raise ValueError("Sharpe ratio window must be >= 2")
    mean = _rolling_mean(arr, window)
    std = _rolling_std(arr, window)
    return (mean - rf) / std


def _impl_sortino_ratio(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"]
    window = int(inputs["window"])
    rf = float(inputs.get("rf", 0.0))
    if window < 2:
        raise ValueError("Sortino ratio window must be >= 2")
    mean = _rolling_mean(arr, window)
    downside_sq = np.where(arr - rf < 0, (arr - rf) ** 2, 0.0)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_data = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(window_data)):
            continue
        ds_window = downside_sq[i - window + 1 : i + 1]
        ds_mean = np.mean(ds_window)
        if ds_mean > 0:
            ds_std = math.sqrt(ds_mean)
            result[i] = (mean[i] - rf) / ds_std
        else:
            result[i] = np.nan  # No downside → undefined
    return result


# ---------------------------------------------------------------------------
# Technicals implementations (numpy)
# ---------------------------------------------------------------------------


def _impl_sma(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"]
    window = int(inputs["window"])
    if window < 1:
        raise ValueError("SMA window must be >= 1")
    return _rolling_mean(arr, window)


def _impl_ema(table: Table, inputs: dict[str, Any]) -> np.ndarray:
    arr = inputs["column"]
    span = int(inputs["window"])
    if span < 1:
        raise ValueError("EMA span must be >= 1")
    n = len(arr)
    if n == 0:
        return arr
    alpha = 2.0 / (span + 1.0)
    result = np.full(n, np.nan)
    # Find the first non-NaN value to seed the EMA
    seed_idx = -1
    for i in range(n):
        if not np.isnan(arr[i]):
            seed_idx = i
            break
    if seed_idx == -1:
        return result  # all NaN
    result[seed_idx] = arr[seed_idx]
    for i in range(seed_idx + 1, n):
        if np.isnan(arr[i]):
            # Propagate previous EMA through NaN gaps
            result[i] = result[i - 1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


# ---------------------------------------------------------------------------
# Register all functions
# ---------------------------------------------------------------------------

_BS_COMMON_PARAMS = [
    ParamDef(
        name="S",
        kind=ParamKind.COL_OR_LIT,
        description="Spot price (column or literal)",
    ),
    ParamDef(
        name="K",
        kind=ParamKind.COL_OR_LIT,
        description="Strike price (column or literal)",
    ),
    ParamDef(
        name="T",
        kind=ParamKind.COL_OR_LIT,
        description="Years to expiry (column or literal)",
    ),
    ParamDef(
        name="r",
        kind=ParamKind.COL_OR_LIT,
        description="Risk-free rate (column or literal)",
    ),
    ParamDef(
        name="sigma",
        kind=ParamKind.COL_OR_LIT,
        description="Implied volatility (column or literal)",
    ),
]

_BS_OPTION_TYPE_PARAM = ParamDef(
    name="option_type",
    kind=ParamKind.LITERAL_STR,
    required=False,
    default="call",
    description='Option type: "call" or "put"',
)

_register(
    FunctionDef(
        name="bs_price",
        category="Greeks",
        description="Black-Scholes option price.",
        params=[*_BS_COMMON_PARAMS, _BS_OPTION_TYPE_PARAM],
        output_dtype="Float64",
        impl=_impl_bs_price,
    )
)

_register(
    FunctionDef(
        name="bs_delta",
        category="Greeks",
        description="Black-Scholes delta. N(d1) for calls, N(d1)-1 for puts.",
        params=[*_BS_COMMON_PARAMS, _BS_OPTION_TYPE_PARAM],
        output_dtype="Float64",
        impl=_impl_bs_delta,
    )
)

_register(
    FunctionDef(
        name="bs_gamma",
        category="Greeks",
        description="Black-Scholes gamma. Same for calls and puts.",
        params=list(_BS_COMMON_PARAMS),
        output_dtype="Float64",
        impl=_impl_bs_gamma,
    )
)

_register(
    FunctionDef(
        name="bs_theta",
        category="Greeks",
        description="Black-Scholes daily theta (annual theta / 365).",
        params=[*_BS_COMMON_PARAMS, _BS_OPTION_TYPE_PARAM],
        output_dtype="Float64",
        impl=_impl_bs_theta,
    )
)

_register(
    FunctionDef(
        name="bs_vega",
        category="Greeks",
        description="Black-Scholes vega per 1% change in volatility. Same for calls and puts.",
        params=list(_BS_COMMON_PARAMS),
        output_dtype="Float64",
        impl=_impl_bs_vega,
    )
)

_register(
    FunctionDef(
        name="bs_rho",
        category="Greeks",
        description="Black-Scholes rho per 1% change in interest rate.",
        params=[*_BS_COMMON_PARAMS, _BS_OPTION_TYPE_PARAM],
        output_dtype="Float64",
        impl=_impl_bs_rho,
    )
)

_register(
    FunctionDef(
        name="simple_return",
        category="Returns",
        description="Simple percentage return: (p_t - p_{t-1}) / p_{t-1}.",
        params=[
            ParamDef(
                name="column", kind=ParamKind.COLUMN, description="Price column name"
            )
        ],
        output_dtype="Float64",
        impl=_impl_simple_return,
    )
)

_register(
    FunctionDef(
        name="log_return",
        category="Returns",
        description="Logarithmic return: log(p_t / p_{t-1}).",
        params=[
            ParamDef(
                name="column", kind=ParamKind.COLUMN, description="Price column name"
            )
        ],
        output_dtype="Float64",
        impl=_impl_log_return,
    )
)

_register(
    FunctionDef(
        name="cumulative_return",
        category="Returns",
        description="Cumulative return: (1 + r).cum_prod() - 1.",
        params=[
            ParamDef(
                name="column", kind=ParamKind.COLUMN, description="Price column name"
            )
        ],
        output_dtype="Float64",
        impl=_impl_cumulative_return,
    )
)

_register(
    FunctionDef(
        name="sma",
        category="Technical",
        description="Simple moving average over a rolling window.",
        params=[
            ParamDef(name="column", kind=ParamKind.COLUMN, description="Column name"),
            ParamDef(
                name="window",
                kind=ParamKind.LITERAL,
                description="Window size (integer)",
            ),
        ],
        output_dtype="Float64",
        impl=_impl_sma,
    )
)

_register(
    FunctionDef(
        name="ema",
        category="Technical",
        description="Exponential moving average (EWM with span=window).",
        params=[
            ParamDef(name="column", kind=ParamKind.COLUMN, description="Column name"),
            ParamDef(
                name="window",
                kind=ParamKind.LITERAL,
                description="Span for EWM (integer)",
            ),
        ],
        output_dtype="Float64",
        impl=_impl_ema,
    )
)

_register(
    FunctionDef(
        name="sharpe_ratio",
        category="Returns",
        description="Rolling Sharpe ratio: (mean(returns) - rf) / std(returns) over a window.",
        params=[
            ParamDef(
                name="column",
                kind=ParamKind.COLUMN,
                description="Returns column name",
            ),
            ParamDef(
                name="window",
                kind=ParamKind.LITERAL,
                description="Rolling window size (integer, >= 2)",
            ),
            ParamDef(
                name="rf",
                kind=ParamKind.LITERAL,
                required=False,
                default=0.0,
                description="Risk-free rate per period (default 0)",
            ),
        ],
        output_dtype="Float64",
        impl=_impl_sharpe_ratio,
    )
)

_register(
    FunctionDef(
        name="sortino_ratio",
        category="Returns",
        description="Rolling Sortino ratio: (mean(returns) - rf) / downside_std(returns) over a window. Downside deviation only considers returns below rf.",
        params=[
            ParamDef(
                name="column",
                kind=ParamKind.COLUMN,
                description="Returns column name",
            ),
            ParamDef(
                name="window",
                kind=ParamKind.LITERAL,
                description="Rolling window size (integer, >= 2)",
            ),
            ParamDef(
                name="rf",
                kind=ParamKind.LITERAL,
                required=False,
                default=0.0,
                description="Risk-free rate per period (default 0)",
            ),
        ],
        output_dtype="Float64",
        impl=_impl_sortino_ratio,
    )
)


# ---------------------------------------------------------------------------
# Function search index (BM25 over function registry)
# ---------------------------------------------------------------------------


class FunctionIndex:
    """FTS5-backed BM25 search index over registered functions."""

    def __init__(self, registry: dict[str, FunctionDef] | None = None) -> None:
        reg = registry or FUNCTION_REGISTRY
        self._functions = list(reg.values())

        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE fn_fts USING fts5(  search_text, tokenize='porter')"
        )
        for i, func in enumerate(self._functions):
            self._conn.execute(
                "INSERT INTO fn_fts(rowid, search_text) VALUES (?, ?)",
                (i, func.search_text),
            )

    def search(self, query: str, top_k: int = 5) -> list[FunctionDef]:
        if not self._functions:
            return []
        fts_query = _expand_query(query)
        if not fts_query:
            return []
        try:
            cursor = self._conn.execute(
                "SELECT rowid FROM fn_fts "
                "WHERE fn_fts MATCH ? "
                "ORDER BY bm25(fn_fts) "
                "LIMIT ?",
                (fts_query, min(top_k, len(self._functions))),
            )
            return [self._functions[row[0]] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []


# ---------------------------------------------------------------------------
# Apply pipeline
# ---------------------------------------------------------------------------


def apply_pipeline(table: Table, steps: list[dict]) -> Table:
    """Apply a sequence of function steps to a Table.

    Each step is a dict with:
        - "function": name of registered function
        - "inputs": dict mapping param names to values (strings=columns, numbers=literals)
        - "output": column name for the result

    Returns the enriched Table.
    """
    if len(steps) > MAX_APPLY_STEPS:
        raise ValueError(
            f"Too many apply steps ({len(steps)}). Maximum is {MAX_APPLY_STEPS}."
        )

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i}: expected a dict, got {type(step).__name__}")

        func_name = step.get("function")
        if not func_name or func_name not in FUNCTION_REGISTRY:
            raise ValueError(
                f"Step {i}: unknown function '{func_name}'. Available: {list(FUNCTION_REGISTRY.keys())}"
            )

        func_def = FUNCTION_REGISTRY[func_name]
        raw_inputs = step.get("inputs", {})
        output_col = step.get("output", func_name)

        if not _OUTPUT_COL_RE.match(output_col):
            raise ValueError(
                f"Step {i}: invalid output column name '{output_col}'. "
                "Must match [a-zA-Z_][a-zA-Z0-9_]{{0,62}}."
            )

        # Resolve inputs
        resolved: dict[str, Any] = {}
        for param in func_def.params:
            if param.name in raw_inputs:
                resolved[param.name] = resolve_input(
                    table, raw_inputs[param.name], param.kind
                )
            elif not param.required and param.default is not None:
                resolved[param.name] = param.default
            elif param.required:
                raise ValueError(
                    f"Step {i} ({func_name}): missing required param '{param.name}'"
                )

        # Execute — impl returns np.ndarray
        result_array = func_def.impl(table, resolved)
        # Convert numpy array to list, replacing NaN with None
        result_list = []
        for v in result_array:
            if isinstance(v, float) and math.isnan(v):
                result_list.append(None)
            else:
                result_list.append(float(v))
        table = table.with_column(output_col, result_list)

    return table
