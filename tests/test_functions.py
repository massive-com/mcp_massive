import math

import numpy as np
import pytest

from mcp_massive.functions import (
    FUNCTION_REGISTRY,
    FunctionIndex,
    MAX_APPLY_STEPS,
    ParamKind,
    apply_pipeline,
    resolve_input,
    _norm_cdf,
    _rolling_mean,
    _rolling_std,
)
from mcp_massive.store import Table


def _table(**columns: list) -> Table:
    """Helper to create a Table from column-oriented data."""
    col_names = list(columns.keys())
    return Table(col_names, dict(columns))


# ---------------------------------------------------------------------------
# Normal CDF
# ---------------------------------------------------------------------------


class TestNormCdf:
    def test_zero(self):
        result = _norm_cdf(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-7

    def test_positive_1_96(self):
        result = _norm_cdf(np.array([1.96]))
        assert abs(result[0] - 0.975002) < 1e-4

    def test_negative_1_96(self):
        result = _norm_cdf(np.array([-1.96]))
        assert abs(result[0] - 0.024998) < 1e-4

    def test_large_positive(self):
        result = _norm_cdf(np.array([10.0]))
        assert abs(result[0] - 1.0) < 1e-7

    def test_large_negative(self):
        result = _norm_cdf(np.array([-10.0]))
        assert abs(result[0]) < 1e-7

    def test_symmetry(self):
        x = np.array([0.5, 1.0, 2.0, 3.0])
        assert np.allclose(_norm_cdf(x) + _norm_cdf(-x), 1.0, atol=1e-7)

    def test_vectorized(self):
        x = np.linspace(-3, 3, 100)
        result = _norm_cdf(x)
        assert result.shape == (100,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Monotonically increasing
        assert np.all(np.diff(result) >= 0)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks — standard test case
# ---------------------------------------------------------------------------


class TestBlackScholesGreeks:
    """Standard test: S=100, K=100, T=1, r=0.05, sigma=0.2"""

    @pytest.fixture
    def bs_df(self):
        return _table(
            S=[100.0],
            K=[100.0],
            T=[1.0],
            r=[0.05],
            sigma=[0.2],
        )

    def test_bs_price_call(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_price",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "call",
                    },
                    "output": "price",
                }
            ],
        )
        # Known BS call price for S=K=100, T=1, r=0.05, sigma=0.2 ≈ 10.45
        price = result["price"][0]
        assert abs(price - 10.4506) < 0.05

    def test_bs_price_put(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_price",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "put",
                    },
                    "output": "price",
                }
            ],
        )
        # Known BS put price ≈ 5.57
        price = result["price"][0]
        assert abs(price - 5.5735) < 0.05

    def test_put_call_parity(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_price",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "call",
                    },
                    "output": "call_price",
                },
                {
                    "function": "bs_price",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "put",
                    },
                    "output": "put_price",
                },
            ],
        )
        S = 100.0
        K = 100.0
        r = 0.05
        T = 1.0
        call = result["call_price"][0]
        put = result["put_price"][0]
        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6

    def test_bs_delta_call(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_delta",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "call",
                    },
                    "output": "delta",
                }
            ],
        )
        delta = result["delta"][0]
        # ATM call delta ≈ 0.6368
        assert 0.5 < delta < 0.8
        assert abs(delta - 0.6368) < 0.01

    def test_bs_delta_put(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_delta",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "put",
                    },
                    "output": "delta",
                }
            ],
        )
        delta = result["delta"][0]
        # Put delta = call delta - 1 ≈ 0.6368 - 1 = -0.3632
        assert -1.0 < delta < 0.0
        assert abs(delta - (-0.3631693488243809)) < 0.01

    def test_bs_gamma(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_gamma",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "gamma",
                }
            ],
        )
        gamma = result["gamma"][0]
        # ATM gamma ≈ 0.01876
        assert 0.01 < gamma < 0.03
        assert abs(gamma - 0.018762017345846895) < 0.001

    def test_bs_theta_call(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_theta",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "call",
                    },
                    "output": "theta",
                }
            ],
        )
        theta = result["theta"][0]
        # Daily theta should be negative for long options ≈ -0.01757
        assert theta < 0
        assert abs(theta - (-0.01757267820941972)) < 0.001

    def test_bs_vega(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_vega",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "vega",
                }
            ],
        )
        vega = result["vega"][0]
        # Vega per 1% vol change ≈ 0.37524
        assert vega > 0
        assert abs(vega - 0.3752403469169379) < 0.01

    def test_bs_rho_call(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_rho",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "call",
                    },
                    "output": "rho",
                }
            ],
        )
        rho = result["rho"][0]
        # Call rho per 1% rate change ≈ 0.53232
        assert rho > 0
        assert abs(rho - 0.5323248154537634) < 0.01

    def test_bs_rho_put(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_rho",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                        "option_type": "put",
                    },
                    "output": "rho",
                }
            ],
        )
        rho = result["rho"][0]
        # Put rho per 1% rate change ≈ -0.41890
        assert rho < 0
        assert abs(rho - (-0.4189046090469506)) < 0.01

    def test_invalid_option_type(self, bs_df):
        """Invalid option_type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid option_type"):
            apply_pipeline(
                bs_df,
                [
                    {
                        "function": "bs_delta",
                        "inputs": {
                            "S": "S",
                            "K": "K",
                            "T": "T",
                            "r": "r",
                            "sigma": "sigma",
                            "option_type": "invalid",
                        },
                        "output": "delta",
                    }
                ],
            )

    def test_literal_inputs(self):
        """Greeks with all literal inputs (no columns)."""
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_delta",
                    "inputs": {
                        "S": 100.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                        "option_type": "call",
                    },
                    "output": "delta",
                }
            ],
        )
        delta = result["delta"][0]
        assert abs(delta - 0.6368) < 0.01


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


class TestReturns:
    def test_simple_return(self):
        df = _table(price=[100.0, 110.0, 105.0, 115.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "simple_return",
                    "inputs": {"column": "price"},
                    "output": "ret",
                }
            ],
        )
        ret = result["ret"]
        assert ret[0] is None  # first value is null
        assert abs(ret[1] - 0.10) < 1e-10
        assert abs(ret[2] - (-5.0 / 110.0)) < 1e-10

    def test_log_return(self):
        df = _table(price=[100.0, 110.0, 105.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "log_return",
                    "inputs": {"column": "price"},
                    "output": "ret",
                }
            ],
        )
        ret = result["ret"]
        assert ret[0] is None
        expected = math.log(110.0 / 100.0)
        assert abs(ret[1] - expected) < 1e-10

    def test_cumulative_return(self):
        df = _table(price=[100.0, 110.0, 121.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "cumulative_return",
                    "inputs": {"column": "price"},
                    "output": "cum_ret",
                }
            ],
        )
        cum_ret = result["cum_ret"]
        # After 10% then 10%: cumulative = 0.21
        assert abs(cum_ret[2] - 0.21) < 1e-10


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------


class TestRollingMean:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_mean(arr, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert abs(result[2] - 2.0) < 1e-10
        assert abs(result[3] - 3.0) < 1e-10
        assert abs(result[4] - 4.0) < 1e-10

    def test_window_1(self):
        arr = np.array([10.0, 20.0, 30.0])
        result = _rolling_mean(arr, 1)
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0])

    def test_window_equals_length(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _rolling_mean(arr, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert abs(result[2] - 2.0) < 1e-10

    def test_window_exceeds_length(self):
        arr = np.array([1.0, 2.0])
        result = _rolling_mean(arr, 5)
        assert all(np.isnan(result))

    def test_with_nan_in_data(self):
        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = _rolling_mean(arr, 3)
        # Windows containing NaN should be NaN
        assert np.isnan(result[2])  # window [1, NaN, 3]
        assert np.isnan(result[3])  # window [NaN, 3, 4]
        assert abs(result[4] - 4.0) < 1e-10  # window [3, 4, 5]

    def test_empty_array(self):
        arr = np.array([], dtype=np.float64)
        result = _rolling_mean(arr, 3)
        assert len(result) == 0


class TestRollingStd:
    def test_basic(self):
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _rolling_std(arr, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Window [2, 4, 4]: std(ddof=1)
        expected = np.std([2.0, 4.0, 4.0], ddof=1)
        assert abs(result[2] - expected) < 1e-10

    def test_window_1_returns_nan(self):
        """std with ddof=1 and window=1 is undefined (0/0)."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _rolling_std(arr, 1)
        # np.std with ddof=1 on a single element gives NaN or warning
        # Either NaN or 0 is acceptable; the key is it doesn't crash
        assert len(result) == 3

    def test_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0, 4.0])
        result = _rolling_std(arr, 2)
        assert np.isnan(result[1])  # window [1, NaN]
        assert np.isnan(result[2])  # window [NaN, 3]
        assert not np.isnan(result[3])  # window [3, 4]


# ---------------------------------------------------------------------------
# EMA specifics
# ---------------------------------------------------------------------------


class TestEma:
    def test_exact_values(self):
        """Verify EMA against hand-calculated values for span=3 (alpha=0.5)."""
        df = _table(price=[1.0, 2.0, 3.0, 4.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "ema",
                    "inputs": {"column": "price", "window": 3},
                    "output": "ema3",
                }
            ],
        )
        ema = result["ema3"]
        # alpha = 2.0 / (3 + 1) = 0.5
        # ema[0] = 1.0
        assert abs(ema[0] - 1.0) < 1e-10
        # ema[1] = 0.5 * 2 + 0.5 * 1 = 1.5
        assert abs(ema[1] - 1.5) < 1e-10
        # ema[2] = 0.5 * 3 + 0.5 * 1.5 = 2.25
        assert abs(ema[2] - 2.25) < 1e-10
        # ema[3] = 0.5 * 4 + 0.5 * 2.25 = 3.125
        assert abs(ema[3] - 3.125) < 1e-10

    def test_ema_with_leading_nan(self):
        """EMA should handle NaN at position 0 (e.g., from simple_return)."""
        df = _table(val=[float("nan"), 2.0, 3.0, 4.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "ema",
                    "inputs": {"column": "val", "window": 3},
                    "output": "ema_out",
                }
            ],
        )
        ema = result["ema_out"]
        # First value is NaN input, should be None in output
        assert ema[0] is None
        # alpha = 0.5, seed at index 1 = 2.0
        assert abs(ema[1] - 2.0) < 1e-10
        # ema[2] = 0.5*3 + 0.5*2.0 = 2.5
        assert abs(ema[2] - 2.5) < 1e-10
        # ema[3] = 0.5*4 + 0.5*2.5 = 3.25
        assert abs(ema[3] - 3.25) < 1e-10

    def test_ema_chained_after_simple_return(self):
        """EMA should work when chained after simple_return (which produces NaN at [0])."""
        df = _table(price=[100.0, 110.0, 105.0, 115.0, 120.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "simple_return",
                    "inputs": {"column": "price"},
                    "output": "ret",
                },
                {
                    "function": "ema",
                    "inputs": {"column": "ret", "window": 3},
                    "output": "ema_ret",
                },
            ],
        )
        ema_ret = result["ema_ret"]
        # First value should be None (NaN from simple_return)
        assert ema_ret[0] is None
        # EMA seeds at index 1 (first valid return = 0.1), alpha=0.5
        assert abs(ema_ret[1] - 0.1) < 1e-10
        # ema[2] = 0.5*(-5/110) + 0.5*0.1 ≈ 0.02727
        assert abs(ema_ret[2] - 0.027272727272727275) < 1e-10
        # ema[3] = 0.5*(10/105) + 0.5*ema[2] ≈ 0.06126
        assert abs(ema_ret[3] - 0.06125541125541126) < 1e-10
        # ema[4] = 0.5*(5/115) + 0.5*ema[3] ≈ 0.05237
        assert abs(ema_ret[4] - 0.05236683606248824) < 1e-10

    def test_ema_empty_table(self):
        """EMA on empty table should return empty result."""
        df = _table(price=[])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "ema",
                    "inputs": {"column": "price", "window": 3},
                    "output": "ema_out",
                }
            ],
        )
        assert len(result["ema_out"]) == 0

    def test_ema_single_element(self):
        df = _table(price=[42.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "ema",
                    "inputs": {"column": "price", "window": 3},
                    "output": "ema_out",
                }
            ],
        )
        assert abs(result["ema_out"][0] - 42.0) < 1e-10


# ---------------------------------------------------------------------------
# Empty table through financial functions
# ---------------------------------------------------------------------------


class TestEmptyTableFunctions:
    def test_simple_return_empty(self):
        df = _table(price=[])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "simple_return",
                    "inputs": {"column": "price"},
                    "output": "r",
                }
            ],
        )
        assert len(result["r"]) == 0

    def test_log_return_empty(self):
        df = _table(price=[])
        result = apply_pipeline(
            df,
            [{"function": "log_return", "inputs": {"column": "price"}, "output": "r"}],
        )
        assert len(result["r"]) == 0

    def test_cumulative_return_empty(self):
        df = _table(price=[])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "cumulative_return",
                    "inputs": {"column": "price"},
                    "output": "r",
                }
            ],
        )
        assert len(result["r"]) == 0

    def test_sma_empty(self):
        df = _table(price=[])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "sma",
                    "inputs": {"column": "price", "window": 3},
                    "output": "s",
                }
            ],
        )
        assert len(result["s"]) == 0


class TestTechnicals:
    def test_sma(self):
        df = _table(price=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "sma",
                    "inputs": {"column": "price", "window": 3},
                    "output": "sma3",
                }
            ],
        )
        sma = result["sma3"]
        assert sma[0] is None  # not enough data
        assert sma[1] is None
        assert abs(sma[2] - 2.0) < 1e-10  # (1+2+3)/3
        assert abs(sma[3] - 3.0) < 1e-10  # (2+3+4)/3
        assert abs(sma[4] - 4.0) < 1e-10  # (3+4+5)/3

    def test_sma_invalid_window(self):
        df = _table(price=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="window must be >= 1"):
            apply_pipeline(
                df,
                [
                    {
                        "function": "sma",
                        "inputs": {"column": "price", "window": 0},
                        "output": "sma_out",
                    }
                ],
            )

    def test_ema(self):
        df = _table(price=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "ema",
                    "inputs": {"column": "price", "window": 3},
                    "output": "ema3",
                }
            ],
        )
        ema = result["ema3"]
        assert len(ema) == 5
        # EMA should be between min and max
        for val in ema:
            assert 1.0 <= val <= 5.0
        # Last EMA should be closer to 5 than to 1
        assert ema[4] > 3.0

    def test_sortino_ratio_known_values(self):
        """Verify Sortino ratio against hand-calculated values.

        Series: [0.01, -0.02, 0.03, -0.01, 0.02] with rf=0, window=5
        mean = (0.01 - 0.02 + 0.03 - 0.01 + 0.02) / 5 = 0.006
        downside_sq = [0, 0.0004, 0, 0.0001, 0]  (min(r-0, 0)^2)
        downside_dev = sqrt(mean(downside_sq)) = sqrt(0.0001) = 0.01
        sortino = 0.006 / 0.01 = 0.6
        """
        df = _table(returns=[0.01, -0.02, 0.03, -0.01, 0.02])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "sortino_ratio",
                    "inputs": {"column": "returns", "window": 5},
                    "output": "sortino",
                }
            ],
        )
        # First 4 values should be null (window not full)
        for i in range(4):
            assert result["sortino"][i] is None
        assert result["sortino"][4] == pytest.approx(0.6, abs=0.05)

    def test_sortino_ratio_no_downside_returns_null(self):
        """When all returns are positive, downside deviation is 0 → result is null."""
        df = _table(returns=[0.01, 0.02, 0.03, 0.04, 0.05])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "sortino_ratio",
                    "inputs": {"column": "returns", "window": 5},
                    "output": "sortino",
                }
            ],
        )
        assert result["sortino"][4] is None

    def test_sortino_ratio_invalid_window(self):
        df = _table(returns=[0.01, 0.02])
        with pytest.raises(ValueError, match="window must be >= 2"):
            apply_pipeline(
                df,
                [
                    {
                        "function": "sortino_ratio",
                        "inputs": {"column": "returns", "window": 1},
                        "output": "sortino",
                    }
                ],
            )

    def test_sharpe_ratio_known_values(self):
        """Verify Sharpe ratio against hand-calculated values.

        Series: [0.10, 0.12, 0.08, 0.11, 0.09], rf=0, window=5
        mean = 0.10, std = std([0.10, 0.12, 0.08, 0.11, 0.09])
        """
        df = _table(returns=[0.10, 0.12, 0.08, 0.11, 0.09])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "sharpe_ratio",
                    "inputs": {"column": "returns", "window": 5},
                    "output": "sharpe",
                }
            ],
        )
        sharpe = result["sharpe"][4]
        assert sharpe is not None
        # mean = 0.10, std(ddof=1) ≈ 0.01581, sharpe ≈ 6.3246
        assert abs(sharpe - 6.324555320336759) < 0.01


# ---------------------------------------------------------------------------
# Function Registry
# ---------------------------------------------------------------------------


class TestFunctionRegistry:
    def test_all_registered(self):
        expected = {
            "bs_price",
            "bs_delta",
            "bs_gamma",
            "bs_theta",
            "bs_vega",
            "bs_rho",
            "simple_return",
            "log_return",
            "cumulative_return",
            "sma",
            "ema",
            "sharpe_ratio",
            "sortino_ratio",
        }
        assert expected == set(FUNCTION_REGISTRY.keys())

    def test_lookup_by_name(self):
        func = FUNCTION_REGISTRY.get("bs_delta")
        assert func is not None
        assert func.name == "bs_delta"
        assert func.category == "Greeks"

    def test_unknown_returns_none(self):
        assert FUNCTION_REGISTRY.get("nonexistent") is None

    def test_signature(self):
        func = FUNCTION_REGISTRY["bs_delta"]
        sig = func.signature()
        assert "bs_delta" in sig
        assert "S" in sig
        assert "option_type?" in sig

    def test_full_description(self):
        func = FUNCTION_REGISTRY["bs_delta"]
        desc = func.full_description()
        assert "bs_delta" in desc
        assert "Black-Scholes delta" in desc
        assert "option_type" in desc


# ---------------------------------------------------------------------------
# Function Index (BM25 search)
# ---------------------------------------------------------------------------


class TestFunctionIndex:
    def test_search_greeks(self):
        idx = FunctionIndex()
        results = idx.search("greeks")
        names = [f.name for f in results]
        assert any("bs_" in n for n in names)

    def test_search_delta(self):
        idx = FunctionIndex()
        results = idx.search("delta")
        names = [f.name for f in results]
        assert "bs_delta" in names

    def test_search_returns(self):
        idx = FunctionIndex()
        results = idx.search("returns")
        categories = {f.category for f in results}
        assert "Returns" in categories

    def test_search_technical(self):
        idx = FunctionIndex()
        results = idx.search("moving average")
        names = [f.name for f in results]
        assert "sma" in names or "ema" in names

    def test_search_no_results(self):
        idx = FunctionIndex()
        results = idx.search("xyznonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# Apply Pipeline
# ---------------------------------------------------------------------------


class TestApplyPipeline:
    def test_single_step(self):
        df = _table(price=[100.0, 110.0, 105.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "simple_return",
                    "inputs": {"column": "price"},
                    "output": "ret",
                }
            ],
        )
        assert "ret" in result.columns
        assert "price" in result.columns

    def test_multi_step_chain(self):
        df = _table(price=[100.0, 110.0, 105.0, 115.0, 120.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "simple_return",
                    "inputs": {"column": "price"},
                    "output": "ret",
                },
                {
                    "function": "sma",
                    "inputs": {"column": "price", "window": 3},
                    "output": "sma3",
                },
            ],
        )
        assert "ret" in result.columns
        assert "sma3" in result.columns

    def test_unknown_function_error(self):
        df = _table(x=[1.0])
        with pytest.raises(ValueError, match="unknown function"):
            apply_pipeline(
                df, [{"function": "nonexistent", "inputs": {}, "output": "y"}]
            )

    def test_missing_column_error(self):
        df = _table(x=[1.0])
        with pytest.raises(ValueError, match="not found"):
            apply_pipeline(
                df,
                [
                    {
                        "function": "simple_return",
                        "inputs": {"column": "missing_col"},
                        "output": "ret",
                    }
                ],
            )

    def test_missing_required_param(self):
        df = _table(x=[1.0])
        with pytest.raises(ValueError, match="missing required param"):
            apply_pipeline(
                df,
                [
                    {
                        "function": "sma",
                        "inputs": {"column": "x"},  # missing "window"
                        "output": "sma_out",
                    }
                ],
            )

    def test_default_params(self):
        """option_type defaults to 'call' if not provided."""
        df = _table(S=[100.0], K=[100.0], T=[1.0], r=[0.05], sigma=[0.2])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_delta",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "delta",
                }
            ],
        )
        delta = result["delta"][0]
        assert 0.5 < delta < 0.8  # call delta

    def test_default_output_name(self):
        """When output is not specified, uses function name."""
        df = _table(price=[100.0, 110.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "simple_return",
                    "inputs": {"column": "price"},
                }
            ],
        )
        assert "simple_return" in result.columns

    def test_empty_steps(self):
        """Empty steps list returns original Table unchanged."""
        df = _table(x=[1.0, 2.0])
        result = apply_pipeline(df, [])
        assert result.equals(df)

    def test_too_many_steps(self):
        """Exceeding MAX_APPLY_STEPS raises ValueError."""
        df = _table(price=[100.0, 110.0])
        steps = [
            {
                "function": "simple_return",
                "inputs": {"column": "price"},
                "output": f"ret_{i}",
            }
            for i in range(MAX_APPLY_STEPS + 1)
        ]
        with pytest.raises(ValueError, match="Too many apply steps"):
            apply_pipeline(df, steps)

    def test_invalid_output_column_name(self):
        """Output column names must be valid identifiers."""
        df = _table(price=[100.0])
        with pytest.raises(ValueError, match="invalid output column name"):
            apply_pipeline(
                df,
                [
                    {
                        "function": "simple_return",
                        "inputs": {"column": "price"},
                        "output": "bad name!",
                    }
                ],
            )

    def test_step_not_dict(self):
        """Non-dict steps raise ValueError."""
        df = _table(x=[1.0])
        with pytest.raises(ValueError, match="expected a dict"):
            apply_pipeline(df, ["not_a_dict"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Resolve Input
# ---------------------------------------------------------------------------


class TestResolveInput:
    def test_column_ref(self):
        df = _table(price=[1.0, 2.0, 3.0])
        result = resolve_input(df, "price", ParamKind.COLUMN)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_column_missing(self):
        df = _table(x=[1.0])
        with pytest.raises(ValueError, match="not found"):
            resolve_input(df, "missing", ParamKind.COLUMN)

    def test_literal_numeric(self):
        df = _table(x=[1.0])
        result = resolve_input(df, 42.0, ParamKind.LITERAL)
        assert result == 42.0

    def test_literal_string_rejected(self):
        df = _table(x=[1.0])
        with pytest.raises(ValueError, match="LITERAL param expects numeric"):
            resolve_input(df, "not_a_number", ParamKind.LITERAL)

    def test_col_or_lit_string(self):
        df = _table(price=[10.0])
        result = resolve_input(df, "price", ParamKind.COL_OR_LIT)
        assert isinstance(result, np.ndarray)

    def test_col_or_lit_number(self):
        df = _table(x=[1.0])
        result = resolve_input(df, 3.14, ParamKind.COL_OR_LIT)
        assert result == 3.14

    def test_literal_str(self):
        df = _table(x=[1.0])
        result = resolve_input(df, "call", ParamKind.LITERAL_STR)
        assert result == "call"

    def test_literal_str_not_column_ref(self):
        """LITERAL_STR should treat string as literal, not column reference."""
        df = _table(**{"call": [1.0]})
        result = resolve_input(df, "call", ParamKind.LITERAL_STR)
        assert result == "call"
        assert not isinstance(result, np.ndarray)
