import math
from typing import cast

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

    def test_bs_gamma_otm(self):
        # OTM call S=80, K=100, T=1, r=0.05, sigma=0.2 → d1=-0.7657, d2=-0.9657.
        # Gamma is always positive but the magnitude moves; verify the formula
        # away from the at-the-money peak.
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_gamma",
                    "inputs": {
                        "S": 80.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                    },
                    "output": "gamma",
                }
            ],
        )
        gamma = result["gamma"][0]
        assert gamma > 0
        assert abs(gamma - 0.018598225671687875) < 1e-6

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

    def test_bs_vega_otm(self):
        # OTM call S=80, K=100, T=1, r=0.05, sigma=0.2.
        # Vega is always positive but smaller off the at-the-money peak;
        # verify the formula at a non-ATM strike.
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_vega",
                    "inputs": {
                        "S": 80.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                    },
                    "output": "vega",
                }
            ],
        )
        vega = result["vega"][0]
        assert vega > 0
        assert abs(vega - 0.23805728859760478) < 1e-6

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

    def test_bs_vanna(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_vanna",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "vanna",
                }
            ],
        )
        vanna = result["vanna"][0]
        # vanna_raw = -phi(d1) * d2 / sigma; per 1% vol → /100
        # phi(0.35) ≈ 0.375240 → vanna_/100 ≈ -0.0028143
        assert vanna < 0
        assert abs(vanna - (-0.002814302601877034)) < 1e-5

    def test_bs_vanna_sign_flip(self):
        # vanna sign tracks -d2.  At ATM (d2 > 0) vanna is negative; at OTM
        # call S=80, K=100 (d2 = -0.9657 < 0) it flips positive.  This
        # catches sign errors the at-the-money case can't.
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_vanna",
                    "inputs": {
                        "S": 80.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                    },
                    "output": "vanna",
                }
            ],
        )
        vanna = result["vanna"][0]
        assert vanna > 0
        assert abs(vanna - 0.014368509417491595) < 1e-5

    def test_bs_volga(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_volga",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "volga",
                }
            ],
        )
        volga = result["volga"][0]
        # volga_raw = vega_raw * d1 * d2 / sigma; per (1% vol)^2 → /10000
        # ATM-ish but d1·d2 > 0 so volga > 0; ≈ 0.0009850
        assert volga > 0
        assert abs(volga - 0.0009850059106569621) < 1e-6

    def test_bs_charm(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_charm",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "charm",
                }
            ],
        )
        charm = result["charm"][0]
        # ATM call delta drifts down with time → charm < 0; ≈ -0.0001799/day
        assert charm < 0
        assert abs(charm - (-0.0001799097553711346)) < 1e-8

    def test_bs_charm_otm(self):
        # OTM call S=80, K=100, T=1, r=0.05, sigma=0.2.
        # Charm magnitude is larger off-the-money; verify formula away from ATM.
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_charm",
                    "inputs": {
                        "S": 80.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                    },
                    "output": "charm",
                }
            ],
        )
        charm = result["charm"][0]
        assert charm < 0
        assert abs(charm - (-0.0005974739640045683)) < 1e-8

    def test_bs_veta(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_veta",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "veta",
                }
            ],
        )
        veta = result["veta"][0]
        # Vega bleeds with time for ATM → veta < 0; ≈ -0.0004511 (Δbs_vega/day)
        assert veta < 0
        assert abs(veta - (-0.0004510594581090589)) < 1e-8

    def test_bs_veta_otm(self):
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_veta",
                    "inputs": {
                        "S": 80.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                    },
                    "output": "veta",
                }
            ],
        )
        veta = result["veta"][0]
        assert veta < 0
        assert abs(veta - (-0.000692103013452991)) < 1e-8

    def test_bs_color(self, bs_df):
        result = apply_pipeline(
            bs_df,
            [
                {
                    "function": "bs_color",
                    "inputs": {
                        "S": "S",
                        "K": "K",
                        "T": "T",
                        "r": "r",
                        "sigma": "sigma",
                    },
                    "output": "color",
                }
            ],
        )
        color = result["color"][0]
        # ATM gamma builds with time → color > 0; ≈ +2.885e-5 (Δbs_gamma/day)
        assert color > 0
        assert abs(color - 2.884981434344266e-05) < 1e-9

    def test_bs_color_sign_flip(self):
        # color sign tracks ATM-ness: ATM gamma builds (color > 0) but OTM
        # gamma decays (color < 0).  Catches sign errors the ATM case can't.
        df = _table(dummy=[1.0])
        result = apply_pipeline(
            df,
            [
                {
                    "function": "bs_color",
                    "inputs": {
                        "S": 80.0,
                        "K": 100.0,
                        "T": 1.0,
                        "r": 0.05,
                        "sigma": 0.2,
                    },
                    "output": "color",
                }
            ],
        )
        color = result["color"][0]
        assert color < 0
        assert abs(color - (-3.116504989883754e-06)) < 1e-9

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

    def test_long_series_matches_per_window_mean(self):
        rng = np.random.default_rng(0)
        arr = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 5000))
        arr[[50, 51, 3000]] = np.nan
        window = 200
        result = _rolling_mean(arr, window)
        windows = np.lib.stride_tricks.sliding_window_view(arr, window)
        expected = np.concatenate([np.full(window - 1, np.nan), windows.mean(axis=1)])
        np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


class TestRollingStd:
    def test_basic(self):
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _rolling_std(arr, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Window [2, 4, 4]: std(ddof=1)
        expected = np.std([2.0, 4.0, 4.0], ddof=1)
        assert abs(result[2] - expected) < 1e-10

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_window_1_returns_nan(self):
        """std with ddof=1 and window=1 is undefined (0/0)."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _rolling_std(arr, 1)
        assert np.all(np.isnan(result))

    def test_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0, 4.0])
        result = _rolling_std(arr, 2)
        assert np.isnan(result[1])  # window [1, NaN]
        assert np.isnan(result[2])  # window [NaN, 3]
        assert not np.isnan(result[3])  # window [3, 4]

    def test_long_series_matches_per_window_std(self):
        rng = np.random.default_rng(1)
        arr = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 5000))
        arr[[50, 3000]] = np.nan
        window = 30
        result = _rolling_std(arr, window)
        windows = np.lib.stride_tricks.sliding_window_view(arr, window)
        expected = np.concatenate(
            [np.full(window - 1, np.nan), windows.std(axis=1, ddof=1)]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-9, equal_nan=True)


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


def _wilder(values: np.ndarray, period: int, start: int) -> np.ndarray:
    """Wilder smoothing seeded with the mean of values[start : start + period]."""
    out = np.full(len(values), np.nan)
    seed = start + period - 1
    if seed >= len(values):
        return out
    out[seed] = values[start : seed + 1].mean()
    for i in range(seed + 1, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def _sma_seeded_ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    first = int(np.argmax(~np.isnan(values)))
    seed = first + period - 1
    alpha = 2.0 / (period + 1.0)
    out[seed] = values[first : seed + 1].mean()
    for i in range(seed + 1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _apply_one(df: Table, function: str, inputs: dict) -> list:
    result = apply_pipeline(
        df, [{"function": function, "inputs": inputs, "output": "out"}]
    )
    return result["out"]


def _as_array(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values])


class TestMoreTechnicals:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 300))
    high = close + rng.uniform(0.1, 2.0, 300)
    low = close - rng.uniform(0.1, 2.0, 300)

    def _df(self) -> Table:
        return _table(
            close=self.close.tolist(), high=self.high.tolist(), low=self.low.tolist()
        )

    def test_rsi_matches_wilder_reference(self):
        out = _as_array(
            _apply_one(self._df(), "rsi", {"column": "close", "window": 14})
        )
        diff = np.concatenate([[np.nan], np.diff(self.close)])
        gain = _wilder(np.where(diff > 0, diff, 0.0), 14, 1)
        loss = _wilder(np.where(diff < 0, -diff, 0.0), 14, 1)
        expected = 100.0 * gain / (gain + loss)
        assert np.isnan(out[:14]).all()
        np.testing.assert_allclose(out, expected, rtol=1e-10, equal_nan=True)

    def test_atr_matches_wilder_reference(self):
        out = _as_array(
            _apply_one(
                self._df(),
                "atr",
                {"high": "high", "low": "low", "close": "close", "window": 14},
            )
        )
        prev = np.concatenate([[np.nan], self.close[:-1]])
        tr = np.maximum.reduce(
            [self.high - self.low, np.abs(self.high - prev), np.abs(self.low - prev)]
        )
        expected = _wilder(tr, 14, 1)
        assert np.isnan(out[:14]).all()
        np.testing.assert_allclose(out, expected, rtol=1e-10, equal_nan=True)

    def test_macd_lines(self):
        df = self._df()
        line = _as_array(_apply_one(df, "macd", {"column": "close"}))
        signal = _as_array(
            _apply_one(df, "macd", {"column": "close", "line": "signal"})
        )
        hist = _as_array(
            _apply_one(df, "macd", {"column": "close", "line": "histogram"})
        )
        expected_line = _sma_seeded_ema(self.close, 12) - _sma_seeded_ema(
            self.close, 26
        )
        expected_line[:25] = np.nan
        np.testing.assert_allclose(line, expected_line, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            signal, _sma_seeded_ema(expected_line, 9), rtol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(hist, line - signal, atol=1e-12, equal_nan=True)

    def test_macd_invalid_line(self):
        with pytest.raises(ValueError, match="Invalid MACD line"):
            _apply_one(self._df(), "macd", {"column": "close", "line": "foo"})

    def test_bbands(self):
        df = self._df()
        bands = {
            band: _as_array(
                _apply_one(
                    df, "bbands", {"column": "close", "window": 20, "band": band}
                )
            )
            for band in ("lower", "middle", "upper")
        }
        windows = np.lib.stride_tricks.sliding_window_view(self.close, 20)
        pad = np.full(19, np.nan)
        mid = np.concatenate([pad, windows.mean(axis=1)])
        std = np.concatenate([pad, windows.std(axis=1)])
        np.testing.assert_allclose(bands["middle"], mid, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            bands["upper"], mid + 2 * std, rtol=1e-9, equal_nan=True
        )
        np.testing.assert_allclose(
            bands["lower"], mid - 2 * std, rtol=1e-9, equal_nan=True
        )

    def test_bbands_invalid(self):
        with pytest.raises(ValueError, match="Invalid Bollinger band"):
            _apply_one(
                self._df(), "bbands", {"column": "close", "window": 20, "band": "x"}
            )
        with pytest.raises(ValueError, match="window must be >= 2"):
            _apply_one(
                self._df(), "bbands", {"column": "close", "window": 1, "band": "upper"}
            )

    @pytest.mark.parametrize("function", ["rsi", "atr"])
    def test_invalid_window(self, function):
        if function == "rsi":
            inputs = {"column": "close", "window": 0}
        else:
            inputs = {"high": "high", "low": "low", "close": "close", "window": 0}
        with pytest.raises(ValueError, match="window must be >= 1"):
            _apply_one(self._df(), function, inputs)


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
            "bs_vanna",
            "bs_volga",
            "bs_charm",
            "bs_veta",
            "bs_color",
            "simple_return",
            "log_return",
            "cumulative_return",
            "sma",
            "ema",
            "rsi",
            "macd",
            "bbands",
            "atr",
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
        bad_steps = cast(list[dict], ["not_a_dict"])
        with pytest.raises(ValueError, match="expected a dict"):
            apply_pipeline(df, bad_steps)


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
