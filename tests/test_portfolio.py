import numpy as np
import pytest

from fxlab.portfolio import simulate, summarize, target_weights
from fxlab.portfolio.engine import PortfolioResult
from fxlab.portfolio.signals import momentum_sign, realized_vol
from fxlab.portfolio.stats import max_drawdown


def const_rebalance(n):
    mask = np.ones(n, dtype=bool)
    return mask


# --- the shift: weight at t earns return at t+1 -----------------------------


def test_weight_earns_the_next_days_return_not_the_current():
    # One asset, weight 1 set on day 0, then held.
    returns = np.array([[0.0], [0.10], [0.20]])  # r[t] is return over [t-1, t]
    weights = np.array([[1.0], [1.0], [1.0]])
    cost = np.zeros((3, 1))
    mask = np.array([True, False, False])

    result = simulate(returns, weights, cost, mask)

    # weight_0 earns returns[1]=0.10; weight_1 earns returns[2]=0.20.
    assert result.daily_returns[0] == pytest.approx(0.10)
    assert result.daily_returns[1] == pytest.approx(0.20)
    assert result.equity[-1] == pytest.approx(1.10 * 1.20)


def test_a_return_before_the_first_weight_is_never_captured():
    """The lookahead guard: returns[0] precedes any decision, so it must vanish."""
    returns = np.array([[9.99], [0.0], [0.0]])  # absurd pre-history move
    weights = np.array([[1.0], [1.0], [1.0]])
    cost = np.zeros((3, 1))

    result = simulate(returns, weights, cost, const_rebalance(3))

    assert np.allclose(result.daily_returns, 0.0)
    assert result.equity[-1] == pytest.approx(1.0)


def test_flipping_the_sign_flips_the_pnl():
    returns = np.array([[0.0], [0.05], [-0.03]])
    cost = np.zeros((3, 1))
    mask = const_rebalance(3)

    long = simulate(returns, np.ones((3, 1)), cost, mask)
    short = simulate(returns, -np.ones((3, 1)), cost, mask)

    assert long.daily_returns == pytest.approx(-short.daily_returns)


# --- costs and caps ---------------------------------------------------------


def test_turnover_cost_is_charged_on_the_weight_change():
    returns = np.array([[0.0], [0.0]])
    weights = np.array([[1.0], [1.0]])
    cost = np.full((2, 1), 0.001)

    result = simulate(returns, weights, cost, np.array([True, False]))

    # Going 0 -> 1 is one unit of turnover at 0.001.
    assert result.turnover[0] == pytest.approx(1.0)
    assert result.daily_returns[0] == pytest.approx(-0.001)


def test_gross_exposure_is_capped():
    # Two assets each demanding weight 2.0 -> gross 4.0, capped to 3.0.
    returns = np.zeros((2, 2))
    weights = np.array([[2.0, 2.0], [2.0, 2.0]])
    cost = np.zeros((2, 2))

    result = simulate(returns, weights, cost, const_rebalance(2), gross_cap=3.0)

    assert result.gross_exposure[0] == pytest.approx(3.0)


def test_nan_weight_is_treated_as_flat():
    returns = np.array([[0.0, 0.0], [0.5, 0.5]])
    weights = np.array([[np.nan, 1.0], [np.nan, 1.0]])
    cost = np.zeros((2, 2))

    result = simulate(returns, weights, cost, const_rebalance(2))

    # Only the second asset is positioned.
    assert result.daily_returns[0] == pytest.approx(0.5)


# --- signals ----------------------------------------------------------------


def test_momentum_sign_is_nan_during_warmup_then_directional():
    closes = np.concatenate([np.linspace(1.0, 2.0, 300)])
    sign = momentum_sign(closes, lookback=252)

    assert np.isnan(sign[:252]).all()
    assert (sign[252:] == 1.0).all()  # steadily rising -> long


def test_momentum_sign_reads_the_past_only():
    """Signal at t must not change when future prices change."""
    closes = np.linspace(1.0, 2.0, 400)
    sign_full = momentum_sign(closes, lookback=252)

    tampered = closes.copy()
    tampered[300:] = 0.001  # wreck the future
    sign_tampered = momentum_sign(tampered, lookback=252)

    assert np.array_equal(
        sign_full[252:300], sign_tampered[252:300], equal_nan=True
    )


def test_realized_vol_positive_after_warmup():
    rng = np.random.default_rng(0)
    closes = 1.1 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    vol = realized_vol(closes, lookback=60)

    assert np.isnan(vol[:60]).all()
    assert (vol[60:] > 0).all()


def test_target_weights_scale_inversely_with_vol():
    rng = np.random.default_rng(1)
    calm = 1.1 * np.cumprod(1 + rng.normal(0, 0.004, 400))
    wild = 1.1 * np.cumprod(1 + rng.normal(0, 0.02, 400))

    w_calm = np.nanmean(np.abs(target_weights(calm, n_assets=1)[300:]))
    w_wild = np.nanmean(np.abs(target_weights(wild, n_assets=1)[300:]))

    assert w_calm > w_wild  # lower vol -> larger position for equal risk


# --- stats ------------------------------------------------------------------


def test_max_drawdown_matches_a_known_curve():
    equity = np.array([1.0, 1.2, 0.9, 1.1])  # peak 1.2 -> trough 0.9
    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1.0)


def test_sharpe_positive_for_upward_drift():
    daily = np.full(252, 0.001)
    equity = np.concatenate([[1.0], np.cumprod(1 + daily)])
    stats = summarize(daily, equity)

    assert stats.sharpe > 0
    assert stats.max_drawdown == pytest.approx(0.0)


def test_uptrending_asset_makes_money_via_momentum():
    """End-to-end: a clean uptrend should be caught long and profit."""
    closes = np.linspace(1.0, 1.5, 600)
    returns = np.full((600, 1), np.nan)
    returns[1:, 0] = closes[1:] / closes[:-1] - 1.0
    weights = target_weights(closes, n_assets=1)
    result = simulate(returns, weights, np.zeros((600, 1)), const_rebalance(600))

    assert result.equity[-1] > 1.0
