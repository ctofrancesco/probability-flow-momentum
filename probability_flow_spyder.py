"""
Probability Flow Momentum — independent quantitative model for Spyder.

This script transparently implements publicly described concepts used in
probability-flow indicators: Wasserstein distance, quantile transport
direction, tail asymmetry, and distributional velocity.

Quantile Transport Direction (QTD) measures the breadth of upward versus
downward displacements along the quantile function. It lies in [-1, +1] and
remains separate from Wasserstein distance. RAW signals are QTD zero-crossings;
CONFIRMED signals additionally require aligned Wasserstein intensity, tail,
and velocity evidence.

This is not the proprietary source code of the TradingView STARK sigma.3
script and does not claim to reproduce it exactly. The output is a diagnostic
score, not a calibrated probability, and is not investment advice.

Running in Spyder:
1. Edit the USER CONFIGURATION section.
2. Press F5 (or run the # %% cells in order).
3. The chart remains in the Plots pane and FIGURE/RESULTS remain available in
   Variable Explorer. No file is written to disk.
   DATA_SOURCE = "demo" does not require an internet connection.
"""

from __future__ import annotations

# %% IMPORT
from dataclasses import dataclass
from pathlib import Path
import math
import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# %% USER CONFIGURATION
# Supported values: "demo", "csv", "yfinance"
DATA_SOURCE = "demo"

# Used only when DATA_SOURCE = "csv".
# Accepted columns (case-insensitive):
# Date, Open, High, Low, Close, Adj Close, Volume.
CSV_PATH = ""

# Used only when DATA_SOURCE = "yfinance".
TICKER = "SPY"
YFINANCE_PERIOD = "5y"
YFINANCE_INTERVAL = "1d"

# Conservative choice: Yahoo may include the current, not-yet-closed bar.
# When False, metrics are displayed but RAW/CONFIRMED signals on the last row
# are suppressed because its close has not been verified.
YFINANCE_LAST_BAR_IS_COMPLETE = False

# Display only inside Spyder: no CSV, PNG, or output directory.
SHOW_FIGURE = True
MAX_BARS_IN_CHART = 360


@dataclass(frozen=True)
class IndicatorConfig:
    """Model parameters, all applied causally."""

    # Number of returns in the current and reference distributions.
    window: int = 64

    # The reference distribution ends `transport_lag` bars earlier.
    # The two windows overlap when this value is smaller than `window`.
    transport_lag: int = 8

    # Fraction of observations used for each tail (64 * 0.15 ~= 10).
    tail_fraction: float = 0.15

    # Quantile grid used for transport direction. Endpoints are excluded to
    # limit tail instability in small samples.
    quantile_min: float = 0.10
    quantile_max: float = 0.90
    quantile_count: int = 17

    # A quantile displacement, expressed in units of the reference window's
    # robust scale, votes +1/-1 only beyond this deadband. QTD remains
    # independent of Wasserstein distance.
    quantile_direction_deadband: float = 0.02

    # Causal EWMA smoothing.
    tail_ewm_span: int = 8
    velocity_ewm_span: int = 5
    score_ewm_span: int = 5

    # Historical baseline for robust z-scores. The baseline is shifted by 1.
    normalization_window: int = 252
    normalization_min_periods: int = 63
    zscore_clip: float = 5.0

    # Distributional-intensity weights (recommended sum = 1).
    wasserstein_weight: float = 0.70
    velocity_weight: float = 0.30

    # Causal confirmation: after a RAW zero-crossing, filters may confirm on
    # the same bar or within the next confirmation_window-1 bars. The signal is
    # recorded on the confirmation bar and is never backdated.
    confirmation_window: int = 8
    confirmation_min_direction: float = 0.25
    confirmation_min_wasserstein_z: float = 0.50
    confirmation_min_tail_z: float = 0.15
    confirmation_min_velocity_z: float = 0.25

    # STABLE / TRANSITION hysteresis. Entering TRANSITION uses confirmation
    # thresholds; returning to STABLE requires lower values.
    stable_max_direction: float = 0.15
    stable_max_wasserstein_z: float = 0.25
    stable_max_tail_z: float = 0.75
    stable_max_velocity_z: float = 0.25

    # Robust-scale floor: avoids division by zero without hiding the first
    # shock after a flat or nearly flat window.
    scale_floor: float = 1e-8

    def validate(self) -> None:
        integer_fields = (
            "window",
            "transport_lag",
            "tail_ewm_span",
            "velocity_ewm_span",
            "score_ewm_span",
            "normalization_window",
            "normalization_min_periods",
            "quantile_count",
            "confirmation_window",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.window < 20:
            raise ValueError("window must be at least 20.")
        if self.transport_lag < 1:
            raise ValueError("transport_lag must be at least 1.")
        if not 0.05 <= self.tail_fraction <= 0.30:
            raise ValueError("tail_fraction must be between 0.05 and 0.30.")
        if math.ceil(self.window * self.tail_fraction) < 5:
            raise ValueError(
                "Each tail contains fewer than 5 observations: increase window "
                "or tail_fraction. At least 8-10 observations are recommended."
            )
        if not 0 < self.quantile_min < 0.5 < self.quantile_max < 1:
            raise ValueError(
                "quantile_min and quantile_max must surround 0.5 and lie in (0, 1)."
            )
        if not np.isclose(self.quantile_min + self.quantile_max, 1.0):
            raise ValueError("The quantile grid must be symmetric around 0.5.")
        if self.quantile_count < 5 or self.quantile_count % 2 == 0:
            raise ValueError("quantile_count must be odd and at least 5.")
        if (
            not np.isfinite(self.quantile_direction_deadband)
            or self.quantile_direction_deadband < 0
        ):
            raise ValueError("quantile_direction_deadband must be finite and >= 0.")
        for name in ("tail_ewm_span", "velocity_ewm_span", "score_ewm_span"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.normalization_min_periods < 20:
            raise ValueError("normalization_min_periods must be at least 20.")
        if self.normalization_window < self.normalization_min_periods:
            raise ValueError(
                "normalization_window cannot be smaller than "
                "normalization_min_periods."
            )
        if not np.isfinite(self.zscore_clip) or self.zscore_clip <= 0:
            raise ValueError("zscore_clip must be positive and finite.")
        weight_names = (
            "wasserstein_weight",
            "velocity_weight",
        )
        for name in weight_names:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not np.isclose(self.wasserstein_weight + self.velocity_weight, 1.0):
            raise ValueError("Intensity weights must sum to 1.")
        if self.confirmation_window < 1:
            raise ValueError("confirmation_window must be at least 1.")
        unit_interval_fields = (
            "confirmation_min_direction",
            "stable_max_direction",
        )
        for name in unit_interval_fields:
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and lie in [0, 1].")
        z_threshold_fields = (
            "confirmation_min_wasserstein_z",
            "confirmation_min_tail_z",
            "confirmation_min_velocity_z",
            "stable_max_wasserstein_z",
            "stable_max_tail_z",
            "stable_max_velocity_z",
        )
        for name in z_threshold_fields:
            value = getattr(self, name)
            if not np.isfinite(value) or abs(value) > self.zscore_clip:
                raise ValueError(f"{name} must lie within the z-score range.")
        if self.stable_max_direction >= self.confirmation_min_direction:
            raise ValueError(
                "stable_max_direction must be below confirmation_min_direction."
            )
        if self.stable_max_wasserstein_z >= self.confirmation_min_wasserstein_z:
            raise ValueError(
                "stable_max_wasserstein_z must be below the confirmation threshold."
            )
        if self.stable_max_velocity_z > self.confirmation_min_velocity_z:
            raise ValueError(
                "stable_max_velocity_z cannot exceed the velocity confirmation "
                "threshold."
            )
        if not np.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor must be positive and finite.")


CONFIG = IndicatorConfig()
FIGURE: plt.Figure | None = None


# %% DATA
def _canonical_column_map(columns: pd.Index) -> dict[str, str]:
    """Return {normalized_name: original_name}."""
    return {
        re.sub(r"[^a-z0-9]+", "", str(column).strip().lower()): str(column)
        for column in columns
    }


def sanitize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize an OHLCV frame without fabricating returns."""
    if raw is None or raw.empty:
        raise ValueError("The dataset is empty.")

    data = raw.copy()
    columns = _canonical_column_map(data.columns)

    # If the date is stored as a column, move it to the index.
    date_key = next(
        (key for key in ("date", "datetime", "timestamp", "time") if key in columns),
        None,
    )
    if date_key is not None:
        if pd.api.types.is_numeric_dtype(data[columns[date_key]].dtype):
            raise ValueError(
                "The date column is numeric and its epoch unit would be ambiguous. "
                "Convert it to ISO-8601 dates first (for example "
                "2026-08-22T16:00:00Z)."
            )
        parsed = pd.to_datetime(data[columns[date_key]], errors="coerce", utc=True)
        if bool(parsed.isna().any()):
            raise ValueError(
                f"The date column contains {int(parsed.isna().sum())} values "
                "that cannot be parsed. Correct them before calculation."
            )
        data = data.copy()
        parsed = parsed.dt.tz_convert(None)
        data.index = pd.DatetimeIndex(parsed, name="Date")
        data.drop(columns=[columns[date_key]], inplace=True)
        columns = _canonical_column_map(data.columns)
    else:
        if isinstance(data.index, pd.RangeIndex) or pd.api.types.is_numeric_dtype(
            data.index.dtype
        ):
            raise ValueError(
                "A genuine Date/Datetime column is required. A numeric index is "
                "not interpreted automatically as a date."
            )
        parsed_index = pd.to_datetime(data.index, errors="coerce", utc=True)
        valid = ~pd.isna(parsed_index)
        if not bool(valid.all()):
            raise ValueError(
                f"The index contains {int((~valid).sum())} unparseable dates."
            )
        data = data.copy()
        data.index = pd.DatetimeIndex(parsed_index).tz_convert(None)
        data.index.name = "Date"

    if "close" not in columns:
        raise ValueError("The dataset does not contain a Close column.")

    standardized = pd.DataFrame(index=data.index)
    for target, key in (
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Volume", "volume"),
    ):
        if key in columns:
            standardized[target] = pd.to_numeric(
                data[columns[key]], errors="coerce"
            )

    # Accept either the common "Adj Close" or "Adjusted Close" name.
    adjusted_key = next(
        (
            key
            for key in ("analysisclose", "adjclose", "adjustedclose")
            if key in columns
        ),
        None,
    )
    if adjusted_key is not None:
        standardized["AnalysisClose"] = pd.to_numeric(
            data[columns[adjusted_key]], errors="coerce"
        )
    else:
        standardized["AnalysisClose"] = standardized["Close"]

    # Keep charting available for close-only files.
    for column in ("Open", "High", "Low"):
        if column not in standardized:
            standardized[column] = standardized["Close"]
        else:
            # Used only to make a partial OHLC bar drawable; AnalysisClose and
            # therefore returns are not filled.
            standardized[column] = standardized[column].fillna(standardized["Close"])
    if "Volume" not in standardized:
        standardized["Volume"] = 0.0

    standardized = standardized[
        ["Open", "High", "Low", "Close", "Volume", "AnalysisClose"]
    ]
    standardized = standardized.replace([np.inf, -np.inf], np.nan)
    standardized = standardized[~standardized.index.duplicated(keep="last")]
    standardized = standardized.sort_index()

    # No backfill or forward-fill: a missing quote remains missing.
    valid_price = (
        standardized[["Close", "AnalysisClose"]].notna().all(axis=1)
        & (standardized["Close"] > 0)
        & (standardized["AnalysisClose"] > 0)
    )
    if not bool(valid_price.all()):
        invalid_count = int((~valid_price).sum())
        raise ValueError(
            f"The dataset contains {invalid_count} rows with missing, infinite, "
            "or non-positive Close/Adj Close values. Correct or remove those rows "
            "explicitly before calculation; the program does not silently bridge "
            "these gaps."
        )
    standardized["Volume"] = standardized["Volume"].fillna(0.0).clip(lower=0.0)

    if len(standardized) < 3:
        raise ValueError("At least 3 valid prices are required.")
    return standardized


def build_demo_data(n_bars: int = 950, seed: int = 73) -> pd.DataFrame:
    """Create deterministic OHLCV data with known regime and tail shifts."""
    if n_bars < 250:
        raise ValueError("The demo requires at least 250 bars.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_bars)

    # Five regimes make mean, variance, and skewness shifts visible.
    edges = np.linspace(0, n_bars, 6, dtype=int)
    drift = np.zeros(n_bars)
    volatility = np.zeros(n_bars)
    tail_sign = np.zeros(n_bars)
    regimes = [
        (0.00025, 0.0080, 0.0),
        (-0.00045, 0.0150, -1.0),
        (0.00005, 0.0065, 0.0),
        (0.00065, 0.0110, 1.0),
        (-0.00010, 0.0170, -0.5),
    ]
    for i, (mu, sigma, skew) in enumerate(regimes):
        drift[edges[i] : edges[i + 1]] = mu
        volatility[edges[i] : edges[i + 1]] = sigma
        tail_sign[edges[i] : edges[i + 1]] = skew

    log_returns = drift + volatility * rng.standard_normal(n_bars)
    shock_mask = rng.random(n_bars) < 0.035
    shock_size = rng.exponential(scale=1.8, size=n_bars) * volatility
    log_returns += shock_mask * tail_sign * shock_size

    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = np.r_[close[0], close[:-1]] * np.exp(
        rng.normal(0.0, volatility * 0.20)
    )
    intraday = np.abs(rng.normal(0.0, volatility * 0.55))
    high = np.maximum(open_, close) * np.exp(intraday)
    low = np.minimum(open_, close) * np.exp(-intraday)
    volume = rng.lognormal(mean=16.2, sigma=0.32, size=n_bars)
    volume *= 1.0 + 18.0 * np.abs(log_returns)

    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "AnalysisClose": close,
        },
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return sanitize_ohlcv(pd.read_csv(path))


def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "For DATA_SOURCE='yfinance', install yfinance with: "
            "pip install yfinance"
        ) from exc

    history = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=True,
        actions=False,
    )
    if history.empty:
        raise ValueError(
            f"No data received for {symbol!r}. Check the symbol, period, interval, "
            "and internet connection."
        )
    return sanitize_ohlcv(history)


def load_data() -> tuple[pd.DataFrame, str]:
    source = DATA_SOURCE.strip().lower()
    if source == "demo":
        return build_demo_data(), "DEMO"
    if source == "csv":
        if not CSV_PATH.strip():
            raise ValueError("Set CSV_PATH when DATA_SOURCE='csv'.")
        return load_csv(CSV_PATH), Path(CSV_PATH).stem
    if source == "yfinance":
        return (
            load_yfinance(TICKER, YFINANCE_PERIOD, YFINANCE_INTERVAL),
            TICKER.upper(),
        )
    raise ValueError("DATA_SOURCE must be 'demo', 'csv', or 'yfinance'.")


# %% PROBABILITY-FLOW CALCULATION
def _robust_reference_scale(values: np.ndarray, floor: float) -> float:
    """Robust scale of the reference window alone, bounded below by floor."""
    median = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - median)))
    q25, q75 = np.quantile(values, [0.25, 0.75], method="linear")
    iqr_sigma = float(q75 - q25) / 1.349
    scale = max(mad_sigma, iqr_sigma)
    return max(scale, floor) if np.isfinite(scale) else np.nan


def _distribution_metrics(
    reference: np.ndarray,
    current: np.ndarray,
    previous_current: np.ndarray,
    tail_fraction: float,
    quantile_levels: np.ndarray,
    quantile_direction_deadband: float,
    scale_floor: float,
) -> dict[str, float]:
    """One-dimensional OT metrics with direction kept separate from Wasserstein.

    For equally weighted empirical samples of the same length, monotone matching
    between order statistics gives the exact Wasserstein-1 distance. QTD instead
    measures the net breadth of quantiles displaced upward versus downward and
    is never multiplied by Wasserstein distance.
    """
    empty = {
        "WassersteinRaw": np.nan,
        "WassersteinNormalized": np.nan,
        "QuantileTransportDirectionRaw": np.nan,
        "QuantileTransportDirection": np.nan,
        "QuantileUpMass": np.nan,
        "QuantileDownMass": np.nan,
        "QuantileStaticMass": np.nan,
        "QuantileActiveMass": np.nan,
        "QuantileDirectionCoherence": np.nan,
        "TailAsymmetryRaw": np.nan,
        "TailAsymmetry": np.nan,
        "TransportVelocityRaw": np.nan,
        "TransportVelocityNormalized": np.nan,
    }
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    previous_current = np.asarray(previous_current, dtype=float)
    if not (len(reference) == len(current) == len(previous_current)):
        raise ValueError("All three windows must have the same length.")
    if not (
        np.isfinite(reference).all()
        and np.isfinite(current).all()
        and np.isfinite(previous_current).all()
    ):
        return empty.copy()

    scale = _robust_reference_scale(reference, scale_floor)
    if not np.isfinite(scale):
        return empty.copy()

    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    prev_cur_sorted = np.sort(previous_current)

    displacement = cur_sorted - ref_sorted
    wasserstein_raw = float(np.mean(np.abs(displacement)))

    # Quantile Transport Direction (QTD): each quantile has equal weight.
    # The raw vote uses only the sign. The operational vote assigns 0 to
    # displacements within the causal deadband, expressed in reference-scale units.
    reference_quantiles = np.quantile(reference, quantile_levels, method="linear")
    current_quantiles = np.quantile(current, quantile_levels, method="linear")
    quantile_displacement = (current_quantiles - reference_quantiles) / scale

    # Floating-point residue around zero must not count as directional breadth
    # (for example, the median of a symmetric expansion).
    numerical_zero = 64.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(quantile_displacement))),
    )
    raw_up_mass = float(np.mean(quantile_displacement > numerical_zero))
    raw_down_mass = float(np.mean(quantile_displacement < -numerical_zero))
    qtd_raw = raw_up_mass - raw_down_mass

    up_mass = float(
        np.mean(quantile_displacement > quantile_direction_deadband)
    )
    down_mass = float(
        np.mean(quantile_displacement < -quantile_direction_deadband)
    )
    static_mass = float(max(0.0, 1.0 - up_mass - down_mass))
    active_mass = up_mass + down_mass
    qtd = float(np.clip(up_mass - down_mass, -1.0, 1.0))
    coherence = abs(qtd) / active_mass if active_mass > 0.0 else 0.0

    # Shape tail asymmetry: remove the median so that a simple translation or a
    # symmetric expansion is not mistaken for tail skew.
    k = max(1, int(math.ceil(len(current) * tail_fraction)))

    def tail_balance(sorted_values: np.ndarray) -> float:
        center = float(np.median(sorted_values))
        upper_extent = float(np.mean(sorted_values[-k:] - center))
        lower_extent = float(np.mean(center - sorted_values[:k]))
        return upper_extent - lower_extent

    tail_change_raw = tail_balance(cur_sorted) - tail_balance(ref_sorted)

    # Effective distributional velocity between windows t-1 and t.
    one_bar_displacement = cur_sorted - prev_cur_sorted
    velocity_raw = float(np.mean(np.abs(one_bar_displacement)))
    previous_scale = _robust_reference_scale(previous_current, scale_floor)
    velocity_normalized = (
        velocity_raw / previous_scale if np.isfinite(previous_scale) else np.nan
    )

    wasserstein_normalized = wasserstein_raw / scale
    return {
        "WassersteinRaw": wasserstein_raw,
        "WassersteinNormalized": float(wasserstein_normalized),
        "QuantileTransportDirectionRaw": float(qtd_raw),
        "QuantileTransportDirection": qtd,
        "QuantileUpMass": up_mass,
        "QuantileDownMass": down_mass,
        "QuantileStaticMass": static_mass,
        "QuantileActiveMass": active_mass,
        "QuantileDirectionCoherence": float(coherence),
        "TailAsymmetryRaw": float(tail_change_raw),
        "TailAsymmetry": float(tail_change_raw / scale),
        "TransportVelocityRaw": velocity_raw,
        "TransportVelocityNormalized": float(velocity_normalized),
    }


def _causal_robust_zscore(
    series: pd.Series,
    lookback: int,
    min_periods: int,
    clip: float,
    epsilon: float = 1e-12,
) -> pd.Series:
    """Median/MAD z-score using a baseline made only of prior bars."""
    history = series.shift(1)
    rolling = history.rolling(lookback, min_periods=min_periods)
    center = rolling.median()
    mad = rolling.apply(
        # rolling.apply may include NaNs from the initial warm-up even after
        # min_periods is met; nanmedian preserves the intended meaning of
        # normalization_min_periods.
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
    )
    mad_scale = 1.4826 * mad
    iqr_scale = (rolling.quantile(0.75) - rolling.quantile(0.25)) / 1.349
    std_scale = rolling.std(ddof=0)
    robust_scale = pd.concat([mad_scale, iqr_scale], axis=1).max(
        axis=1, skipna=True
    )
    # Standard deviation is only a fallback when both robust scales collapse;
    # it cannot dominate the baseline because of a single outlier. If the
    # baseline is perfectly flat, epsilon still makes the first deviation
    # observable (before it is bounded by `clip`).
    scale = robust_scale.where(robust_scale > epsilon, std_scale)
    baseline_ready = center.notna()
    safe_scale = scale.where(scale > epsilon, epsilon)
    difference = series - center
    z = (difference / safe_scale).where(baseline_ready)
    return z.replace([np.inf, -np.inf], np.nan).clip(-clip, clip)


def _causal_zero_centered_score(
    series: pd.Series,
    lookback: int,
    min_periods: int,
    clip: float,
    epsilon: float = 1e-12,
) -> pd.Series:
    """Causal scale that preserves zero and sign as the natural reference.

    This is used for directional components: unlike a z-score centered on the
    historical median, a positive value cannot become negative merely because
    it is below its recent baseline.
    """
    history = series.shift(1)
    rolling = history.rolling(lookback, min_periods=min_periods)
    median_absolute = rolling.apply(lambda x: np.nanmedian(np.abs(x)), raw=True)
    median_absolute_scale = median_absolute / 0.6744897501960817
    iqr_scale = (rolling.quantile(0.75) - rolling.quantile(0.25)) / 1.349
    rms_scale = rolling.apply(
        lambda x: np.sqrt(np.nanmean(np.square(x))), raw=True
    )
    robust_scale = pd.concat([median_absolute_scale, iqr_scale], axis=1).max(
        axis=1, skipna=True
    )
    # RMS is a fallback, not a competitor to the robust scales. Here too, the
    # floor preserves the first non-zero signal after a flat baseline.
    scale = robust_scale.where(robust_scale > epsilon, rms_scale)
    baseline_ready = median_absolute.notna()
    safe_scale = scale.where(scale > epsilon, epsilon)
    score = (series / safe_scale).where(baseline_ready)
    return score.replace([np.inf, -np.inf], np.nan).clip(-clip, clip)


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(-30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _classify_states_and_signals(
    result: pd.DataFrame, config: IndicatorConfig
) -> pd.DataFrame:
    """Build states and signals in strictly causal order.

    RAW_BUY/RAW_SELL are zero-crossings of significant QTD. Confirmation may
    arrive on the same bar or within `confirmation_window`, but it is always
    recorded on the bar where all filters are actually satisfied.
    """
    n_rows = len(result)
    metrics_ready_values: list[bool] = []
    direction_sign_values: list[int] = []
    wasserstein_gate_values: list[bool] = []
    tail_buy_gate_values: list[bool] = []
    tail_sell_gate_values: list[bool] = []
    velocity_gate_values: list[bool] = []
    raw_signals: list[str] = []
    confirmed_signals: list[str] = []
    confirmation_lags: list[float] = []
    pending_sides: list[str] = []
    pending_remaining: list[int] = []
    market_states: list[str] = []
    directional_regimes: list[str] = []
    regimes: list[str] = []
    state_before_values: list[str] = []
    regime_changed_values: list[bool] = []
    state_reasons: list[str] = []

    has_ever_been_ready = False
    continuous_ready = False
    last_nonzero_sign = 0
    pending_side = ""
    pending_age = -1
    current_market_state = "WARMUP"
    previous_regime = "WARMUP"

    for _, row in result.iterrows():
        direction = row["QuantileTransportDirection"]
        wasserstein_z = row["WassersteinZ"]
        tail_z = row["TailAsymmetryZ"]
        velocity_z = row["TransportVelocityZ"]
        ready = bool(
            np.isfinite(direction)
            and np.isfinite(wasserstein_z)
            and np.isfinite(tail_z)
            and np.isfinite(velocity_z)
        )
        metrics_ready_values.append(ready)
        state_before_values.append(previous_regime)

        raw_signal = ""
        confirmed_signal = ""
        confirmation_lag = np.nan
        direction_sign = 0
        wasserstein_gate = False
        tail_buy_gate = False
        tail_sell_gate = False
        velocity_gate = False
        directional_regime = "NEUTRAL"

        if not ready:
            current_market_state = (
                "UNAVAILABLE" if has_ever_been_ready else "WARMUP"
            )
            continuous_ready = False
            last_nonzero_sign = 0
            pending_side = ""
            pending_age = -1
            regime = current_market_state
            reason = (
                "NORMALIZER_NOT_READY"
                if not has_ever_been_ready
                else "METRIC_NOT_AVAILABLE"
            )
        else:
            direction_sign = 1 if direction > 0.0 else (-1 if direction < 0.0 else 0)
            wasserstein_gate = bool(
                wasserstein_z >= config.confirmation_min_wasserstein_z
            )
            tail_buy_gate = bool(tail_z >= config.confirmation_min_tail_z)
            tail_sell_gate = bool(tail_z <= -config.confirmation_min_tail_z)
            velocity_gate = bool(
                velocity_z >= config.confirmation_min_velocity_z
            )
            direction_buy_gate = bool(
                direction >= config.confirmation_min_direction
            )
            direction_sell_gate = bool(
                direction <= -config.confirmation_min_direction
            )
            buy_filters = bool(
                direction_buy_gate
                and wasserstein_gate
                and tail_buy_gate
                and velocity_gate
            )
            sell_filters = bool(
                direction_sell_gate
                and wasserstein_gate
                and tail_sell_gate
                and velocity_gate
            )

            first_ready_after_gap = not continuous_ready
            if first_ready_after_gap:
                last_nonzero_sign = direction_sign
                pending_side = ""
                pending_age = -1
                continuous_ready = True
            elif direction_sign != 0:
                if last_nonzero_sign == -1 and direction_sign == 1:
                    raw_signal = "RAW_BUY"
                    pending_side = "BUY"
                    pending_age = 0
                elif last_nonzero_sign == 1 and direction_sign == -1:
                    raw_signal = "RAW_SELL"
                    pending_side = "SELL"
                    pending_age = 0
                last_nonzero_sign = direction_sign

            # Confirmation is evaluated on the RAW bar (lag 0) and then on
            # subsequent bars. No confirmation is backdated.
            if pending_side and pending_age >= 0:
                side_still_valid = (
                    pending_side == "BUY" and direction_sign >= 0
                ) or (pending_side == "SELL" and direction_sign <= 0)
                filters_pass = (
                    pending_side == "BUY" and buy_filters
                ) or (pending_side == "SELL" and sell_filters)
                if side_still_valid and filters_pass:
                    confirmed_signal = pending_side
                    confirmation_lag = float(pending_age)
                    pending_side = ""
                    pending_age = -1
                elif not side_still_valid:
                    pending_side = ""
                    pending_age = -1
                else:
                    pending_age += 1
                    if pending_age >= config.confirmation_window:
                        pending_side = ""
                        pending_age = -1

            stable_condition = bool(
                abs(direction) <= config.stable_max_direction
                and wasserstein_z <= config.stable_max_wasserstein_z
                and abs(tail_z) <= config.stable_max_tail_z
                and velocity_z <= config.stable_max_velocity_z
            )
            # TRANSITION represents meaningful distributional activity, not
            # merely a directional sign. The noisier tail component opens the
            # state only when aligned with an already significant direction.
            transition_entry = bool(
                raw_signal != ""
                or wasserstein_gate
                or velocity_gate
                or (
                    abs(direction) >= config.confirmation_min_direction
                    and abs(tail_z) >= config.confirmation_min_tail_z
                )
            )

            if first_ready_after_gap:
                current_market_state = "STABLE" if stable_condition else "TRANSITION"
            elif raw_signal:
                current_market_state = "TRANSITION"
            elif current_market_state == "STABLE" and transition_entry:
                current_market_state = "TRANSITION"
            elif current_market_state == "TRANSITION" and stable_condition:
                current_market_state = "STABLE"

            if buy_filters:
                directional_regime = "BULL"
            elif sell_filters:
                directional_regime = "BEAR"

            regime = (
                directional_regime
                if directional_regime in ("BULL", "BEAR")
                else current_market_state
            )
            if confirmed_signal:
                reason = f"CONFIRMED_{confirmed_signal}"
            elif raw_signal:
                reason = "RAW_ZERO_CROSS"
            elif directional_regime != "NEUTRAL":
                reason = "FILTERED_DIRECTION"
            elif current_market_state == "STABLE":
                reason = "LOW_DISTRIBUTIONAL_ACTIVITY"
            else:
                reason = "DISTRIBUTIONAL_TRANSITION"

            has_ever_been_ready = True

        wasserstein_gate_values.append(wasserstein_gate)
        tail_buy_gate_values.append(tail_buy_gate)
        tail_sell_gate_values.append(tail_sell_gate)
        velocity_gate_values.append(velocity_gate)
        direction_sign_values.append(direction_sign)
        raw_signals.append(raw_signal)
        confirmed_signals.append(confirmed_signal)
        confirmation_lags.append(confirmation_lag)
        pending_sides.append(pending_side)
        pending_remaining.append(
            max(0, config.confirmation_window - pending_age)
            if pending_age >= 0
            else 0
        )
        market_states.append(current_market_state)
        directional_regimes.append(directional_regime)
        regimes.append(regime)
        changed = regime != previous_regime
        regime_changed_values.append(changed)
        state_reasons.append(reason)
        previous_regime = regime

    if not all(len(values) == n_rows for values in (regimes, raw_signals, confirmed_signals)):
        raise RuntimeError("Internal error while aligning states and signals.")

    result = result.copy()
    result["MetricsReady"] = metrics_ready_values
    result["DirectionSign"] = direction_sign_values
    result["WassersteinGate"] = wasserstein_gate_values
    result["TailBuyGate"] = tail_buy_gate_values
    result["TailSellGate"] = tail_sell_gate_values
    result["VelocityGate"] = velocity_gate_values
    result["RawSignalAtClose"] = raw_signals
    result["RawBuyAtClose"] = result["RawSignalAtClose"].eq("RAW_BUY")
    result["RawSellAtClose"] = result["RawSignalAtClose"].eq("RAW_SELL")
    result["ConfirmedSignalAtClose"] = confirmed_signals
    result["ConfirmedBuyAtClose"] = result["ConfirmedSignalAtClose"].eq("BUY")
    result["ConfirmedSellAtClose"] = result["ConfirmedSignalAtClose"].eq("SELL")
    result["ConfirmationLagBars"] = confirmation_lags
    result["PendingSide"] = pending_sides
    result["PendingBarsRemaining"] = pending_remaining
    result["MarketState"] = market_states
    result["DirectionalRegime"] = directional_regimes
    result["StateBefore"] = state_before_values
    result["Regime"] = regimes
    result["RegimeChanged"] = regime_changed_values
    result["StateReason"] = state_reasons

    # Backward compatibility: SignalAtClose and ActionNextBar represent only
    # CONFIRMED signals.
    result["SignalAtClose"] = result["ConfirmedSignalAtClose"]
    result["BullSignal"] = result["ConfirmedBuyAtClose"]
    result["BearSignal"] = result["ConfirmedSellAtClose"]
    result["ShiftSignal"] = False
    result["RawActionNextBar"] = result["RawSignalAtClose"].shift(1).fillna("")
    result["ConfirmedActionNextBar"] = (
        result["ConfirmedSignalAtClose"].shift(1).fillna("")
    )
    result["ActionNextBar"] = result["ConfirmedActionNextBar"]
    return result


def calculate_probability_flow(
    ohlcv: pd.DataFrame, config: IndicatorConfig = CONFIG
) -> pd.DataFrame:
    """Calculate all metrics and return a DataFrame aligned with prices."""
    config.validate()
    data = sanitize_ohlcv(ohlcv)
    returns = np.log(data["AnalysisClose"] / data["AnalysisClose"].shift(1))

    required = config.window + config.transport_lag + config.normalization_min_periods
    if returns.notna().sum() < required:
        raise ValueError(
            f"Insufficient data: approximately {required + 1} prices are required; "
            f"{len(data)} rows are available."
        )

    metric_names = [
        "WassersteinRaw",
        "WassersteinNormalized",
        "QuantileTransportDirectionRaw",
        "QuantileTransportDirection",
        "QuantileUpMass",
        "QuantileDownMass",
        "QuantileStaticMass",
        "QuantileActiveMass",
        "QuantileDirectionCoherence",
        "TailAsymmetryRaw",
        "TailAsymmetry",
        "TransportVelocityRaw",
        "TransportVelocityNormalized",
    ]
    arrays = {name: np.full(len(data), np.nan, dtype=float) for name in metric_names}
    return_values = returns.to_numpy(dtype=float)
    quantile_levels = np.linspace(
        config.quantile_min,
        config.quantile_max,
        config.quantile_count,
    )

    # Indexing: current contains [i-window+1, ..., i]; reference ends at i-lag.
    # LogReturn[0] is NaN, so the first fully valid reference starts at position
    # 1 and produces the first output at window + transport_lag.
    first_position = config.window + config.transport_lag
    for i in range(first_position, len(data)):
        current_start = i - config.window + 1
        current = return_values[current_start : i + 1]

        reference_end = i - config.transport_lag + 1  # exclusive
        reference_start = reference_end - config.window
        reference = return_values[reference_start:reference_end]

        previous_current = return_values[current_start - 1 : i]
        metrics = _distribution_metrics(
            reference,
            current,
            previous_current,
            config.tail_fraction,
            quantile_levels,
            config.quantile_direction_deadband,
            config.scale_floor,
        )
        for name, value in metrics.items():
            arrays[name][i] = value

    result = data.copy()
    result["LogReturn"] = returns
    for name, values in arrays.items():
        result[name] = values

    # Backward-compatible aliases. TransportDirection is now the genuine
    # multi-quantile direction; SignedTransport is intentionally removed.
    result["Wasserstein"] = result["WassersteinNormalized"]
    result["TransportDirection"] = result["QuantileTransportDirection"]

    result["TailAsymmetrySmoothed"] = result["TailAsymmetry"].ewm(
        span=config.tail_ewm_span,
        adjust=False,
        min_periods=1,
    ).mean()
    result["TransportVelocity"] = result["TransportVelocityNormalized"].ewm(
        span=config.velocity_ewm_span,
        adjust=False,
        min_periods=1,
    ).mean()

    result["WassersteinZ"] = _causal_robust_zscore(
        np.log1p(result["WassersteinNormalized"]),
        config.normalization_window,
        config.normalization_min_periods,
        config.zscore_clip,
    )
    result["TailAsymmetryZ"] = _causal_zero_centered_score(
        result["TailAsymmetrySmoothed"],
        config.normalization_window,
        config.normalization_min_periods,
        config.zscore_clip,
    )
    result["TransportVelocityZ"] = _causal_robust_zscore(
        np.log1p(result["TransportVelocity"]),
        config.normalization_window,
        config.normalization_min_periods,
        config.zscore_clip,
    )

    # The directional line contains no W1, tail, or velocity component: it is
    # exclusively multi-quantile QTD, causally smoothed for display.
    result["QuantileTransportDirectionSmoothed"] = result[
        "QuantileTransportDirection"
    ].ewm(
        span=config.score_ewm_span,
        adjust=False,
        min_periods=1,
    ).mean()
    result["ProbabilityFlowScore"] = (
        100.0 * result["QuantileTransportDirectionSmoothed"]
    )

    activity_linear = (
        config.wasserstein_weight * result["WassersteinZ"]
        + config.velocity_weight * result["TransportVelocityZ"]
        - 0.5
    )
    result["TransitionIntensity"] = 100.0 * _sigmoid(activity_linear)

    # Confidence combines directional breadth with support from the three
    # filters; it is not a probability of profit or future direction.
    expected_tail = np.sign(result["QuantileTransportDirection"]) * result[
        "TailAsymmetryZ"
    ]
    confirmation_support = _sigmoid(
        (
            result["WassersteinZ"] - config.confirmation_min_wasserstein_z
            + result["TransportVelocityZ"] - config.confirmation_min_velocity_z
            + expected_tail - config.confirmation_min_tail_z
        )
        / 3.0
    )
    result["ConfidenceScore"] = 100.0 * (
        result["QuantileTransportDirection"].abs().clip(0.0, 1.0)
        * confirmation_support
    )
    result = _classify_states_and_signals(result, config)
    return result


# %% CHART
PALETTE = {
    "navy": "#0B2D6B",
    "text": "#26364F",
    "grid": "#E6ECF4",
    "background": "#F5F8FC",
    "panel": "#FFFFFF",
    "bull": "#2A9D8F",
    "bear": "#EF5350",
    "transition": "#D4A017",
    "wasserstein": "#1D4ED8",
    "direction": "#10B981",
    "tail": "#F59E0B",
    "velocity": "#8B5CF6",
}


def _draw_candles(ax: plt.Axes, view: pd.DataFrame, x: np.ndarray) -> None:
    if len(x) < 2:
        candle_width = 0.65
    else:
        candle_width = 0.68 * float(np.median(np.diff(x)))

    for xi, (_, row) in zip(x, view.iterrows()):
        color = PALETTE["bull"] if row["Close"] >= row["Open"] else PALETTE["bear"]
        ax.vlines(xi, row["Low"], row["High"], color=color, linewidth=0.8, alpha=0.9)
        lower = min(row["Open"], row["Close"])
        height = abs(row["Close"] - row["Open"])
        min_height = max(abs(row["Close"]) * 1e-5, 1e-8)
        body = Rectangle(
            (xi - candle_width / 2.0, lower),
            candle_width,
            max(height, min_height),
            facecolor=color,
            edgecolor=color,
            linewidth=0.6,
        )
        ax.add_patch(body)


def _shade_regimes(ax: plt.Axes, regimes: pd.Series, x: np.ndarray) -> None:
    colors = {
        "STABLE": "#EEF3F8",
        "TRANSITION": "#FFF1B8",
        "WARMUP": "#E9EEF5",
        "UNAVAILABLE": "#E9EEF5",
        "PROVISIONAL": "#E9EEF5",
    }
    if len(regimes) == 0:
        return
    width = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    start = 0
    values = regimes.astype(str).to_numpy()
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            ax.axvspan(
                x[start] - width / 2.0,
                x[i - 1] + width / 2.0,
                color=colors.get(values[start], colors["WARMUP"]),
                alpha=0.46,
                linewidth=0,
                zorder=0,
            )
            start = i


def plot_probability_flow(
    result: pd.DataFrame,
    label: str,
    config: IndicatorConfig = CONFIG,
    max_bars: int = MAX_BARS_IN_CHART,
    show: bool = True,
) -> plt.Figure:
    """Create a three-panel dashboard: price, volume, and probability flow."""
    if max_bars < 50:
        raise ValueError("max_bars must be at least 50.")
    view = result.iloc[-max_bars:].copy()
    x = mdates.date2num(view.index.to_pydatetime())

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
        }
    )
    fig = plt.figure(figsize=(16, 9), facecolor=PALETTE["background"])
    grid = fig.add_gridspec(3, 1, height_ratios=[6.2, 1.15, 3.0], hspace=0.04)
    ax_price = fig.add_subplot(grid[0])
    ax_volume = fig.add_subplot(grid[1], sharex=ax_price)
    ax_flow = fig.add_subplot(grid[2], sharex=ax_price)

    for ax in (ax_price, ax_volume, ax_flow):
        ax.set_facecolor(PALETTE["panel"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.75)
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_color(PALETTE["grid"])
        ax.spines["bottom"].set_color(PALETTE["grid"])

    _draw_candles(ax_price, view, x)
    ax_price.set_title(
        f"{label}  |  Probability Flow Momentum — independent model",
        loc="left",
        color=PALETTE["navy"],
        fontweight="bold",
        pad=10,
    )
    last_close = float(view["Close"].iloc[-1])
    ax_price.axhline(last_close, color=PALETTE["bear"], linestyle=":", linewidth=0.9)
    ax_price.text(
        1.002,
        last_close,
        f" {last_close:,.2f} ",
        transform=ax_price.get_yaxis_transform(),
        va="center",
        ha="left",
        color="white",
        fontsize=8,
        bbox={"facecolor": PALETTE["bear"], "edgecolor": "none", "pad": 2.5},
    )

    volume_colors = np.where(
        view["Close"].to_numpy() >= view["Open"].to_numpy(),
        PALETTE["bull"],
        PALETTE["bear"],
    )
    width = 0.72 * float(np.median(np.diff(x))) if len(x) > 1 else 0.72
    ax_volume.bar(x, view["Volume"], width=width, color=volume_colors, alpha=0.45)
    ax_volume.set_ylabel("Vol.", rotation=0, labelpad=18)
    ax_volume.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                f"{value / 1e9:.1f}B" if abs(value) >= 1e9 else f"{value / 1e6:.0f}M"
            )
        )
    )

    _shade_regimes(ax_flow, view["MarketState"], x)

    score_plot = view["QuantileTransportDirectionSmoothed"]
    ax_flow.plot(
        x,
        score_plot,
        color=PALETTE["navy"],
        linewidth=2.4,
        label="QTD smoothed",
        zorder=5,
    )
    ax_flow.fill_between(
        x,
        0.0,
        score_plot.to_numpy(dtype=float),
        where=(score_plot.to_numpy(dtype=float) >= 0),
        color=PALETTE["bull"],
        alpha=0.15,
        interpolate=True,
    )
    ax_flow.fill_between(
        x,
        0.0,
        score_plot.to_numpy(dtype=float),
        where=(score_plot.to_numpy(dtype=float) < 0),
        color=PALETTE["bear"],
        alpha=0.15,
        interpolate=True,
    )
    ax_flow.plot(
        x,
        view["WassersteinZ"],
        color=PALETTE["wasserstein"],
        linewidth=1.0,
        alpha=0.72,
        label="Wasserstein z",
    )
    ax_flow.plot(
        x,
        view["QuantileTransportDirection"],
        color=PALETTE["direction"],
        linewidth=1.1,
        alpha=0.76,
        label="QTD",
    )
    ax_flow.plot(
        x,
        view["TailAsymmetryZ"],
        color=PALETTE["tail"],
        linewidth=1.0,
        alpha=0.72,
        label="Tail asymmetry z",
    )
    ax_flow.plot(
        x,
        view["TransportVelocityZ"],
        color=PALETTE["velocity"],
        linewidth=1.0,
        alpha=0.72,
        label="Velocity z",
    )

    ax_flow.axhline(0.0, color=PALETTE["text"], linewidth=0.8, alpha=0.7)
    ax_flow.axhline(
        config.confirmation_min_direction,
        color=PALETTE["bull"],
        linestyle="--",
        linewidth=0.9,
    )
    ax_flow.axhline(
        -config.confirmation_min_direction,
        color=PALETTE["bear"],
        linestyle="--",
        linewidth=0.9,
    )
    ax_flow.set_ylabel("Score / z", rotation=0, labelpad=28)

    # RAW: small hollow marker at the QTD zero-crossing.
    for signal, marker, color in (
        ("RAW_BUY", "^", PALETTE["bull"]),
        ("RAW_SELL", "v", PALETTE["bear"]),
    ):
        mask = view["RawSignalAtClose"].eq(signal).to_numpy()
        if not mask.any():
            continue
        signal_y = view["QuantileTransportDirection"].to_numpy(dtype=float)[mask]
        ax_flow.scatter(
            x[mask],
            signal_y,
            marker=marker,
            s=30,
            facecolors="none",
            edgecolors=color,
            linewidth=1.2,
            alpha=0.70,
            zorder=7,
        )

    # CONFIRMED: filled marker and label on the actual confirmation bar.
    for signal, marker, color, y_offset in (
        ("BUY", "^", PALETTE["bull"], 14),
        ("SELL", "v", PALETTE["bear"], -18),
    ):
        mask = view["ConfirmedSignalAtClose"].eq(signal).to_numpy()
        if not mask.any():
            continue
        signal_y = view["QuantileTransportDirection"].to_numpy(dtype=float)[mask]
        ax_flow.scatter(
            x[mask],
            signal_y,
            marker=marker,
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=9,
        )
        for xi, yi in zip(x[mask], signal_y):
            ax_flow.annotate(
                signal,
                (xi, yi),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if y_offset >= 0 else "top",
                color="white",
                fontsize=6.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.20", "fc": color, "ec": "none"},
                zorder=9,
            )

    latest_valid = view.dropna(
        subset=[
            "QuantileTransportDirection",
            "TransitionIntensity",
            "WassersteinNormalized",
        ]
    )
    if not latest_valid.empty:
        latest = latest_valid.iloc[-1]
        status_color = {
            "STABLE": "#DCE6F0",
            "TRANSITION": "#F2C94C",
        }.get(str(latest["MarketState"]), "#DCE6F0")
        is_provisional = str(latest.get("BarStatus", "CONFIRMED")) == "PROVISIONAL"
        provisional_tag = " (PROV.)" if is_provisional else ""
        status = (
            f"STATE {latest['MarketState']}{provisional_tag}  ·  "
            f"DIR {latest['DirectionalRegime']}  ·  "
            f"QTD {latest['QuantileTransportDirection']:+.2f}  ·  "
            f"INT {latest['TransitionIntensity']:.1f}  ·  "
            f"W1 {latest['WassersteinNormalized']:.3f}"
        )
        ax_flow.text(
            0.985,
            0.965,
            status,
            transform=ax_flow.transAxes,
            ha="right",
            va="top",
            color=PALETTE["navy"],
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.45", "fc": status_color, "ec": "none", "alpha": 0.90},
            zorder=10,
        )

    finite_components = np.column_stack(
        [
            view["WassersteinZ"].to_numpy(dtype=float),
            view["QuantileTransportDirection"].to_numpy(dtype=float),
            view["TailAsymmetryZ"].to_numpy(dtype=float),
            view["TransportVelocityZ"].to_numpy(dtype=float),
            score_plot.to_numpy(dtype=float),
        ]
    )
    if np.isfinite(finite_components).any():
        max_abs = float(np.nanmax(np.abs(finite_components)))
    else:
        max_abs = 2.5
    y_limit = max(2.5, min(5.5, max_abs * 1.08))
    ax_flow.set_ylim(-y_limit, y_limit)
    ax_flow.legend(
        loc="upper left",
        ncol=5,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor=PALETTE["grid"],
        fontsize=7.5,
    )

    ax_flow.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax_flow.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_flow.xaxis.get_major_locator()))
    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_volume.get_xticklabels(), visible=False)
    margin = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    ax_flow.set_xlim(x[0] - margin, x[-1] + margin)
    fig.text(
        0.012,
        0.008,
        "Signal calculated at the close; any action occurs no earlier than the next bar. "
        "Experimental research only, not investment advice.",
        fontsize=7.5,
        color=PALETTE["text"],
    )
    fig.text(
        0.988,
        0.008,
        "Background: STABLE / TRANSITION  ·  △ RAW  ·  ▲ CONFIRMED",
        ha="right",
        fontsize=7.5,
        color=PALETTE["text"],
    )
    fig.subplots_adjust(left=0.045, right=0.955, top=0.955, bottom=0.055)

    if show:
        plt.show()
    return fig


# %% IN-MEMORY EXECUTION
def _apply_bar_status(
    result: pd.DataFrame,
    source: str,
    yfinance_last_bar_is_complete: bool,
) -> pd.DataFrame:
    """Mark the last Yahoo bar without making it an operational event or state."""
    marked = result.copy()
    marked["BarStatus"] = "CONFIRMED"
    if (
        source.strip().lower() == "yfinance"
        and not yfinance_last_bar_is_complete
        and not marked.empty
    ):
        last_index = marked.index[-1]
        marked.loc[last_index, "BarStatus"] = "PROVISIONAL"
        marked.loc[last_index, "RawSignalAtClose"] = ""
        marked.loc[last_index, "ConfirmedSignalAtClose"] = ""
        marked.loc[last_index, "SignalAtClose"] = ""
        marked.loc[
            last_index,
            [
                "RawBuyAtClose",
                "RawSellAtClose",
                "ConfirmedBuyAtClose",
                "ConfirmedSellAtClose",
                "BullSignal",
                "BearSignal",
                "ShiftSignal",
            ],
        ] = False
        marked.loc[last_index, "MetricsReady"] = False
        marked.loc[last_index, "MarketState"] = "PROVISIONAL"
        marked.loc[last_index, "DirectionalRegime"] = "NEUTRAL"
        marked.loc[last_index, "Regime"] = "PROVISIONAL"
        marked.loc[last_index, "RegimeChanged"] = False
        marked.loc[last_index, "StateReason"] = "LAST_BAR_NOT_CONFIRMED"
        marked.loc[last_index, "PendingSide"] = ""
        marked.loc[last_index, "PendingBarsRemaining"] = 0
        marked.loc[last_index, "ConfirmationLagBars"] = np.nan
        # ActionNextBar values come from the previous bar and remain unchanged;
        # they may be valid actions on the current bar.
    return marked


def main() -> pd.DataFrame:
    global FIGURE
    CONFIG.validate()
    data, label = load_data()
    result = calculate_probability_flow(data, CONFIG)
    result = _apply_bar_status(
        result,
        source=DATA_SOURCE,
        yfinance_last_bar_is_complete=YFINANCE_LAST_BAR_IS_COMPLETE,
    )

    FIGURE = plot_probability_flow(
        result,
        label=label,
        config=CONFIG,
        max_bars=MAX_BARS_IN_CHART,
        show=SHOW_FIGURE,
    )

    valid = result.dropna(
        subset=["QuantileTransportDirection", "TransitionIntensity"]
    )
    if valid.empty:
        warnings.warn("Calculation completed, but no post-warm-up values exist yet.")
    else:
        latest = valid.iloc[-1]
        print("\nProbability Flow Momentum — summary")
        print(f"Instrument: {label}")
        print(f"Latest bar: {valid.index[-1]}")
        print(f"State: {latest['MarketState']}")
        print(f"Directional regime: {latest['DirectionalRegime']}")
        print(
            "Quantile Transport Direction: "
            f"{latest['QuantileTransportDirection']:+.3f}"
        )
        print(f"Intensity: {latest['TransitionIntensity']:.2f}")
        print(
            "Normalized Wasserstein: "
            f"{latest['WassersteinNormalized']:.4f}"
        )
        if latest["RawSignalAtClose"]:
            print(f"RAW signal: {latest['RawSignalAtClose']}")
        if latest["ConfirmedSignalAtClose"]:
            print(f"CONFIRMED signal: {latest['ConfirmedSignalAtClose']}")
        if latest["BarStatus"] == "PROVISIONAL":
            print(
                "WARNING: the latest Yahoo bar is marked PROVISIONAL; "
                "no RAW/CONFIRMED signal is emitted on this row."
            )

    print(
        "No file saved: results remain in RESULTS and the chart in FIGURE/Plots."
    )
    return result


if __name__ == "__main__":
    RESULTS = main()
