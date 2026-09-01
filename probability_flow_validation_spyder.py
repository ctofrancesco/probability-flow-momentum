# -*- coding: utf-8 -*-
"""Separate walk-forward validation for Probability Flow Momentum.

This file imports `probability_flow_spyder.py` instead of duplicating the
engine. It compares three confirmation policies across multiple instruments:

STRICT
    Direction + Wasserstein + aligned Tail Asymmetry + Velocity.
TAIL_VETO
    Direction + Wasserstein + Velocity; the tail blocks only when strongly
    opposed.
SUPPORT_OR
    Direction + at least one of Wasserstein, aligned Tail Asymmetry, or Velocity.
    It is deliberately permissive and should be read as a stress test, not as a
    more reliable confirmation by definition.

Calibration uses only the training segment preceding each fold. OOS metrics
start at the bar following the signal, consistently with `ActionNextBar`. RAW
performance is reported as a benchmark and compared with the three CONFIRMED
policies. Points overlaid on chart bars show individual folds; they are not IID
confidence intervals because multi-bar outcomes may overlap in time.

The RAW/pending state machine is restarted at the beginning of each test. A
newly calibrated threshold is therefore never applied retroactively to a
candidate born in the previous fold. An outcome may end in the next fold,
because this affects only future measurement of an already frozen signal.

All outputs remain in Spyder's Variable Explorer and Plots pane. The program
does not save CSV files, images, or any other output.

Running in Spyder:
1. Keep this file in the same directory as probability_flow_spyder.py.
2. Edit the USER CONFIGURATION section if needed.
3. Press F5.
"""

from __future__ import annotations

# %% IMPORT
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# %% USER CONFIGURATION
DATA_SOURCE = "yfinance"  # "yfinance" or "demo"
TICKERS = ("SPY", "QQQ", "BTC-USD", "EURUSD=X")
YFINANCE_PERIOD = "max"
YFINANCE_INTERVAL = "1d"

# Conservative choice: drop the latest Yahoo bar because it may still be
# forming. On daily history this costs only one bar.
DROP_LAST_YFINANCE_BAR = True

# Expanding walk-forward: the initial training segment covers 55% of history;
# each subsequent fold includes all previously observed folds and tests the next.
INITIAL_TRAIN_RATIO = 0.55
NUM_WALKFORWARD_FOLDS = 4

# Parsimonious calibration: vary all evidence-threshold severities together.
# The tail veto stays fixed; multiplying it by the same scale would make the
# veto less strict exactly when the other filters become stricter.
# A small grid limits overfitting. 1.00 equals the engine defaults.
THRESHOLD_SCALES = (0.75, 1.00, 1.25)
CALIBRATION_HORIZON = 10
MIN_CALIBRATION_EVENTS = 12

FORWARD_HORIZONS = (1, 5, 10, 20)

# Use the same return cohort for every horizon: an event is included only when
# both the entry Open and the maximum-horizon Close exist. MAE/MFE use separate
# sample sizes when an intermediate High/Low is missing. Outcomes may cross a
# fold boundary; thresholds and signals remain frozen at OOS confirmation time.
COMMON_OUTCOME_COHORT = True

# The tail threshold used as a veto is deliberately much stronger than the
# directional-support threshold.
TAIL_VETO_Z = 0.75

# Total entry-plus-exit cost deducted from returns. Leave at 0.0 for a purely
# diagnostic reading of the indicator.
ROUND_TRIP_COST_BPS = 0.0

SHOW_FIGURES = True

# Diagnostic only: above this share, the state is flagged as nearly degenerate.
# This value neither modifies nor calibrates the engine.
STATE_DOMINANCE_WARNING_PCT = 0.90


@dataclass(frozen=True)
class ValidationConfig:
    """Validator parameters; does not modify engine configuration."""

    data_source: str = DATA_SOURCE
    tickers: tuple[str, ...] = TICKERS
    yfinance_period: str = YFINANCE_PERIOD
    yfinance_interval: str = YFINANCE_INTERVAL
    drop_last_yfinance_bar: bool = DROP_LAST_YFINANCE_BAR
    initial_train_ratio: float = INITIAL_TRAIN_RATIO
    n_folds: int = NUM_WALKFORWARD_FOLDS
    threshold_scales: tuple[float, ...] = THRESHOLD_SCALES
    calibration_horizon: int = CALIBRATION_HORIZON
    min_calibration_events: int = MIN_CALIBRATION_EVENTS
    horizons: tuple[int, ...] = FORWARD_HORIZONS
    common_outcome_cohort: bool = COMMON_OUTCOME_COHORT
    tail_veto_z: float = TAIL_VETO_Z
    round_trip_cost_bps: float = ROUND_TRIP_COST_BPS
    show_figures: bool = SHOW_FIGURES
    state_dominance_warning_pct: float = STATE_DOMINANCE_WARNING_PCT

    def validate(self) -> None:
        if self.data_source not in ("yfinance", "demo"):
            raise ValueError("data_source must be 'yfinance' or 'demo'.")
        if not self.tickers or any(not str(ticker).strip() for ticker in self.tickers):
            raise ValueError("tickers must contain at least one valid symbol.")
        if not 0.30 <= self.initial_train_ratio <= 0.85:
            raise ValueError("initial_train_ratio must be between 0.30 and 0.85.")
        if (
            isinstance(self.n_folds, bool)
            or not isinstance(self.n_folds, (int, np.integer))
            or self.n_folds < 1
        ):
            raise ValueError("n_folds must be at least 1.")
        if not self.threshold_scales:
            raise ValueError("threshold_scales cannot be empty.")
        if any(not np.isfinite(value) or value <= 0 for value in self.threshold_scales):
            raise ValueError("Every threshold scale must be positive and finite.")
        if (
            isinstance(self.calibration_horizon, bool)
            or not isinstance(self.calibration_horizon, (int, np.integer))
            or self.calibration_horizon < 1
        ):
            raise ValueError("calibration_horizon must be at least 1.")
        if (
            isinstance(self.min_calibration_events, bool)
            or not isinstance(self.min_calibration_events, (int, np.integer))
            or self.min_calibration_events < 2
        ):
            raise ValueError("min_calibration_events must be at least 2.")
        if not self.horizons or any(
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
            for horizon in self.horizons
        ):
            raise ValueError("Horizons must be positive integers.")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be sorted and contain no duplicates.")
        if not isinstance(self.common_outcome_cohort, (bool, np.bool_)):
            raise ValueError("common_outcome_cohort must be boolean.")
        if not np.isfinite(self.tail_veto_z) or self.tail_veto_z <= 0:
            raise ValueError("tail_veto_z must be positive and finite.")
        if self.tail_veto_z > BASE_INDICATOR_CONFIG.zscore_clip:
            raise ValueError(
                "tail_veto_z cannot exceed the engine's zscore_clip; "
                "the veto would become unreachable."
            )
        if not np.isfinite(self.round_trip_cost_bps) or self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be finite and non-negative.")
        if (
            not np.isfinite(self.state_dominance_warning_pct)
            or not 0.50 < self.state_dominance_warning_pct < 1.0
        ):
            raise ValueError(
                "state_dominance_warning_pct must be between 0.50 and 1.0."
            )


VALIDATION_CONFIG = ValidationConfig()
POLICIES = ("STRICT", "TAIL_VETO", "SUPPORT_OR")
PERFORMANCE_SERIES = ("RAW",) + POLICIES
SERIES_COLORS = {
    "RAW": "#7C8799",
    "STRICT": "#0B2D6B",
    "TAIL_VETO": "#D4A017",
    "SUPPORT_OR": "#2A9D8F",
}


@dataclass(frozen=True)
class ConfirmationThresholds:
    """Thresholds selected using only one fold's training segment."""

    scale: float
    direction: float
    wasserstein_z: float
    tail_z: float
    velocity_z: float
    tail_veto_z: float


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


# %% IMPORT FROZEN ENGINE
def _load_probability_flow_engine() -> Any:
    """Import the engine without running `main()` or writing bytecode."""
    engine_path = Path(__file__).resolve().with_name("probability_flow_spyder.py")
    if not engine_path.is_file():
        raise FileNotFoundError(
            "probability_flow_spyder.py must be in the same directory as the "
            "validator."
        )

    module_name = "_probability_flow_engine_for_validation"
    specification = importlib.util.spec_from_file_location(module_name, engine_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to import the engine from {engine_path}.")

    module = importlib.util.module_from_spec(specification)
    # dataclasses resolves annotations through sys.modules during import.
    sys.modules[module_name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


ENGINE = _load_probability_flow_engine()
BASE_INDICATOR_CONFIG = ENGINE.CONFIG


# %% MULTI-ASSET DATA LOADING
def _provenance_flag(
    raw: pd.DataFrame,
    normalized_columns: dict[str, Any],
    key: str,
) -> np.ndarray | None:
    """Read a boolean flag without converting the string ``False`` to True."""
    if key not in normalized_columns:
        return None
    values = raw[normalized_columns[key]]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.fillna(0.0).ne(0.0).to_numpy(dtype=bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "t", "yes", "y", "1"}).to_numpy(dtype=bool)


def _observed_ohlc_masks(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Separate observed Opens from High/Low paths usable for MAE/MFE.

    An observed Open makes the next-bar return executable; High and Low are
    used only for excursions. An already sanitized frame must preserve these
    flags explicitly, otherwise genuine OHLC data cannot be distinguished from
    values derived from Close alone.
    """
    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: empty dataset.")

    normalized_columns = {
        "".join(
            character
            for character in str(column).strip().lower()
            if character.isalnum()
        ): column
        for column in raw.columns
    }
    missing = [
        name
        for name in ("open", "high", "low", "close")
        if name not in normalized_columns
    ]
    if missing:
        raise ValueError(
            f"{ticker}: validation requires genuine OHLC data; missing {missing}. "
            "A close-only series can be used by the charting engine, but not for "
            "entries, returns, MAE, or MFE."
        )

    has_legacy_provenance = "observedohlc" in normalized_columns
    has_split_provenance = {
        "observedopen",
        "observedexcursion",
    }.issubset(normalized_columns)
    has_complete_provenance = has_legacy_provenance or has_split_provenance
    if "analysisclose" in normalized_columns and not has_complete_provenance:
        raise ValueError(
            f"{ticker}: the frame appears to be already sanitized (it contains "
            "AnalysisClose), but it does not preserve both ObservedOpen and "
            "ObservedExcursion flags (or legacy ObservedOHLC). Pass the original "
            "OHLC data or complete provenance; the validator does not assume "
            "synthesized values were observed."
        )

    ohlc = pd.DataFrame(
        {
            name.capitalize(): pd.to_numeric(
                raw[normalized_columns[name]], errors="coerce"
            )
            for name in ("open", "high", "low", "close")
        }
    )
    open_values = ohlc["Open"].to_numpy(dtype=float)
    high_values = ohlc["High"].to_numpy(dtype=float)
    low_values = ohlc["Low"].to_numpy(dtype=float)
    close_values = ohlc["Close"].to_numpy(dtype=float)

    open_observed = np.isfinite(open_values) & (open_values > 0.0)
    close_observed = np.isfinite(close_values) & (close_values > 0.0)
    high_observed = np.isfinite(high_values) & (high_values > 0.0)
    low_observed = np.isfinite(low_values) & (low_values > 0.0)

    # Small tolerance for provider rounding after auto_adjust.
    tolerance = 1e-10
    high_valid = high_values >= close_values * (1.0 - tolerance)
    low_valid = low_values <= close_values * (1.0 + tolerance)
    range_valid = high_values >= low_values * (1.0 - tolerance)
    # When Open exists, High/Low must also contain the opening price.
    high_valid &= (~open_observed) | (
        high_values >= open_values * (1.0 - tolerance)
    )
    low_valid &= (~open_observed) | (
        low_values <= open_values * (1.0 + tolerance)
    )
    excursion_observed = (
        close_observed
        & high_observed
        & low_observed
        & high_valid
        & low_valid
        & range_valid
    )

    legacy_provenance = _provenance_flag(
        raw, normalized_columns, "observedohlc"
    )
    open_provenance = _provenance_flag(
        raw, normalized_columns, "observedopen"
    )
    excursion_provenance = _provenance_flag(
        raw, normalized_columns, "observedexcursion"
    )
    if legacy_provenance is not None:
        open_observed &= legacy_provenance
        excursion_observed &= legacy_provenance
    if open_provenance is not None:
        open_observed &= open_provenance
    if excursion_provenance is not None:
        excursion_observed &= excursion_provenance

    date_key = next(
        (
            key
            for key in ("date", "datetime", "timestamp", "time")
            if key in normalized_columns
        ),
        None,
    )
    raw_dates = (
        raw[normalized_columns[date_key]] if date_key is not None else raw.index
    )
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce", utc=True)
    if bool(pd.isna(parsed_dates).any()):
        raise ValueError(f"{ticker}: unparseable dates are present.")

    observed_frame = pd.DataFrame(
        {
            "ObservedOpen": open_observed,
            "ObservedExcursion": excursion_observed,
        },
        index=pd.DatetimeIndex(parsed_dates).tz_convert(None),
        dtype=bool,
    )
    observed_frame = observed_frame[
        ~observed_frame.index.duplicated(keep="last")
    ].sort_index()
    observed_frame["ObservedOHLC"] = (
        observed_frame["ObservedOpen"] & observed_frame["ObservedExcursion"]
    )
    return observed_frame


def _sanitize_validation_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Sanitize prices while preserving OHLC provenance."""
    observed = _observed_ohlc_masks(raw, ticker)
    data = ENGINE.sanitize_ohlcv(raw)
    for column in ("ObservedOpen", "ObservedExcursion", "ObservedOHLC"):
        data[column] = observed[column].reindex(data.index).fillna(False).astype(bool)
    data.attrs["InvalidOpenBars"] = int((~data["ObservedOpen"]).sum())
    data.attrs["InvalidExcursionBars"] = int(
        (~data["ObservedExcursion"]).sum()
    )
    return data


def _download_yfinance(ticker: str, config: ValidationConfig) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "For multi-asset validation, install yfinance with: "
            "pip install yfinance"
        ) from exc

    history = yf.Ticker(ticker).history(
        period=config.yfinance_period,
        interval=config.yfinance_interval,
        auto_adjust=True,
        actions=False,
    )
    if history.empty:
        raise ValueError("Yahoo Finance returned no observations.")
    data = _sanitize_validation_ohlcv(history, ticker)
    if config.drop_last_yfinance_bar:
        if len(data) < 3:
            raise ValueError("Insufficient data after dropping the latest bar.")
        data = data.iloc[:-1].copy()
    return data


def load_validation_data(
    config: ValidationConfig = VALIDATION_CONFIG,
    supplied_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load assets; supplied_data enables CSV/in-memory tests without downloads."""
    config.validate()
    datasets: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    if supplied_data is not None:
        for ticker, frame in supplied_data.items():
            try:
                datasets[str(ticker)] = _sanitize_validation_ohlcv(
                    frame,
                    str(ticker),
                )
            except Exception as exc:  # one invalid asset does not invalidate others
                errors.append({"Ticker": str(ticker), "Error": str(exc)})
        return datasets, pd.DataFrame(errors, columns=["Ticker", "Error"])

    if config.data_source == "demo":
        for position, ticker in enumerate(config.tickers):
            demo = ENGINE.build_demo_data(
                n_bars=1400,
                seed=73 + 97 * position,
            )
            demo["ObservedOpen"] = True
            demo["ObservedExcursion"] = True
            demo["ObservedOHLC"] = True
            demo.attrs["InvalidOpenBars"] = 0
            demo.attrs["InvalidExcursionBars"] = 0
            datasets[ticker] = demo
        return datasets, pd.DataFrame(columns=["Ticker", "Error"])

    for ticker in config.tickers:
        try:
            datasets[ticker] = _download_yfinance(ticker, config)
        except Exception as exc:  # preserve results for all other assets
            errors.append({"Ticker": ticker, "Error": str(exc)})
    return datasets, pd.DataFrame(errors, columns=["Ticker", "Error"])


# %% CAUSAL CONFIRMATION POLICIES
def thresholds_from_scale(
    scale: float,
    tail_veto_z: float,
) -> ConfirmationThresholds:
    """Scale engine thresholds together without using test data."""
    return ConfirmationThresholds(
        scale=float(scale),
        direction=float(
            min(1.0, BASE_INDICATOR_CONFIG.confirmation_min_direction * scale)
        ),
        wasserstein_z=float(
            BASE_INDICATOR_CONFIG.confirmation_min_wasserstein_z * scale
        ),
        tail_z=float(BASE_INDICATOR_CONFIG.confirmation_min_tail_z * scale),
        velocity_z=float(BASE_INDICATOR_CONFIG.confirmation_min_velocity_z * scale),
        # The veto has the opposite orientation to evidence thresholds: a
        # higher absolute threshold blocks fewer events. It therefore stays
        # fixed ex ante.
        tail_veto_z=float(tail_veto_z),
    )


def _policy_filters_pass(
    row: pd.Series,
    side: str,
    policy: str,
    thresholds: ConfirmationThresholds,
) -> bool:
    direction = float(row["QuantileTransportDirection"])
    wasserstein_z = float(row["WassersteinZ"])
    tail_z = float(row["TailAsymmetryZ"])
    velocity_z = float(row["TransportVelocityZ"])

    sign = 1.0 if side == "BUY" else -1.0
    direction_gate = sign * direction >= thresholds.direction
    wasserstein_gate = wasserstein_z >= thresholds.wasserstein_z
    aligned_tail_gate = sign * tail_z >= thresholds.tail_z
    velocity_gate = velocity_z >= thresholds.velocity_z
    opposing_tail_veto = sign * tail_z <= -thresholds.tail_veto_z

    if policy == "STRICT":
        return bool(
            direction_gate
            and wasserstein_gate
            and aligned_tail_gate
            and velocity_gate
        )
    if policy == "TAIL_VETO":
        return bool(
            direction_gate
            and wasserstein_gate
            and velocity_gate
            and not opposing_tail_veto
        )
    if policy == "SUPPORT_OR":
        return bool(
            direction_gate
            and (wasserstein_gate or aligned_tail_gate or velocity_gate)
        )
    raise ValueError(f"Unknown policy: {policy!r}.")


def generate_policy_events(
    metrics: pd.DataFrame,
    policy: str,
    thresholds: ConfirmationThresholds,
    confirmation_window: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link each CONFIRMED event to a recent RAW event with a causal state machine.

    Confirmation is emitted on the bar where filters become true, never on a
    previous RAW bar. Lags 0..confirmation_window-1 are allowed.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    if confirmation_window is None:
        confirmation_window = BASE_INDICATOR_CONFIG.confirmation_window
    if confirmation_window < 1:
        raise ValueError("confirmation_window must be at least 1.")

    required_columns = (
        "MetricsReady",
        "RawSignalAtClose",
        "QuantileTransportDirection",
        "WassersteinZ",
        "TailAsymmetryZ",
        "TransportVelocityZ",
    )
    missing = [column for column in required_columns if column not in metrics]
    if missing:
        raise ValueError(f"Missing engine columns: {missing}.")

    signal_values = np.full(len(metrics), "", dtype=object)
    lag_values = np.full(len(metrics), np.nan, dtype=float)
    raw_record_values = np.full(len(metrics), np.nan, dtype=float)
    records: list[dict[str, Any]] = []
    pending_record: int | None = None

    for position, (timestamp, row) in enumerate(metrics.iterrows()):
        ready = bool(row["MetricsReady"])
        finite = bool(
            np.isfinite(row["QuantileTransportDirection"])
            and np.isfinite(row["WassersteinZ"])
            and np.isfinite(row["TailAsymmetryZ"])
            and np.isfinite(row["TransportVelocityZ"])
        )
        if not (ready and finite):
            if pending_record is not None:
                records[pending_record]["Status"] = "UNAVAILABLE"
            pending_record = None
            continue

        raw_signal = str(row["RawSignalAtClose"])
        if raw_signal in ("RAW_BUY", "RAW_SELL"):
            if pending_record is not None:
                records[pending_record]["Status"] = "REPLACED_BY_OPPOSITE_RAW"
            side = "BUY" if raw_signal == "RAW_BUY" else "SELL"
            records.append(
                {
                    "RawRecord": len(records),
                    "RawPosition": position,
                    "RawTime": timestamp,
                    "Side": side,
                    "ConfirmationPosition": np.nan,
                    "ConfirmationTime": pd.NaT,
                    "LagBars": np.nan,
                    "Status": "PENDING",
                }
            )
            pending_record = len(records) - 1

        if pending_record is None:
            continue

        record = records[pending_record]
        side = str(record["Side"])
        lag = position - int(record["RawPosition"])
        direction = float(row["QuantileTransportDirection"])
        side_still_valid = (side == "BUY" and direction >= 0.0) or (
            side == "SELL" and direction <= 0.0
        )

        if not side_still_valid:
            record["Status"] = "DIRECTION_REVERSED"
            pending_record = None
        elif _policy_filters_pass(row, side, policy, thresholds):
            record["ConfirmationPosition"] = position
            record["ConfirmationTime"] = timestamp
            record["LagBars"] = float(lag)
            record["Status"] = "CONFIRMED"
            signal_values[position] = side
            lag_values[position] = float(lag)
            raw_record_values[position] = float(record["RawRecord"])
            pending_record = None
        elif lag >= confirmation_window - 1:
            record["Status"] = "EXPIRED"
            pending_record = None

    if pending_record is not None:
        records[pending_record]["Status"] = "RIGHT_CENSORED"

    signal_frame = pd.DataFrame(
        {
            "PolicySignalAtClose": signal_values,
            "PolicyBuyAtClose": signal_values == "BUY",
            "PolicySellAtClose": signal_values == "SELL",
            "PolicyConfirmationLagBars": lag_values,
            "RawRecord": raw_record_values,
        },
        index=metrics.index,
    )
    event_frame = pd.DataFrame(
        records,
        columns=[
            "RawRecord",
            "RawPosition",
            "RawTime",
            "Side",
            "ConfirmationPosition",
            "ConfirmationTime",
            "LagBars",
            "Status",
        ],
    )
    return signal_frame, event_frame


def build_raw_events(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert engine RAW zero-crossings into OOS-evaluable events."""
    if "RawSignalAtClose" not in metrics:
        raise ValueError("RawSignalAtClose is missing from the engine output.")

    records: list[dict[str, Any]] = []
    for position, (timestamp, raw_signal) in enumerate(
        metrics["RawSignalAtClose"].items()
    ):
        if raw_signal not in ("RAW_BUY", "RAW_SELL"):
            continue
        side = "BUY" if raw_signal == "RAW_BUY" else "SELL"
        records.append(
            {
                "RawRecord": len(records),
                "RawPosition": position,
                "RawTime": timestamp,
                "Side": side,
                # The zero-crossing is itself the RAW signal: operational time
                # and position coincide, with entry at the next Open.
                "ConfirmationPosition": position,
                "ConfirmationTime": timestamp,
                "LagBars": 0.0,
                "Status": "RAW_SIGNAL",
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "RawRecord",
            "RawPosition",
            "RawTime",
            "Side",
            "ConfirmationPosition",
            "ConfirmationTime",
            "LagBars",
            "Status",
        ],
    )


def _offset_event_positions(events: pd.DataFrame, offset: int) -> pd.DataFrame:
    """Map events calculated on a slice back to global positions."""
    shifted = events.copy()
    if shifted.empty or offset == 0:
        return shifted
    shifted["RawPosition"] = shifted["RawPosition"] + int(offset)
    confirmation = pd.to_numeric(
        shifted["ConfirmationPosition"], errors="coerce"
    )
    shifted["ConfirmationPosition"] = confirmation.where(
        confirmation.isna(), confirmation + int(offset)
    )
    return shifted


# %% WALK-FORWARD AND OUTCOMES
def make_walk_forward_folds(
    n_rows: int,
    config: ValidationConfig = VALIDATION_CONFIG,
) -> list[WalkForwardFold]:
    config.validate()
    initial_train_end = int(np.floor(n_rows * config.initial_train_ratio))
    minimum_train = (
        BASE_INDICATOR_CONFIG.window
        + BASE_INDICATOR_CONFIG.transport_lag
        + BASE_INDICATOR_CONFIG.normalization_min_periods
        + max(config.horizons)
    )
    if initial_train_end < minimum_train:
        raise ValueError(
            f"Initial training segment is too short: {initial_train_end} bars; "
            f"at least {minimum_train} are required."
        )
    if n_rows - initial_train_end < config.n_folds * max(config.horizons):
        raise ValueError(
            "The OOS period is too short for the selected folds and horizons."
        )

    boundaries = np.linspace(
        initial_train_end,
        n_rows,
        config.n_folds + 1,
        dtype=int,
    )
    folds: list[WalkForwardFold] = []
    for fold_number in range(config.n_folds):
        test_start = int(boundaries[fold_number])
        test_end = int(boundaries[fold_number + 1])
        folds.append(
            WalkForwardFold(
                fold=fold_number + 1,
                train_start=0,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def _event_outcome_rows(
    ticker: str,
    policy: str,
    fold: int,
    data: pd.DataFrame,
    events: pd.DataFrame,
    start_position: int,
    end_position: int,
    horizons: tuple[int, ...],
    round_trip_cost_bps: float,
    threshold_scale: float,
    event_status: str = "CONFIRMED",
    outcome_end_position: int | None = None,
    require_all_horizons: bool = False,
) -> list[dict[str, Any]]:
    """Executable outcomes with entry at the next bar's Open.

    `end_position` bounds the bars on which a signal may originate.
    `outcome_end_position` bounds only observable future prices, so in OOS
    evaluation it may cross a fold boundary without changing the already
    frozen signal or thresholds.
    """
    if events.empty:
        return []
    eligible_events = events.loc[events["Status"].eq(event_status)].copy()
    eligible_events = eligible_events.loc[
        eligible_events["ConfirmationPosition"].between(
            start_position,
            end_position - 1,
            inclusive="both",
        )
    ]
    if eligible_events.empty:
        return []

    if outcome_end_position is None:
        outcome_end_position = end_position
    outcome_end_position = min(int(outcome_end_position), len(data))
    if outcome_end_position <= 0:
        return []

    # If a CSV contains unadjusted OHLC and Adj Close, engine metrics operate on
    # AnalysisClose. Apply the same factor to all OHLC values to avoid false
    # returns around splits or dividends. With Yahoo auto_adjust=True, this
    # factor is simply 1.
    close_unadjusted = data["Close"].to_numpy(dtype=float)
    analysis_close = data["AnalysisClose"].to_numpy(dtype=float)
    adjustment_factor = analysis_close / close_unadjusted
    open_values = data["Open"].to_numpy(dtype=float) * adjustment_factor
    high_values = data["High"].to_numpy(dtype=float) * adjustment_factor
    low_values = data["Low"].to_numpy(dtype=float) * adjustment_factor
    close_values = analysis_close
    legacy_observed = (
        data["ObservedOHLC"].to_numpy(dtype=bool)
        if "ObservedOHLC" in data
        else np.ones(len(data), dtype=bool)
    )
    observed_open = (
        data["ObservedOpen"].to_numpy(dtype=bool)
        if "ObservedOpen" in data
        else legacy_observed
    )
    observed_excursion = (
        data["ObservedExcursion"].to_numpy(dtype=bool)
        if "ObservedExcursion" in data
        else legacy_observed
    )
    total_cost = round_trip_cost_bps / 10_000.0
    rows: list[dict[str, Any]] = []
    maximum_horizon = max(horizons)

    for _, event in eligible_events.iterrows():
        confirmation_position = int(event["ConfirmationPosition"])
        entry_position = confirmation_position + 1
        if entry_position >= outcome_end_position:
            continue
        maximum_exit_position = confirmation_position + maximum_horizon
        if (
            require_all_horizons
            and maximum_exit_position >= outcome_end_position
        ):
            continue
        if not observed_open[entry_position]:
            continue
        entry_price = float(open_values[entry_position])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        for horizon in horizons:
            exit_position = confirmation_position + horizon
            if exit_position >= outcome_end_position:
                continue
            exit_price = float(close_values[exit_position])
            path_high_values = high_values[entry_position : exit_position + 1]
            path_low_values = low_values[entry_position : exit_position + 1]
            if not (np.isfinite(exit_price) and exit_price > 0):
                continue
            excursion_available = bool(
                observed_excursion[entry_position : exit_position + 1].all()
                and np.isfinite(path_high_values).all()
                and np.isfinite(path_low_values).all()
            )

            if event["Side"] == "BUY":
                gross_return = exit_price / entry_price - 1.0
            else:
                gross_return = (entry_price - exit_price) / entry_price

            if excursion_available:
                path_high = float(np.max(path_high_values))
                path_low = float(np.min(path_low_values))
                if event["Side"] == "BUY":
                    mfe = path_high / entry_price - 1.0
                    mae = path_low / entry_price - 1.0
                else:
                    mfe = (entry_price - path_low) / entry_price
                    mae = (entry_price - path_high) / entry_price
                # Explicit convention: MAE <= 0 and MFE >= 0.
                mae = min(0.0, float(mae))
                mfe = max(0.0, float(mfe))
            else:
                mae = np.nan
                mfe = np.nan

            rows.append(
                {
                    "Ticker": ticker,
                    "Policy": policy,
                    "Fold": fold,
                    "ThresholdScale": threshold_scale,
                    "RawTime": event["RawTime"],
                    "SignalTime": event["ConfirmationTime"],
                    "EntryTime": data.index[entry_position],
                    "ExitTime": data.index[exit_position],
                    "Side": event["Side"],
                    "LagBars": float(event["LagBars"]),
                    "Horizon": int(horizon),
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "ReturnGross": float(gross_return),
                    "ReturnNet": float(gross_return - total_cost),
                    "MAE": float(mae),
                    "MFE": float(mfe),
                    "ExcursionAvailable": excursion_available,
                }
            )
    return rows


def _calibration_score(
    data: pd.DataFrame,
    events: pd.DataFrame,
    train_end: int,
    config: ValidationConfig,
) -> tuple[float, int, float]:
    """Descriptive training t-statistic; no OOS observation is read."""
    rows = _event_outcome_rows(
        ticker="CALIBRATION",
        policy="CALIBRATION",
        fold=0,
        data=data,
        events=events,
        start_position=0,
        end_position=train_end,
        horizons=(config.calibration_horizon,),
        round_trip_cost_bps=config.round_trip_cost_bps,
        threshold_scale=np.nan,
    )
    if not rows:
        return np.nan, 0, np.nan
    returns = np.asarray([row["ReturnNet"] for row in rows], dtype=float)
    returns = returns[np.isfinite(returns)]
    event_count = int(len(returns))
    mean_return = float(np.mean(returns)) if event_count else np.nan
    if event_count < config.min_calibration_events:
        return np.nan, event_count, mean_return
    standard_deviation = float(np.std(returns, ddof=1))
    score = mean_return * np.sqrt(event_count) / max(standard_deviation, 1e-8)
    return float(score), event_count, mean_return


def _select_thresholds_on_training(
    ticker: str,
    policy: str,
    metrics_prefix: pd.DataFrame,
    fold: WalkForwardFold,
    config: ValidationConfig,
) -> tuple[
    ConfirmationThresholds,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    str,
]:
    candidate_rows: list[dict[str, Any]] = []
    candidate_outputs: dict[
        float,
        tuple[ConfirmationThresholds, pd.DataFrame, pd.DataFrame],
    ] = {}

    for scale in config.threshold_scales:
        thresholds = thresholds_from_scale(scale, config.tail_veto_z)
        signal_frame, events = generate_policy_events(
            metrics_prefix,
            policy,
            thresholds,
        )
        score, event_count, mean_return = _calibration_score(
            metrics_prefix,
            events,
            fold.train_end,
            config,
        )
        candidate_outputs[float(scale)] = (thresholds, signal_frame, events)
        candidate_rows.append(
            {
                "Ticker": ticker,
                "Policy": policy,
                "Fold": fold.fold,
                "TrainEndPosition": fold.train_end,
                "Scale": float(scale),
                "CalibrationScore": score,
                "CalibrationEvents": event_count,
                "CalibrationMeanReturn": mean_return,
                "Selected": False,
            }
        )

    valid_candidates = [
        row for row in candidate_rows if np.isfinite(row["CalibrationScore"])
    ]
    if valid_candidates:
        # For near ties, prefer more events and then the scale closest to 1.
        chosen = max(
            valid_candidates,
            key=lambda row: (
                row["CalibrationScore"],
                row["CalibrationEvents"],
                -abs(row["Scale"] - 1.0),
            ),
        )
        status = "CALIBRATED_ON_TRAIN"
    else:
        chosen = min(candidate_rows, key=lambda row: abs(row["Scale"] - 1.0))
        status = "FALLBACK_SCALE_1_INSUFFICIENT_EVENTS"

    selected_scale = float(chosen["Scale"])
    for row in candidate_rows:
        row["Selected"] = bool(np.isclose(row["Scale"], selected_scale))
        row["SelectionStatus"] = status
    thresholds, signal_frame, events = candidate_outputs[selected_scale]
    return thresholds, signal_frame, events, candidate_rows, status


def _state_row(
    ticker: str,
    fold: WalkForwardFold,
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    test = metrics.iloc[fold.test_start : fold.test_end]
    operational = test.loc[test["MarketState"].isin(("STABLE", "TRANSITION"))]
    stable_count = int(operational["MarketState"].eq("STABLE").sum())
    transition_count = int(operational["MarketState"].eq("TRANSITION").sum())
    operational_count = stable_count + transition_count
    return {
        "Ticker": ticker,
        "Fold": fold.fold,
        "TestStart": test.index[0],
        "TestEnd": test.index[-1],
        "OperationalBars": operational_count,
        "StableBars": stable_count,
        "TransitionBars": transition_count,
        "StablePct": stable_count / operational_count if operational_count else np.nan,
        "TransitionPct": (
            transition_count / operational_count if operational_count else np.nan
        ),
    }


def _state_gate_row(
    ticker: str,
    fold: WalkForwardFold,
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Diagnose which conditions make the STABLE state rare."""
    test = metrics.iloc[fold.test_start : fold.test_end]
    ready = test.loc[test["MetricsReady"]].copy()
    ready_count = int(len(ready))

    if ready_count:
        direction_quiet = ready["QuantileTransportDirection"].abs().le(
            BASE_INDICATOR_CONFIG.stable_max_direction
        )
        wasserstein_quiet = ready["WassersteinZ"].le(
            BASE_INDICATOR_CONFIG.stable_max_wasserstein_z
        )
        tail_quiet = ready["TailAsymmetryZ"].abs().le(
            BASE_INDICATOR_CONFIG.stable_max_tail_z
        )
        velocity_quiet = ready["TransportVelocityZ"].le(
            BASE_INDICATOR_CONFIG.stable_max_velocity_z
        )
        joint_quiet = (
            direction_quiet & wasserstein_quiet & tail_quiet & velocity_quiet
        )
        transition_trigger = (
            ready["RawSignalAtClose"].ne("")
            | ready["WassersteinZ"].ge(
                BASE_INDICATOR_CONFIG.confirmation_min_wasserstein_z
            )
            | ready["TransportVelocityZ"].ge(
                BASE_INDICATOR_CONFIG.confirmation_min_velocity_z
            )
            | (
                ready["QuantileTransportDirection"].abs().ge(
                    BASE_INDICATOR_CONFIG.confirmation_min_direction
                )
                & ready["TailAsymmetryZ"].abs().ge(
                    BASE_INDICATOR_CONFIG.confirmation_min_tail_z
                )
            )
        )
    else:
        empty = pd.Series(dtype=bool)
        direction_quiet = empty
        wasserstein_quiet = empty
        tail_quiet = empty
        velocity_quiet = empty
        joint_quiet = empty
        transition_trigger = empty

    return {
        "Ticker": ticker,
        "Fold": fold.fold,
        "ReadyBars": ready_count,
        "DirectionQuietBars": int(direction_quiet.sum()),
        "WassersteinQuietBars": int(wasserstein_quiet.sum()),
        "TailQuietBars": int(tail_quiet.sum()),
        "VelocityQuietBars": int(velocity_quiet.sum()),
        "JointQuietBars": int(joint_quiet.sum()),
        "TransitionTriggerBars": int(transition_trigger.sum()),
    }


def _fold_signal_row(
    ticker: str,
    policy: str,
    fold: WalkForwardFold,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    thresholds: ConfirmationThresholds,
    selection_status: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_in_test = events.loc[
        events["RawPosition"].between(
            fold.test_start,
            fold.test_end - 1,
            inclusive="both",
        )
    ].copy()
    last_eligible_raw_position = (
        fold.test_end - BASE_INDICATOR_CONFIG.confirmation_window
    )
    eligible_raw = raw_in_test.loc[
        raw_in_test["RawPosition"] <= last_eligible_raw_position
    ]
    confirmed_eligible = eligible_raw.loc[
        eligible_raw["Status"].eq("CONFIRMED")
        & (eligible_raw["ConfirmationPosition"] < fold.test_end)
    ]
    confirmed_in_test = events.loc[
        events["Status"].eq("CONFIRMED")
        & events["ConfirmationPosition"].between(
            fold.test_start,
            fold.test_end - 1,
            inclusive="both",
        )
    ].copy()

    row = {
        "Ticker": ticker,
        "Policy": policy,
        "Fold": fold.fold,
        "TrainStart": metrics.index[fold.train_start],
        "TrainEnd": metrics.index[fold.train_end - 1],
        "TestStart": metrics.index[fold.test_start],
        "TestEnd": metrics.index[fold.test_end - 1],
        "RawCount": int(len(raw_in_test)),
        "EligibleRawCount": int(len(eligible_raw)),
        "ConfirmedEligibleRawCount": int(len(confirmed_eligible)),
        "ConfirmedCount": int(len(confirmed_in_test)),
        "ConfirmRate": (
            len(confirmed_eligible) / len(eligible_raw)
            if len(eligible_raw)
            else np.nan
        ),
        "AvgConfirmationLag": (
            float(confirmed_in_test["LagBars"].mean())
            if len(confirmed_in_test)
            else np.nan
        ),
        "BuyCount": int(confirmed_in_test["Side"].eq("BUY").sum()),
        "SellCount": int(confirmed_in_test["Side"].eq("SELL").sum()),
        "ThresholdScale": thresholds.scale,
        "DirectionThreshold": thresholds.direction,
        "WassersteinZThreshold": thresholds.wasserstein_z,
        "TailZThreshold": thresholds.tail_z,
        "VelocityZThreshold": thresholds.velocity_z,
        "TailVetoZThreshold": thresholds.tail_veto_z,
        "SelectionStatus": selection_status,
    }
    if not confirmed_in_test.empty:
        confirmed_in_test.insert(0, "Ticker", ticker)
        confirmed_in_test.insert(1, "Policy", policy)
        confirmed_in_test.insert(2, "Fold", fold.fold)
        confirmed_in_test.insert(3, "ThresholdScale", thresholds.scale)
    return row, confirmed_in_test


# %% AGGREGATION AND CHARTS
def _aggregate_signal_summary(
    fold_results: pd.DataFrame,
    confirmed_events: pd.DataFrame,
) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (ticker, policy), group in fold_results.groupby(
        ["Ticker", "Policy"],
        sort=False,
    ):
        ordered_group = group.sort_values("Fold")
        selected_scales = ordered_group["ThresholdScale"].to_numpy(dtype=float)
        eligible = int(group["EligibleRawCount"].sum())
        confirmed_eligible = int(group["ConfirmedEligibleRawCount"].sum())
        event_subset = (
            confirmed_events.loc[
                confirmed_events["Ticker"].eq(ticker)
                & confirmed_events["Policy"].eq(policy)
            ]
            if not confirmed_events.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "Ticker": ticker,
                "Policy": policy,
                "RawCount": int(group["RawCount"].sum()),
                "EligibleRawCount": eligible,
                "ConfirmedEligibleRawCount": confirmed_eligible,
                "ConfirmedCount": int(group["ConfirmedCount"].sum()),
                "ConfirmRate": (
                    confirmed_eligible / eligible if eligible else np.nan
                ),
                "AvgConfirmationLag": (
                    float(event_subset["LagBars"].mean())
                    if not event_subset.empty
                    else np.nan
                ),
                "BuyCount": int(group["BuyCount"].sum()),
                "SellCount": int(group["SellCount"].sum()),
                "MeanSelectedScale": float(group["ThresholdScale"].mean()),
                "SelectedScalePath": " / ".join(
                    f"{scale:.2f}" for scale in selected_scales
                ),
                "SelectedScaleChanges": int(
                    np.count_nonzero(np.diff(selected_scales))
                ),
                "CalibrationFallbackFolds": int(
                    group["SelectionStatus"]
                    .eq("FALLBACK_SCALE_1_INSUFFICIENT_EVENTS")
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_state_summary(state_folds: pd.DataFrame) -> pd.DataFrame:
    if state_folds.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, group in state_folds.groupby("Ticker", sort=False):
        stable = int(group["StableBars"].sum())
        transition = int(group["TransitionBars"].sum())
        total = stable + transition
        rows.append(
            {
                "Ticker": ticker,
                "OperationalBars": total,
                "StableBars": stable,
                "TransitionBars": transition,
                "StablePct": stable / total if total else np.nan,
                "TransitionPct": transition / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_state_gate_summary(state_gate_folds: pd.DataFrame) -> pd.DataFrame:
    """Pooled frequencies of individual quiet-state gates, without retuning."""
    if state_gate_folds.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    count_columns = (
        "DirectionQuietBars",
        "WassersteinQuietBars",
        "TailQuietBars",
        "VelocityQuietBars",
        "JointQuietBars",
        "TransitionTriggerBars",
    )
    for ticker, group in state_gate_folds.groupby("Ticker", sort=False):
        ready = int(group["ReadyBars"].sum())
        totals = {column: int(group[column].sum()) for column in count_columns}
        rows.append(
            {
                "Ticker": ticker,
                "ReadyBars": ready,
                **totals,
                "DirectionQuietPct": (
                    totals["DirectionQuietBars"] / ready if ready else np.nan
                ),
                "WassersteinQuietPct": (
                    totals["WassersteinQuietBars"] / ready if ready else np.nan
                ),
                "TailQuietPct": totals["TailQuietBars"] / ready if ready else np.nan,
                "VelocityQuietPct": (
                    totals["VelocityQuietBars"] / ready if ready else np.nan
                ),
                "JointQuietPct": (
                    totals["JointQuietBars"] / ready if ready else np.nan
                ),
                "TransitionTriggerPct": (
                    totals["TransitionTriggerBars"] / ready if ready else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_performance(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Policy",
        "Horizon",
        "N",
        "NExcursion",
        "ExcursionCoverage",
        "MeanReturnNet",
        "MedianReturnNet",
        "WinRate",
        "MeanMAE",
        "MeanMFE",
    ]
    if outcomes.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (ticker, policy, horizon), group in outcomes.groupby(
        ["Ticker", "Policy", "Horizon"],
        sort=False,
    ):
        n_excursion = int(group["ExcursionAvailable"].astype(bool).sum())
        rows.append(
            {
                "Ticker": ticker,
                "Policy": policy,
                "Horizon": int(horizon),
                "N": int(len(group)),
                "NExcursion": n_excursion,
                "ExcursionCoverage": float(n_excursion / len(group)),
                "MeanReturnNet": float(group["ReturnNet"].mean()),
                "MedianReturnNet": float(group["ReturnNet"].median()),
                "WinRate": float(group["ReturnNet"].gt(0.0).mean()),
                "MeanMAE": float(group["MAE"].mean()),
                "MeanMFE": float(group["MFE"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _aggregate_fold_performance(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Observed dispersion across folds without assuming IID events."""
    columns = [
        "Ticker",
        "Policy",
        "Fold",
        "Horizon",
        "N",
        "NExcursion",
        "ExcursionCoverage",
        "MeanReturnNet",
        "MeanMAE",
        "MeanMFE",
        "WinRate",
    ]
    if outcomes.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (ticker, policy, fold, horizon), group in outcomes.groupby(
        ["Ticker", "Policy", "Fold", "Horizon"],
        sort=False,
    ):
        n_excursion = int(group["ExcursionAvailable"].astype(bool).sum())
        rows.append(
            {
                "Ticker": ticker,
                "Policy": policy,
                "Fold": int(fold),
                "Horizon": int(horizon),
                "N": int(len(group)),
                "NExcursion": n_excursion,
                "ExcursionCoverage": float(n_excursion / len(group)),
                "MeanReturnNet": float(group["ReturnNet"].mean()),
                "MeanMAE": float(group["MAE"].mean()),
                "MeanMFE": float(group["MFE"].mean()),
                "WinRate": float(group["ReturnNet"].gt(0.0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _aggregate_performance_by_side(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Separate BUY and SELL so pooled results do not hide asymmetries."""
    columns = [
        "Ticker",
        "Policy",
        "Side",
        "Horizon",
        "N",
        "NExcursion",
        "ExcursionCoverage",
        "MeanReturnNet",
        "MedianReturnNet",
        "WinRate",
        "MeanMAE",
        "MeanMFE",
    ]
    if outcomes.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (ticker, policy, side, horizon), group in outcomes.groupby(
        ["Ticker", "Policy", "Side", "Horizon"],
        sort=False,
    ):
        n_excursion = int(group["ExcursionAvailable"].astype(bool).sum())
        rows.append(
            {
                "Ticker": ticker,
                "Policy": policy,
                "Side": side,
                "Horizon": int(horizon),
                "N": int(len(group)),
                "NExcursion": n_excursion,
                "ExcursionCoverage": float(n_excursion / len(group)),
                "MeanReturnNet": float(group["ReturnNet"].mean()),
                "MedianReturnNet": float(group["ReturnNet"].median()),
                "WinRate": float(group["ReturnNet"].gt(0.0).mean()),
                "MeanMAE": float(group["MAE"].mean()),
                "MeanMFE": float(group["MFE"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _plot_grouped_bars(
    ax: plt.Axes,
    table: pd.DataFrame,
    value: str,
    title: str,
    ylabel: str,
    percent: bool = False,
    column_order: tuple[str, ...] = POLICIES,
    ticker_order: list[str] | None = None,
    fold_table: pd.DataFrame | None = None,
    annotation_mode: str | None = None,
    zero_line: bool = False,
) -> None:
    if table.empty or value not in table:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()
        return
    pivot = table.pivot(index="Ticker", columns="Policy", values=value)
    if ticker_order is not None:
        pivot = pivot.reindex(index=[ticker for ticker in ticker_order if ticker in pivot.index])
    pivot = pivot.reindex(columns=[name for name in column_order if name in pivot.columns])
    pivot.plot(
        kind="bar",
        ax=ax,
        color=[SERIES_COLORS.get(column, "#64748B") for column in pivot.columns],
        width=0.78,
        legend=False,
    )

    # Bars are pooled estimates. Points show dispersion across individual folds
    # without pretending that overlapping signals are independent.
    if fold_table is not None and not fold_table.empty and value in fold_table:
        for column_position, policy in enumerate(pivot.columns):
            if column_position >= len(ax.containers):
                continue
            patches = ax.containers[column_position].patches
            for row_position, ticker in enumerate(pivot.index):
                if row_position >= len(patches):
                    continue
                values = fold_table.loc[
                    fold_table["Ticker"].eq(ticker)
                    & fold_table["Policy"].eq(policy),
                    value,
                ].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                center = patches[row_position].get_x() + patches[row_position].get_width() / 2
                jitter = (
                    np.linspace(-0.025, 0.025, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                ax.scatter(
                    center + jitter,
                    values,
                    s=18,
                    facecolors="#FFFFFF",
                    edgecolors=SERIES_COLORS.get(policy, "#334155"),
                    linewidths=0.9,
                    zorder=4,
                )

    # Display sample sizes directly on bars so a return based on few events is
    # not mistaken for a more stable estimate.
    if annotation_mode is not None:
        for column_position, policy in enumerate(pivot.columns):
            if column_position >= len(ax.containers):
                continue
            patches = ax.containers[column_position].patches
            for row_position, ticker in enumerate(pivot.index):
                if row_position >= len(patches):
                    continue
                row = table.loc[
                    table["Ticker"].eq(ticker) & table["Policy"].eq(policy)
                ]
                if row.empty:
                    continue
                if annotation_mode == "conversion":
                    label = (
                        f"{int(row.iloc[0]['ConfirmedEligibleRawCount'])}/"
                        f"{int(row.iloc[0]['EligibleRawCount'])}"
                    )
                elif annotation_mode == "confirmed_n":
                    label = f"n={int(row.iloc[0]['ConfirmedCount'])}"
                elif annotation_mode == "outcome_n":
                    label = f"n={int(row.iloc[0]['N'])}"
                elif annotation_mode == "excursion_n":
                    # Avoid repeating the return panel's sample size. Report it
                    # only when MAE/MFE use a subset because High/Low are incomplete.
                    if int(row.iloc[0]["NExcursion"]) == int(row.iloc[0]["N"]):
                        continue
                    label = f"nMFE={int(row.iloc[0]['NExcursion'])}"
                else:
                    raise ValueError(f"Unknown annotation_mode: {annotation_mode}")
                patch = patches[row_position]
                height = float(patch.get_height())
                if not np.isfinite(height):
                    continue
                anchor = height
                if fold_table is not None and not fold_table.empty and value in fold_table:
                    fold_values = fold_table.loc[
                        fold_table["Ticker"].eq(ticker)
                        & fold_table["Policy"].eq(policy),
                        value,
                    ].dropna().to_numpy(dtype=float)
                    if len(fold_values):
                        anchor = max(anchor, float(np.max(fold_values)))
                ax.annotate(
                    label,
                    (patch.get_x() + patch.get_width() / 2, anchor),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#334155",
                )

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0.0, color="#26364F", linewidth=1.1, zorder=2)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def plot_validation_summary(
    signal_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    horizon: int,
    fold_results: pd.DataFrame | None = None,
    fold_performance: pd.DataFrame | None = None,
    state_folds: pd.DataFrame | None = None,
    round_trip_cost_bps: float = 0.0,
    show: bool = True,
) -> list[plt.Figure]:
    """Compare instruments and policies directly without writing files."""
    figures: list[plt.Figure] = []
    ticker_order = (
        signal_summary["Ticker"].drop_duplicates().tolist()
        if not signal_summary.empty
        else state_summary["Ticker"].drop_duplicates().tolist()
    )
    performance_horizon = performance_summary.loc[
        performance_summary["Horizon"].eq(horizon)
    ]
    fold_performance_horizon = (
        fold_performance.loc[fold_performance["Horizon"].eq(horizon)]
        if fold_performance is not None and not fold_performance.empty
        else None
    )

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    figure.suptitle(
        "Probability Flow Momentum — OOS walk-forward validation",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.963,
        "4 expanding folds · bars = pooled OOS aggregate · points = individual folds",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    _plot_grouped_bars(
        axes[0, 0],
        signal_summary,
        "ConfirmRate",
        "Eligible RAW → CONFIRMED",
        "Confirmation rate",
        percent=True,
        column_order=POLICIES,
        ticker_order=ticker_order,
        fold_table=fold_results,
        annotation_mode="conversion",
    )
    axes[0, 0].set_ylim(0.0, 1.06)
    axes[0, 0].set_yticks(np.linspace(0.0, 1.0, 6))
    _plot_grouped_bars(
        axes[0, 1],
        signal_summary,
        "AvgConfirmationLag",
        "Average RAW → CONFIRMED lag — confirmed only",
        "Bars",
        column_order=POLICIES,
        ticker_order=ticker_order,
        fold_table=fold_results,
        annotation_mode="confirmed_n",
    )
    _plot_grouped_bars(
        axes[1, 0],
        performance_horizon,
        "MeanReturnNet",
        f"Average directional return at {horizon} bars",
        "Return",
        percent=True,
        column_order=PERFORMANCE_SERIES,
        ticker_order=ticker_order,
        fold_table=fold_performance_horizon,
        annotation_mode="outcome_n",
        zero_line=True,
    )
    _plot_grouped_bars(
        axes[1, 1],
        performance_horizon,
        "MeanMFE",
        f"Average directional MFE at {horizon} bars — unrealized",
        "MFE",
        percent=True,
        column_order=PERFORMANCE_SERIES,
        ticker_order=ticker_order,
        fold_table=fold_performance_horizon,
        annotation_mode="excursion_n",
    )
    legend_handles = [
        Patch(
            facecolor=SERIES_COLORS[name],
            label=("RAW benchmark — performance only" if name == "RAW" else name),
        )
        for name in PERFORMANCE_SERIES
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
    )
    figure.text(
        0.01,
        0.012,
        (
            "BUY and SELL pooled with directional sign · entry at the next Open · "
            f"round-trip costs: {round_trip_cost_bps:g} bps · "
            "n = returns; nMFE appears only when complete OHLC paths are fewer · "
            "fold points, not IID confidence intervals; a point may be absent when "
            "a fold contains no events, or for MFE when it has no complete OHLC path"
        ),
        fontsize=8,
        color="#475569",
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.91))
    figures.append(figure)

    state_figure, state_axis = plt.subplots(figsize=(11, 5.5))
    if state_summary.empty:
        state_axis.text(0.5, 0.5, "Insufficient state data", ha="center")
        state_axis.set_axis_off()
    else:
        state_plot = state_summary.set_index("Ticker")[["StablePct", "TransitionPct"]]
        state_plot = state_plot.reindex(
            [ticker for ticker in ticker_order if ticker in state_plot.index]
        )
        state_plot.plot(
            kind="bar",
            stacked=True,
            ax=state_axis,
            color=["#DCE6F0", "#F2C94C"],
            width=0.68,
        )
        state_axis.set_title(
            "OOS state occupancy — base engine, pooled by bars across folds",
            loc="left",
            fontweight="bold",
        )
        state_axis.set_xlabel("")
        state_axis.set_ylabel("Percentage of operational bars")
        state_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        state_axis.tick_params(axis="x", rotation=0)
        state_axis.grid(axis="y", alpha=0.25)
        state_axis.set_axisbelow(True)
        state_axis.legend(
            ["STABLE", "TRANSITION"],
            title="",
            loc="upper right",
            bbox_to_anchor=(1.0, -0.08),
            ncol=2,
            frameon=False,
        )
        state_axis.set_ylim(0.0, 1.08)

        for position, ticker in enumerate(state_plot.index):
            stable_value = float(state_plot.loc[ticker, "StablePct"])
            transition_value = float(state_plot.loc[ticker, "TransitionPct"])
            if np.isfinite(stable_value):
                state_axis.text(
                    position,
                    max(stable_value / 2.0, 0.012),
                    f"{stable_value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#26364F",
                )
            if np.isfinite(transition_value):
                state_axis.text(
                    position,
                    stable_value + transition_value / 2.0,
                    f"{transition_value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#26364F",
                )
            n_bars = int(
                state_summary.loc[
                    state_summary["Ticker"].eq(ticker), "OperationalBars"
                ].iloc[0]
            )
            state_axis.text(
                position,
                1.015,
                f"n={n_bars}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#475569",
            )

            if state_folds is not None and not state_folds.empty:
                fold_values = state_folds.loc[
                    state_folds["Ticker"].eq(ticker), "StablePct"
                ].dropna().to_numpy(dtype=float)
                if len(fold_values):
                    jitter = (
                        np.linspace(-0.045, 0.045, len(fold_values))
                        if len(fold_values) > 1
                        else np.array([0.0])
                    )
                    state_axis.scatter(
                        position + jitter,
                        fold_values,
                        s=22,
                        facecolors="#FFFFFF",
                        edgecolors="#0B2D6B",
                        linewidths=0.9,
                        zorder=4,
                    )
        state_figure.text(
            0.01,
            0.012,
            "Points = each fold's STABLE share; TRANSITION is the complement.",
            fontsize=8,
            color="#475569",
        )
    state_figure.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    figures.append(state_figure)

    if show:
        plt.show()
    return figures


# %% COMPLETE IN-MEMORY EXECUTION
def run_validation(
    config: ValidationConfig = VALIDATION_CONFIG,
    supplied_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run download/calculation/calibration/OOS and return RAM objects only."""
    config.validate()
    datasets, load_errors = load_validation_data(config, supplied_data)
    if not datasets:
        error_text = (
            load_errors.to_string(index=False)
            if not load_errors.empty
            else "No dataset is available."
        )
        raise RuntimeError(f"Validation did not start:\n{error_text}")

    validation_data: dict[str, pd.DataFrame] = {}
    sample_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    state_gate_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    confirmed_event_frames: list[pd.DataFrame] = []
    raw_event_frames: list[pd.DataFrame] = []
    selected_signal_data: dict[tuple[str, str, int], pd.DataFrame] = {}
    runtime_errors: list[dict[str, str]] = []

    print("\nProbability Flow Momentum — Walk-Forward Validation")
    print("Engine imported, not duplicated. No file will be saved.\n")
    source_label = "SUPPLIED_MEMORY" if supplied_data is not None else config.data_source
    print(f"Data source: {source_label.upper()}")
    if config.data_source == "demo" and supplied_data is None:
        print("WARNING: DEMO uses synthetic series, not real market quotes.")
    print()

    for ticker, raw_data in datasets.items():
        print(f"[{ticker}] calculating causal metrics...")
        try:
            metrics = ENGINE.calculate_probability_flow(
                raw_data,
                BASE_INDICATOR_CONFIG,
            )
            for column in (
                "ObservedOpen",
                "ObservedExcursion",
                "ObservedOHLC",
            ):
                metrics[column] = (
                    raw_data[column]
                    .reindex(metrics.index)
                    .fillna(False)
                    .astype(bool)
                )
            folds = make_walk_forward_folds(len(metrics), config)
            raw_events = build_raw_events(metrics)
        except Exception as exc:
            runtime_errors.append({"Ticker": ticker, "Error": str(exc)})
            print(f"  excluded: {exc}")
            continue

        validation_data[ticker] = metrics
        sample_rows.append(
            {
                "Ticker": ticker,
                "FirstDate": metrics.index[0],
                "LastDate": metrics.index[-1],
                "TotalBars": int(len(metrics)),
                "OOSStart": metrics.index[folds[0].test_start],
                "OOSEnd": metrics.index[folds[-1].test_end - 1],
                "OOSBars": int(sum(fold.test_end - fold.test_start for fold in folds)),
                "InvalidOpenBars": int((~metrics["ObservedOpen"]).sum()),
                "InvalidOOSOpenBars": int(
                    (~metrics["ObservedOpen"].iloc[folds[0].test_start :]).sum()
                ),
                "InvalidExcursionBars": int(
                    (~metrics["ObservedExcursion"]).sum()
                ),
                "InvalidOOSExcursionBars": int(
                    (
                        ~metrics["ObservedExcursion"].iloc[
                            folds[0].test_start :
                        ]
                    ).sum()
                ),
            }
        )
        for fold in folds:
            raw_in_fold = raw_events.loc[
                raw_events["RawPosition"].between(
                    fold.test_start,
                    fold.test_end - 1,
                    inclusive="both",
                )
            ].copy()
            if not raw_in_fold.empty:
                raw_in_fold.insert(0, "Ticker", ticker)
                raw_in_fold.insert(1, "Fold", fold.fold)
                raw_event_frames.append(raw_in_fold)
            state_rows.append(_state_row(ticker, fold, metrics))
            state_gate_rows.append(_state_gate_row(ticker, fold, metrics))
            # Selection sees training data only. The pending state machine is
            # then initialized from scratch on the test segment: a new threshold
            # is not applied retroactively to candidates born in the prior fold.
            training_metrics = metrics.iloc[: fold.train_end].copy()
            test_metrics = metrics.iloc[fold.test_start : fold.test_end].copy()

            # RAW benchmark: same executable timing as confirmations, but no
            # additional filter and zero lag.
            outcome_rows.extend(
                _event_outcome_rows(
                    ticker=ticker,
                    policy="RAW",
                    fold=fold.fold,
                    data=metrics,
                    events=raw_events,
                    start_position=fold.test_start,
                    end_position=fold.test_end,
                    horizons=config.horizons,
                    round_trip_cost_bps=config.round_trip_cost_bps,
                    threshold_scale=np.nan,
                    event_status="RAW_SIGNAL",
                    outcome_end_position=len(metrics),
                    require_all_horizons=config.common_outcome_cohort,
                )
            )

            for policy in POLICIES:
                (
                    thresholds,
                    _training_signal_frame,
                    _training_events,
                    candidates,
                    selection_status,
                ) = _select_thresholds_on_training(
                    ticker,
                    policy,
                    training_metrics,
                    fold,
                    config,
                )
                calibration_rows.extend(candidates)
                signal_frame, events_local = generate_policy_events(
                    test_metrics,
                    policy,
                    thresholds,
                )
                events = _offset_event_positions(events_local, fold.test_start)
                fold_row, confirmed_in_test = _fold_signal_row(
                    ticker,
                    policy,
                    fold,
                    metrics,
                    events,
                    thresholds,
                    selection_status,
                )
                fold_rows.append(fold_row)
                if not confirmed_in_test.empty:
                    confirmed_event_frames.append(confirmed_in_test)

                outcome_rows.extend(
                    _event_outcome_rows(
                        ticker=ticker,
                        policy=policy,
                        fold=fold.fold,
                        data=metrics,
                        events=events,
                        start_position=fold.test_start,
                        end_position=fold.test_end,
                        horizons=config.horizons,
                        round_trip_cost_bps=config.round_trip_cost_bps,
                        threshold_scale=thresholds.scale,
                        outcome_end_position=len(metrics),
                        require_all_horizons=config.common_outcome_cohort,
                    )
                )
                signal_columns = [
                    "PolicySignalAtClose",
                    "PolicyBuyAtClose",
                    "PolicySellAtClose",
                    "PolicyConfirmationLagBars",
                    "RawRecord",
                ]
                selected_signal_data[(ticker, policy, fold.fold)] = signal_frame[
                    signal_columns
                ].copy()

        print(f"  completed {len(folds)} expanding OOS folds.")

    fold_results = pd.DataFrame(fold_rows)
    sample_summary = pd.DataFrame(sample_rows)
    state_folds = pd.DataFrame(state_rows)
    state_gate_folds = pd.DataFrame(state_gate_rows)
    calibration_results = pd.DataFrame(calibration_rows)
    forward_returns = pd.DataFrame(outcome_rows)
    raw_events_all = (
        pd.concat(raw_event_frames, ignore_index=True)
        if raw_event_frames
        else pd.DataFrame()
    )
    confirmed_events = (
        pd.concat(confirmed_event_frames, ignore_index=True)
        if confirmed_event_frames
        else pd.DataFrame()
    )
    signal_summary = _aggregate_signal_summary(fold_results, confirmed_events)
    state_summary = _aggregate_state_summary(state_folds)
    state_gate_summary = _aggregate_state_gate_summary(state_gate_folds)
    if not state_summary.empty:
        dominant = state_summary[["StablePct", "TransitionPct"]].max(axis=1)
        state_available = state_summary[["StablePct", "TransitionPct"]].notna().all(
            axis=1
        )
        state_summary["StateDiagnostic"] = np.select(
            [
                ~state_available,
                dominant >= config.state_dominance_warning_pct,
            ],
            ["INSUFFICIENT_DATA", "NEAR_DEGENERATE"],
            default="OK",
        )
    performance_summary = _aggregate_performance(forward_returns)
    fold_performance = _aggregate_fold_performance(forward_returns)
    performance_by_side = _aggregate_performance_by_side(forward_returns)
    errors = pd.concat(
        [load_errors, pd.DataFrame(runtime_errors, columns=["Ticker", "Error"])],
        ignore_index=True,
    )

    figures = plot_validation_summary(
        signal_summary,
        performance_summary,
        state_summary,
        horizon=max(config.horizons),
        fold_results=fold_results,
        fold_performance=fold_performance,
        state_folds=state_folds,
        round_trip_cost_bps=config.round_trip_cost_bps,
        show=config.show_figures,
    )

    if not signal_summary.empty:
        display_columns = [
            "Ticker",
            "Policy",
            "RawCount",
            "EligibleRawCount",
            "ConfirmedEligibleRawCount",
            "ConfirmedCount",
            "ConfirmRate",
            "AvgConfirmationLag",
            "MeanSelectedScale",
            "SelectedScalePath",
            "SelectedScaleChanges",
            "CalibrationFallbackFolds",
        ]
        print("\nOOS signal summary")
        print(
            "ConfirmRate = ConfirmedEligibleRawCount / EligibleRawCount; "
            "censored terminal RAW events are excluded from the rate."
        )
        print(signal_summary[display_columns].to_string(index=False))
    if not sample_summary.empty:
        print("\nEffective periods (native calendars may differ)")
        print(sample_summary.to_string(index=False))
        if (
            sample_summary["InvalidOOSOpenBars"].gt(0).any()
            or sample_summary["InvalidOOSExcursionBars"].gt(0).any()
        ):
            print(
                "OHLC quality: an unobserved Open excludes only the associated "
                "entry; incomplete High/Low data preserve the return but not "
                "MAE/MFE. No synthesized OHLC value is treated as observed."
            )
    if not state_summary.empty:
        print("\nOOS state occupancy (base configuration)")
        print(
            state_summary[
                [
                    "Ticker",
                    "OperationalBars",
                    "StablePct",
                    "TransitionPct",
                    "StateDiagnostic",
                ]
            ].to_string(index=False)
        )
    if not state_gate_summary.empty:
        print("\nOOS STABLE-gate diagnostics (no recalibration)")
        print(
            state_gate_summary[
                [
                    "Ticker",
                    "DirectionQuietPct",
                    "WassersteinQuietPct",
                    "TailQuietPct",
                    "VelocityQuietPct",
                    "JointQuietPct",
                    "TransitionTriggerPct",
                ]
            ].to_string(index=False)
        )
    performance_at_plot_horizon = performance_summary.loc[
        performance_summary["Horizon"].eq(max(config.horizons))
    ]
    if not performance_at_plot_horizon.empty:
        print(f"\nOOS performance at {max(config.horizons)} bars — RAW vs CONFIRMED")
        print(
            performance_at_plot_horizon[
                [
                    "Ticker",
                    "Policy",
                    "N",
                    "NExcursion",
                    "ExcursionCoverage",
                    "MeanReturnNet",
                    "MedianReturnNet",
                    "WinRate",
                    "MeanMAE",
                    "MeanMFE",
                ]
            ].to_string(index=False)
        )
    if not state_summary.empty and state_summary["StateDiagnostic"].eq(
        "NEAR_DEGENERATE"
    ).any():
        affected = ", ".join(
            state_summary.loc[
                state_summary["StateDiagnostic"].eq("NEAR_DEGENERATE"), "Ticker"
            ].astype(str)
        )
        print(
            "\nSTATE WARNING: one class exceeds "
            f"{config.state_dominance_warning_pct:.0%} for {affected}. "
            "STABLE/TRANSITION is not sufficiently discriminative; inspect gate "
            "diagnostics before retuning the engine. Any retuning derived from "
            "these OOS results then requires a new untouched holdout or a nested "
            "walk-forward design."
        )
    if not errors.empty:
        print("\nUnprocessed assets")
        print(errors.to_string(index=False))
    print("\nNo file saved: all outputs remain in Spyder.")

    return {
        "VALIDATION_DATA": validation_data,
        "SAMPLE_SUMMARY": sample_summary,
        "FOLD_RESULTS": fold_results,
        "SIGNAL_SUMMARY": signal_summary,
        "STATE_FOLDS": state_folds,
        "STATE_SUMMARY": state_summary,
        "STATE_GATE_FOLDS": state_gate_folds,
        "STATE_GATE_SUMMARY": state_gate_summary,
        "CALIBRATION_RESULTS": calibration_results,
        "RAW_EVENTS": raw_events_all,
        "CONFIRMED_EVENTS": confirmed_events,
        "FORWARD_RETURNS": forward_returns,
        "PERFORMANCE_SUMMARY": performance_summary,
        "FOLD_PERFORMANCE": fold_performance,
        "PERFORMANCE_BY_SIDE": performance_by_side,
        "POLICY_SIGNALS": selected_signal_data,
        "VALIDATION_FIGURES": figures,
        "VALIDATION_ERRORS": errors,
        "DATA_SOURCE": source_label,
    }


if __name__ == "__main__":
    VALIDATION_OUTPUTS = run_validation()
    VALIDATION_DATA = VALIDATION_OUTPUTS["VALIDATION_DATA"]
    SAMPLE_SUMMARY = VALIDATION_OUTPUTS["SAMPLE_SUMMARY"]
    FOLD_RESULTS = VALIDATION_OUTPUTS["FOLD_RESULTS"]
    SIGNAL_SUMMARY = VALIDATION_OUTPUTS["SIGNAL_SUMMARY"]
    STATE_FOLDS = VALIDATION_OUTPUTS["STATE_FOLDS"]
    STATE_SUMMARY = VALIDATION_OUTPUTS["STATE_SUMMARY"]
    STATE_GATE_FOLDS = VALIDATION_OUTPUTS["STATE_GATE_FOLDS"]
    STATE_GATE_SUMMARY = VALIDATION_OUTPUTS["STATE_GATE_SUMMARY"]
    CALIBRATION_RESULTS = VALIDATION_OUTPUTS["CALIBRATION_RESULTS"]
    RAW_EVENTS = VALIDATION_OUTPUTS["RAW_EVENTS"]
    CONFIRMED_EVENTS = VALIDATION_OUTPUTS["CONFIRMED_EVENTS"]
    FORWARD_RETURNS = VALIDATION_OUTPUTS["FORWARD_RETURNS"]
    PERFORMANCE_SUMMARY = VALIDATION_OUTPUTS["PERFORMANCE_SUMMARY"]
    FOLD_PERFORMANCE = VALIDATION_OUTPUTS["FOLD_PERFORMANCE"]
    PERFORMANCE_BY_SIDE = VALIDATION_OUTPUTS["PERFORMANCE_BY_SIDE"]
    POLICY_SIGNALS = VALIDATION_OUTPUTS["POLICY_SIGNALS"]
    VALIDATION_FIGURES = VALIDATION_OUTPUTS["VALIDATION_FIGURES"]
    VALIDATION_ERRORS = VALIDATION_OUTPUTS["VALIDATION_ERRORS"]
