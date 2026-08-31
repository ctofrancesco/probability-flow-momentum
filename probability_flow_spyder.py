#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:02:49 2026

@author: francescocapuzzimato


Probability Flow Momentum — modello quantitativo indipendente per Spyder.

Questo script implementa in modo trasparente i concetti pubblicamente descritti
per indicatori basati su probability flow: distanza di Wasserstein, direzione
del trasporto tra quantili, asimmetria delle code e velocità distributiva.

La Quantile Transport Direction (QTD) è la massa dei quantili spostati verso
l'alto meno quella spostata verso il basso, quindi appartiene a [-1, +1] e non
contiene Wasserstein. I segnali RAW sono i suoi cambi di segno; i CONFIRMED
richiedono anche intensità Wasserstein, coda e velocità coerenti.

Non è il codice proprietario dello script TradingView STARK sigma.3 e non ne
pretende la replica esatta. Il risultato è uno score diagnostico, non una
probabilità calibrata e non costituisce consulenza finanziaria.

Esecuzione in Spyder:
1. Modificare la sezione CONFIGURAZIONE UTENTE.
2. Premere F5 (oppure eseguire le celle # %% in ordine).
3. Il grafico resta nel pannello Plot e nelle variabili FIGURE/RESULTS del
   Variable Explorer. Non viene scritto alcun file su disco.
   Con DATA_SOURCE = "demo" non serve Internet.
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


# %% CONFIGURAZIONE UTENTE
# Valori possibili: "demo", "csv", "yfinance"
DATA_SOURCE = "demo"

# Usato soltanto con DATA_SOURCE = "csv".
# Colonne accettate (senza distinzione maiuscole/minuscole):
# Date, Open, High, Low, Close, Adj Close, Volume.
CSV_PATH = ""

# Usati soltanto con DATA_SOURCE = "yfinance".
TICKER = "SPY"
YFINANCE_PERIOD = "5y"
YFINANCE_INTERVAL = "1d"

# Scelta prudenziale: Yahoo può includere la candela corrente non ancora chiusa.
# Con False, le metriche vengono mostrate ma RAW/CONFIRMED sull'ultima riga
# vengono soppressi perché la chiusura non è ancora verificata.
YFINANCE_LAST_BAR_IS_COMPLETE = False

# Visualizzazione esclusivamente dentro Spyder: nessun CSV, PNG o cartella.
SHOW_FIGURE = True
MAX_BARS_IN_CHART = 360


@dataclass(frozen=True)
class IndicatorConfig:
    """Parametri del modello, tutti applicati in modo causale."""

    # Numero di rendimenti nella distribuzione corrente e di riferimento.
    window: int = 64

    # La distribuzione di riferimento termina `transport_lag` barre prima.
    # Con un valore inferiore a window le due finestre si sovrappongono.
    transport_lag: int = 8

    # Frazione di osservazioni usata per ciascuna coda (64 * 0.15 ~= 10).
    tail_fraction: float = 0.15

    # Griglia di quantili della direzione di trasporto. Gli estremi sono esclusi
    # per limitare l'instabilità delle code con campioni piccoli.
    quantile_min: float = 0.10
    quantile_max: float = 0.90
    quantile_count: int = 17

    # Uno spostamento quantile, espresso in unità della scala robusta della
    # reference, vota +1/-1 solo oltre questa deadband. QTD resta indipendente
    # dalla distanza Wasserstein.
    quantile_direction_deadband: float = 0.02

    # Smoothing causale EWMA.
    tail_ewm_span: int = 8
    velocity_ewm_span: int = 5
    score_ewm_span: int = 5

    # Baseline storica per gli z-score robusti. La baseline è shiftata di 1.
    normalization_window: int = 252
    normalization_min_periods: int = 63
    zscore_clip: float = 5.0

    # Pesi dell'intensità distributiva (somma consigliata = 1).
    wasserstein_weight: float = 0.70
    velocity_weight: float = 0.30

    # Conferma causale: dopo un RAW zero-cross, i filtri possono confermare sulla
    # stessa barra o entro le successive confirmation_window-1 barre. Il segnale
    # viene registrato sulla barra della conferma, mai retrodatato.
    confirmation_window: int = 8
    confirmation_min_direction: float = 0.25
    confirmation_min_wasserstein_z: float = 0.50
    confirmation_min_tail_z: float = 0.15
    confirmation_min_velocity_z: float = 0.25

    # Isteresi STABLE / TRANSITION. L'ingresso in TRANSITION usa le soglie di
    # conferma; il ritorno a STABLE richiede valori più bassi.
    stable_max_direction: float = 0.15
    stable_max_wasserstein_z: float = 0.25
    stable_max_tail_z: float = 0.75
    stable_max_velocity_z: float = 0.25

    # Limite inferiore della scala robusta: evita divisioni per zero senza
    # nascondere il primo shock dopo una finestra piatta o quasi piatta.
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
                raise ValueError(f"{name} deve essere un numero intero.")
        if self.window < 20:
            raise ValueError("window deve essere almeno 20.")
        if self.transport_lag < 1:
            raise ValueError("transport_lag deve essere almeno 1.")
        if not 0.05 <= self.tail_fraction <= 0.30:
            raise ValueError("tail_fraction deve essere compreso tra 0.05 e 0.30.")
        if math.ceil(self.window * self.tail_fraction) < 5:
            raise ValueError(
                "La coda contiene meno di 5 osservazioni: aumentare window "
                "o tail_fraction. Sono consigliate almeno 8-10 osservazioni."
            )
        if not 0 < self.quantile_min < 0.5 < self.quantile_max < 1:
            raise ValueError(
                "quantile_min e quantile_max devono circondare 0.5 e stare in (0, 1)."
            )
        if not np.isclose(self.quantile_min + self.quantile_max, 1.0):
            raise ValueError("La griglia quantile deve essere simmetrica rispetto a 0.5.")
        if self.quantile_count < 5 or self.quantile_count % 2 == 0:
            raise ValueError("quantile_count deve essere dispari e almeno 5.")
        if (
            not np.isfinite(self.quantile_direction_deadband)
            or self.quantile_direction_deadband < 0
        ):
            raise ValueError("quantile_direction_deadband deve essere finita e >= 0.")
        for name in ("tail_ewm_span", "velocity_ewm_span", "score_ewm_span"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} deve essere almeno 1.")
        if self.normalization_min_periods < 20:
            raise ValueError("normalization_min_periods deve essere almeno 20.")
        if self.normalization_window < self.normalization_min_periods:
            raise ValueError(
                "normalization_window non può essere minore di "
                "normalization_min_periods."
            )
        if not np.isfinite(self.zscore_clip) or self.zscore_clip <= 0:
            raise ValueError("zscore_clip deve essere positivo e finito.")
        weight_names = (
            "wasserstein_weight",
            "velocity_weight",
        )
        for name in weight_names:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} deve essere finito e non negativo.")
        if not np.isclose(self.wasserstein_weight + self.velocity_weight, 1.0):
            raise ValueError("I pesi di intensità devono sommare a 1.")
        if self.confirmation_window < 1:
            raise ValueError("confirmation_window deve essere almeno 1.")
        unit_interval_fields = (
            "confirmation_min_direction",
            "stable_max_direction",
        )
        for name in unit_interval_fields:
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} deve essere finito e in [0, 1].")
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
                raise ValueError(f"{name} deve rientrare nel range degli z-score.")
        if self.stable_max_direction >= self.confirmation_min_direction:
            raise ValueError(
                "stable_max_direction deve essere minore di confirmation_min_direction."
            )
        if self.stable_max_wasserstein_z >= self.confirmation_min_wasserstein_z:
            raise ValueError(
                "stable_max_wasserstein_z deve essere minore della soglia di conferma."
            )
        if self.stable_max_velocity_z > self.confirmation_min_velocity_z:
            raise ValueError(
                "stable_max_velocity_z non può superare la soglia velocity "
                "di conferma."
            )
        if not np.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor deve essere positivo e finito.")


CONFIG = IndicatorConfig()
FIGURE: plt.Figure | None = None


# %% DATI
def _canonical_column_map(columns: pd.Index) -> dict[str, str]:
    """Restituisce {nome_normalizzato: nome_originale}."""
    return {
        re.sub(r"[^a-z0-9]+", "", str(column).strip().lower()): str(column)
        for column in columns
    }


def sanitize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Valida e uniforma un DataFrame OHLCV senza inventare rendimenti."""
    if raw is None or raw.empty:
        raise ValueError("Il dataset è vuoto.")

    data = raw.copy()
    columns = _canonical_column_map(data.columns)

    # Se la data è una colonna, la trasforma nell'indice.
    date_key = next(
        (key for key in ("date", "datetime", "timestamp", "time") if key in columns),
        None,
    )
    if date_key is not None:
        if pd.api.types.is_numeric_dtype(data[columns[date_key]].dtype):
            raise ValueError(
                "La colonna data è numerica e l'unità epoch sarebbe ambigua. "
                "Convertirla prima in date ISO-8601 (per esempio "
                "2026-08-22T16:00:00Z)."
            )
        parsed = pd.to_datetime(data[columns[date_key]], errors="coerce", utc=True)
        if bool(parsed.isna().any()):
            raise ValueError(
                f"La colonna data contiene {int(parsed.isna().sum())} valori "
                "non interpretabili. Correggerli prima del calcolo."
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
                "Manca una vera colonna Date/Datetime. Un indice numerico non "
                "viene interpretato automaticamente come data."
            )
        parsed_index = pd.to_datetime(data.index, errors="coerce", utc=True)
        valid = ~pd.isna(parsed_index)
        if not bool(valid.all()):
            raise ValueError(
                f"L'indice contiene {int((~valid).sum())} date non interpretabili."
            )
        data = data.copy()
        data.index = pd.DatetimeIndex(parsed_index).tz_convert(None)
        data.index.name = "Date"

    if "close" not in columns:
        raise ValueError("Manca la colonna Close nel dataset.")

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

    # Permette file con il classico nome "Adj Close" oppure "Adjusted Close".
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

    # Per un file close-only rende comunque possibile il grafico.
    for column in ("Open", "High", "Low"):
        if column not in standardized:
            standardized[column] = standardized["Close"]
        else:
            # Serve soltanto a rendere disegnabile una barra con OHLC parziale;
            # AnalysisClose e quindi i rendimenti non vengono riempiti.
            standardized[column] = standardized[column].fillna(standardized["Close"])
    if "Volume" not in standardized:
        standardized["Volume"] = 0.0

    standardized = standardized[
        ["Open", "High", "Low", "Close", "Volume", "AnalysisClose"]
    ]
    standardized = standardized.replace([np.inf, -np.inf], np.nan)
    standardized = standardized[~standardized.index.duplicated(keep="last")]
    standardized = standardized.sort_index()

    # Non si fa backfill/forward-fill: una quotazione assente resta assente.
    valid_price = (
        standardized[["Close", "AnalysisClose"]].notna().all(axis=1)
        & (standardized["Close"] > 0)
        & (standardized["AnalysisClose"] > 0)
    )
    if not bool(valid_price.all()):
        invalid_count = int((~valid_price).sum())
        raise ValueError(
            f"Il dataset contiene {invalid_count} righe con Close/Adj Close "
            "mancante, infinito o non positivo. Correggerle o rimuoverle "
            "esplicitamente prima del calcolo; il programma non attraversa "
            "silenziosamente questi gap."
        )
    standardized["Volume"] = standardized["Volume"].fillna(0.0).clip(lower=0.0)

    if len(standardized) < 3:
        raise ValueError("Servono almeno 3 prezzi validi.")
    return standardized


def build_demo_data(n_bars: int = 950, seed: int = 73) -> pd.DataFrame:
    """Crea dati OHLCV deterministici con cambi noti di regime e di coda."""
    if n_bars < 250:
        raise ValueError("Per la demo servono almeno 250 barre.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_bars)

    # Cinque regimi per rendere visibili shift di media, varianza e skewness.
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
        raise FileNotFoundError(f"File CSV non trovato: {path}")
    return sanitize_ohlcv(pd.read_csv(path))


def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # dipendenza opzionale
    except ImportError as exc:
        raise ImportError(
            "Per DATA_SOURCE='yfinance' installare yfinance con: "
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
            f"Nessun dato ricevuto per {symbol!r}. Controllare simbolo, periodo, "
            "intervallo e connessione."
        )
    return sanitize_ohlcv(history)


def load_data() -> tuple[pd.DataFrame, str]:
    source = DATA_SOURCE.strip().lower()
    if source == "demo":
        return build_demo_data(), "DEMO"
    if source == "csv":
        if not CSV_PATH.strip():
            raise ValueError("Impostare CSV_PATH quando DATA_SOURCE='csv'.")
        return load_csv(CSV_PATH), Path(CSV_PATH).stem
    if source == "yfinance":
        return (
            load_yfinance(TICKER, YFINANCE_PERIOD, YFINANCE_INTERVAL),
            TICKER.upper(),
        )
    raise ValueError("DATA_SOURCE deve essere 'demo', 'csv' oppure 'yfinance'.")


# %% CALCOLO DEL PROBABILITY FLOW
def _robust_reference_scale(values: np.ndarray, floor: float) -> float:
    """Scala robusta della sola reference, limitata inferiormente da floor."""
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
    """Metriche OT 1D con direzione multi-quantile separata da Wasserstein.

    Per campioni empirici equiprobabili e della stessa lunghezza, l'accoppiamento
    monotono tra statistiche d'ordine fornisce la Wasserstein-1 esatta. La QTD
    usa invece la massa netta dei quantili spostati verso l'alto o verso il
    basso: non viene mai moltiplicata per Wasserstein.
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
        raise ValueError("Le tre finestre devono avere la stessa lunghezza.")
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

    # Quantile Transport Direction (QTD): ciascun quantile ha la stessa massa.
    # Il voto grezzo usa il solo segno. Il voto operativo assegna 0 agli
    # spostamenti entro la deadband causale, espressa nella scala della reference.
    reference_quantiles = np.quantile(reference, quantile_levels, method="linear")
    current_quantiles = np.quantile(current, quantile_levels, method="linear")
    quantile_displacement = (current_quantiles - reference_quantiles) / scale

    # Un residuo floating-point attorno allo zero non deve valere come massa
    # trasportata (per esempio la mediana di un'espansione simmetrica).
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

    # Tail asymmetry della forma: la mediana viene rimossa, così una semplice
    # traslazione o un'espansione simmetrica non viene scambiata per skew di coda.
    k = max(1, int(math.ceil(len(current) * tail_fraction)))

    def tail_balance(sorted_values: np.ndarray) -> float:
        center = float(np.median(sorted_values))
        upper_extent = float(np.mean(sorted_values[-k:] - center))
        lower_extent = float(np.mean(center - sorted_values[:k]))
        return upper_extent - lower_extent

    tail_change_raw = tail_balance(cur_sorted) - tail_balance(ref_sorted)

    # Velocità distributiva effettiva tra la finestra t-1 e quella t.
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
    """Z-score mediana/MAD con baseline composta solo da barre precedenti."""
    history = series.shift(1)
    rolling = history.rolling(lookback, min_periods=min_periods)
    center = rolling.median()
    mad = rolling.apply(
        # rolling.apply può includere NaN appartenenti al warmup iniziale anche
        # quando min_periods è soddisfatto; nanmedian mantiene il significato
        # dichiarato di normalization_min_periods.
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
    )
    mad_scale = 1.4826 * mad
    iqr_scale = (rolling.quantile(0.75) - rolling.quantile(0.25)) / 1.349
    std_scale = rolling.std(ddof=0)
    robust_scale = pd.concat([mad_scale, iqr_scale], axis=1).max(
        axis=1, skipna=True
    )
    # La deviazione standard è soltanto un fallback quando le due scale robuste
    # collassano; non può dominare la baseline in presenza di un singolo outlier.
    # Se anche la baseline è perfettamente piatta, epsilon rende comunque
    # osservabile il primo scostamento (che verrà poi limitato da `clip`).
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
    """Scala causale che conserva zero e il segno come riferimento naturale.

    È usata per componenti direzionali: diversamente da uno z-score centrato
    sulla mediana storica, un valore positivo non può diventare negativo solo
    perché è inferiore alla propria baseline recente.
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
    # RMS è un fallback, non un concorrente delle scale robuste. Anche qui il
    # floor conserva il primo segnale non nullo dopo una baseline piatta.
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
    """Costruisce stati e segnali in ordine strettamente causale.

    RAW_BUY/RAW_SELL sono zero-cross della QTD significativa. Una conferma può
    arrivare sulla stessa barra o entro `confirmation_window`, ma viene sempre
    registrata sulla barra in cui tutti i filtri sono realmente soddisfatti.
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

            # La conferma è valutata anche sulla barra RAW (lag 0), poi sulle
            # successive. Nessuna conferma viene retrodatata.
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
            # TRANSITION rappresenta attività distributiva apprezzabile, non
            # semplicemente un segno direzionale. La coda, più rumorosa, apre
            # lo stato solo se concorda con una direzione già significativa.
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
        raise RuntimeError("Errore interno nell'allineamento di stati e segnali.")

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

    # Compatibilità con il precedente schema: SignalAtClose e ActionNextBar
    # rappresentano esclusivamente i segnali CONFERMATI.
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
    """Calcola tutte le metriche e restituisce un DataFrame allineato ai prezzi."""
    config.validate()
    data = sanitize_ohlcv(ohlcv)
    returns = np.log(data["AnalysisClose"] / data["AnalysisClose"].shift(1))

    required = config.window + config.transport_lag + config.normalization_min_periods
    if returns.notna().sum() < required:
        raise ValueError(
            f"Dati insufficienti: servono circa {required + 1} prezzi; "
            f"sono disponibili {len(data)} righe."
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

    # Indici: current contiene [i-window+1, ..., i]; reference termina a i-lag.
    # LogReturn[0] è NaN, quindi il primo riferimento interamente valido inizia
    # alla posizione 1 e produce il primo output a window + transport_lag.
    first_position = config.window + config.transport_lag
    for i in range(first_position, len(data)):
        current_start = i - config.window + 1
        current = return_values[current_start : i + 1]

        reference_end = i - config.transport_lag + 1  # esclusivo
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

    # Alias compatibili con la versione precedente. TransportDirection ora è
    # la vera direzione multi-quantile; SignedTransport è intenzionalmente rimosso.
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

    # La linea direzionale non contiene W1, tail o velocity: è esclusivamente
    # la QTD multi-quantile, smussata causalmente per la visualizzazione.
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

    # Confidence combina breadth direzionale e supporto dei tre filtri; non è
    # una probabilità di profitto o di direzione futura.
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


# %% GRAFICO
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
    """Crea dashboard a tre pannelli: prezzo, volumi e probability flow."""
    if max_bars < 50:
        raise ValueError("max_bars deve essere almeno 50.")
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
        f"{label}  |  Probability Flow Momentum — modello indipendente",
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
        label="Asimmetria z",
    )
    ax_flow.plot(
        x,
        view["TransportVelocityZ"],
        color=PALETTE["velocity"],
        linewidth=1.0,
        alpha=0.72,
        label="Velocità z",
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

    # RAW: marker piccolo e vuoto sullo zero-cross QTD.
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

    # CONFIRMED: marker pieno e label sulla barra reale di conferma.
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
        "Segnale calcolato alla chiusura; eventuale azione non prima della barra successiva. "
        "Uso sperimentale, non consulenza finanziaria.",
        fontsize=7.5,
        color=PALETTE["text"],
    )
    fig.text(
        0.988,
        0.008,
        "Sfondo: STABLE / TRANSITION  ·  △ RAW  ·  ▲ CONFIRMED",
        ha="right",
        fontsize=7.5,
        color=PALETTE["text"],
    )
    fig.subplots_adjust(left=0.045, right=0.955, top=0.955, bottom=0.055)

    if show:
        plt.show()
    return fig


# %% ESECUZIONE IN MEMORIA
def _apply_bar_status(
    result: pd.DataFrame,
    source: str,
    yfinance_last_bar_is_complete: bool,
) -> pd.DataFrame:
    """Marca l'ultima barra Yahoo senza renderla evento o stato operativo."""
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
        # Le ActionNextBar derivano dalla barra precedente e restano intatte:
        # possono essere azioni valide sulla barra corrente.
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
        warnings.warn("Calcolo completato, ma non ci sono ancora valori post-warmup.")
    else:
        latest = valid.iloc[-1]
        print("\nProbability Flow Momentum — riepilogo")
        print(f"Strumento: {label}")
        print(f"Ultima barra: {valid.index[-1]}")
        print(f"Stato: {latest['MarketState']}")
        print(f"Regime direzionale: {latest['DirectionalRegime']}")
        print(
            "Quantile Transport Direction: "
            f"{latest['QuantileTransportDirection']:+.3f}"
        )
        print(f"Intensità: {latest['TransitionIntensity']:.2f}")
        print(
            "Wasserstein normalizzata: "
            f"{latest['WassersteinNormalized']:.4f}"
        )
        if latest["RawSignalAtClose"]:
            print(f"Segnale RAW: {latest['RawSignalAtClose']}")
        if latest["ConfirmedSignalAtClose"]:
            print(f"Segnale CONFIRMED: {latest['ConfirmedSignalAtClose']}")
        if latest["BarStatus"] == "PROVISIONAL":
            print(
                "ATTENZIONE: ultima barra Yahoo marcata PROVISIONAL; "
                "nessun RAW/CONFIRMED viene emesso su questa riga."
            )

    print(
        "Nessun file salvato: risultati in RESULTS e grafico in FIGURE/Plot."
    )
    return result


if __name__ == "__main__":
    RESULTS = main()
